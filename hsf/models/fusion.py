"""Fusion modules for the dual-stream model.

Interface (one module per scale P3/P4/P5):
    forward(evt, rgb, h) -> (evt_out, rgb_out, neck_in, h_new)
        evt, rgb  : backbone features of the two streams at this scale (B, C, H, W)
        h         : hidden state from the previous time step (temporal fusion only), else None
        evt_out   : feature that continues in the event backbone
        rgb_out   : feature that continues in the RGB backbone
        neck_in   : fused feature for the shared neck (C channels)
        h_new     : new hidden state (or None)

'add'  - HSF ablation baseline (Tab. II, first row): element-wise addition, streams unchanged.
'af'   - Adaptive Fusion (Sec. III-D, Fig. 4)
'tfs'  - Temporal Feature Selection (Sec. III-C, Fig. 3) on the event stream + addition
'hsf'  - full HSF: TFS then AF
Stateful modules (tfs, hsf) carry a hidden state h per scale across consecutive frames.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


class AddFusion(nn.Module):
    def __init__(self, c):
        super().__init__()

    def forward(self, evt, rgb, h=None):
        return evt, rgb, evt + rgb, None


class AdaptivePool(nn.Module):
    """Eq. 10: F_o = lambda_a * AvgPool(F) + lambda_m * MaxPool(F), lambdas learnable in (0, 1)."""

    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.lambda_a = nn.Parameter(torch.zeros(1))     # sigmoid(0) = 0.5
        self.lambda_m = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return (torch.sigmoid(self.lambda_a) * F.adaptive_avg_pool2d(x, self.grid)
                + torch.sigmoid(self.lambda_m) * F.adaptive_max_pool2d(x, self.grid))


class AFBlock(nn.Module):
    """Transformer block with adaptive residual weights (Eq. 11-12)."""

    def __init__(self, c, heads=8, mlp_ratio=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(c, c * mlp_ratio), nn.GELU(), nn.Linear(c * mlp_ratio, c))
        # learnable residual weights; 1.0 everywhere = a standard pre-norm transformer block at init
        self.lambda_sa1, self.lambda_sa2 = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        self.lambda_mlp1, self.lambda_mlp2 = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))

    def forward(self, f):
        y = self.ln1(f)
        f = self.lambda_sa1 * self.attn(y, y, y, need_weights=False)[0] + self.lambda_sa2 * f      # Eq. 11
        f = self.lambda_mlp1 * self.mlp(self.ln2(f)) + self.lambda_mlp2 * f                     # Eq. 12
        return f


class AFFusion(nn.Module):
    """Adaptive Fusion module (HSF Sec. III-D, Fig. 4), a lighter Cross-Modality Fusion Transformer.

    1. each modality is reduced to a grid x grid token map by adaptive mixed pooling,
    2. event and RGB tokens are concatenated, a learnable position embedding is added, and one
       transformer block with adaptive residual weights mixes them (cross-modal self-attention),
    3. the token maps are upsampled back and added to the corresponding stream ("⊕" in Fig. 2),
    4. the two updated streams are concatenated and a 1x1 conv gives the neck input ("©" + Conv).
    Not specified in the paper (our choice): grid 8x8 (as in CFT), one block, 8 heads, MLP ratio 4,
    and zero-initialised learnable gates on the three places where new features enter:
        evt' = evt + lambda_e * up(fe),  rgb' = rgb + lambda_v * up(fv),
        neck = evt' + rgb' + lambda_out * Conv([evt', rgb'])
    so at initialisation AF equals the 'add' baseline and the pretrained single-stream backbones,
    neck and head stay valid (with random gates the neck receives random features and the model
    has to relearn from near zero mAP). Training then decides how much of AF to use.
    """

    def __init__(self, c, grid=8, heads=8, blocks=1):
        super().__init__()
        self.grid = grid
        self.pool_e, self.pool_v = AdaptivePool(grid), AdaptivePool(grid)
        self.pos = nn.Parameter(torch.zeros(1, 2 * grid * grid, c))
        self.blocks = nn.Sequential(*[AFBlock(c, heads) for _ in range(blocks)])
        self.ln = nn.LayerNorm(c)
        self.out = Conv(2 * c, c, 1)                          # Conv + BN + SiLU (Ultralytics)
        self.lambda_e = nn.Parameter(torch.zeros(1))
        self.lambda_v = nn.Parameter(torch.zeros(1))
        self.lambda_out = nn.Parameter(torch.zeros(1))

    def forward(self, evt, rgb, h=None):
        b, c, H, W = evt.shape
        n = self.grid * self.grid
        te = self.pool_e(evt).flatten(2).transpose(1, 2)      # (B, n, C)
        tv = self.pool_v(rgb).flatten(2).transpose(1, 2)
        t = self.ln(self.blocks(torch.cat([te, tv], 1) + self.pos))
        fe = t[:, :n].transpose(1, 2).reshape(b, c, self.grid, self.grid)
        fv = t[:, n:].transpose(1, 2).reshape(b, c, self.grid, self.grid)
        evt = evt + self.lambda_e * F.interpolate(fe, size=(H, W), mode='bilinear', align_corners=False)
        rgb = rgb + self.lambda_v * F.interpolate(fv, size=(H, W), mode='bilinear', align_corners=False)
        neck = evt + rgb + self.lambda_out * self.out(torch.cat([evt, rgb], 1))
        return evt, rgb, neck, None


class TFSCell(nn.Module):
    """Temporal Feature Selection (HSF Sec. III-C, Fig. 3, Eq. 6-9): a convolutional GRU over the
    event features whose gates also look at the RGB features.
        z_t  = sigmoid(W_z [h_{t-1}, x_evt, x_rgb])          update gate        (Eq. 6)
        r_t  = sigmoid(W_r [h_{t-1}, x_evt, x_rgb])          reset gate         (Eq. 7)
        h~_t = tanh(W [r_t * h_{t-1}, x_evt])                 candidate          (Eq. 8)
        h_t  = (1 - z_t) * h_{t-1} + z_t * h~_t               new state / output (Eq. 9)
    W_z, W_r, W are 1x1 convolutions. Our addition (not in the paper): the event stream continues
    with x_evt + lambda_t * h_t where lambda_t starts at 0, so the pretrained event backbone is
    undisturbed at initialisation (same reasoning as the gates in AFFusion).
    """

    def __init__(self, c):
        super().__init__()
        self.w_z = nn.Conv2d(3 * c, c, 1)
        self.w_r = nn.Conv2d(3 * c, c, 1)
        self.w_h = nn.Conv2d(2 * c, c, 1)
        self.lambda_t = nn.Parameter(torch.zeros(1))

    def forward(self, evt, rgb, h):
        if h is None:
            h = torch.zeros_like(evt)
        zr_in = torch.cat([h, evt, rgb], 1)
        z = torch.sigmoid(self.w_z(zr_in))
        r = torch.sigmoid(self.w_r(zr_in))
        h_cand = torch.tanh(self.w_h(torch.cat([r * h, evt], 1)))
        h_new = (1 - z) * h + z * h_cand
        return evt + self.lambda_t * h_new, h_new


class TFSFusion(nn.Module):
    """TFS on the event stream, then element-wise addition (ablation row "TFS only")."""
    stateful = True

    def __init__(self, c):
        super().__init__()
        self.tfs = TFSCell(c)

    def forward(self, evt, rgb, h=None):
        evt, h = self.tfs(evt, rgb, h)
        return evt, rgb, evt + rgb, h


class HSFFusion(nn.Module):
    """Full HSF fusion at one scale: TFS on the event stream, then AF (Fig. 2)."""
    stateful = True

    def __init__(self, c):
        super().__init__()
        self.tfs = TFSCell(c)
        self.af = AFFusion(c)

    def forward(self, evt, rgb, h=None):
        evt, h = self.tfs(evt, rgb, h)
        evt, rgb, neck, _ = self.af(evt, rgb)
        return evt, rgb, neck, h


ILLUM_DIM = 16 + 48          # luminance histogram (16 bins) + 4x4 mean colour


def illum_descriptor(rgb):
    """Global lighting descriptor of the RGB network input (B, 3, H, W) in [0, 1] -> (B, 64):
    a 16-bin luminance histogram and the 4x4 average colour. Not learnable; the letterbox padding
    is constant grey and simply occupies a fixed share of one bin."""
    rgb = rgb.float()
    lum = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    small = F.avg_pool2d(lum, 4).flatten(1)                                   # (B, N)
    idx = (small.clamp(0, 0.9999) * 16).long()
    hist = torch.zeros(rgb.shape[0], 16, device=rgb.device).scatter_add_(1, idx, torch.ones_like(small))
    hist = hist / small.shape[1]
    col = F.adaptive_avg_pool2d(rgb, 4).flatten(1)                            # (B, 48)
    return torch.cat([hist, col], 1)


class IllumGate(nn.Module):
    """Illumination-aware fusion (our extension, not in HSF): wraps a fusion module and decides per
    image and per channel how much of the fused contribution reaches the neck and the RGB stream:
        g        = 2 * sigmoid(MLP([illum descriptor, GAP(evt), GAP(rgb)]))   in (0, 2), (B, C)
        neck     = rgb + g * (neck_base - rgb)       (add: rgb + g * evt)
        rgb_out  = rgb + g * (rgb_base - rgb)        (AF's injection into the RGB stream)
    The MLP's last layer starts at zero, so g = 1 and the model equals the wrapped fusion at
    initialisation. keep (B,) from the model zeroes g for modality-dropout samples (pure RGB path),
    which teaches the neck to also work at g = 0 so the gate can really switch the event stream off
    in good light. last_gate keeps the per-image mean of g for analysis."""

    def __init__(self, base, c, hidden=64):
        super().__init__()
        self.base = base
        self.stateful = bool(getattr(base, 'stateful', False))
        self.mlp = nn.Sequential(nn.LayerNorm(ILLUM_DIM + 2 * c), nn.Linear(ILLUM_DIM + 2 * c, hidden),
                                 nn.SiLU(), nn.Linear(hidden, c))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.last_gate = None

    def forward(self, evt, rgb, h=None, illum=None, keep=None):
        evt2, rgb2, neck, h = self.base(evt, rgb, h)
        z = torch.cat([illum, evt.float().mean((2, 3)), rgb.float().mean((2, 3))], 1)
        g = 2 * torch.sigmoid(self.mlp(z))
        if keep is not None:
            g = g * keep[:, None]
        self.last_gate = g.detach().mean(1)
        g = g[:, :, None, None].to(neck.dtype)
        return evt2, rgb + g * (rgb2 - rgb), rgb + g * (neck - rgb), h


# ----------------------------------------------------------------------------------------------------
# Reliability-weighted fusion (docs/fusion_model.md): both streams are noisy observations of one latent
# feature; the ML / MMSE combination weights each by its precision 1/sigma^2. The precisions are
# predicted from sensor physics measured on the raw inputs (RGB contrast-to-noise, event Poisson SNR,
# saturation) plus a learned residual.
# ----------------------------------------------------------------------------------------------------
GEAR_C = 8.0                     # gray levels per event in GEAR (hsf/data/gear.py)
PAD = 114.0 / 255.0              # letterbox fill of both inputs (hsf/data/dataset.py RGB_FILL / EVT_FILL)
STRIDES = (8, 16, 32)            # P3 / P4 / P5


@torch.no_grad()
def reliability_signals(rgb, evt):
    """Full-resolution inputs of the reliability gate, computed without gradients in float32.
    rgb (B, 3, S, S) in [0, 1]; evt (B, 3, S, S) = [num, feat, h] as fed to the network,
    num = tanh(n * c / 255) with n events per pixel in the frame's window.
    Returns y (B,1,S,S) luma, n (B,1,S,S) events/pixel (0 in the padding), valid (B,1,S,S) mask,
    sigma_n (B,1,1,1) RGB noise std (Immerkaer 1996: Laplacian-difference MAD), n_bg (B,1,1,1)
    mean events/pixel over valid pixels (background rate: objects cover a small share of the frame)."""
    rgb, evt = rgb.float(), evt.float()
    valid = ((evt - PAD).abs().amax(1, keepdim=True) > 2e-3).float()          # padding: all channels = 114
    y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    k = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]], device=y.device).view(1, 1, 3, 3)
    lap = F.conv2d(y, k, padding=1).abs() * valid
    sigma_n = (math.sqrt(math.pi / 2) / 6) * lap.sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp(min=1)
    n = torch.atanh(evt[:, 0:1].clamp(0, 0.999)) * (255.0 / GEAR_C) * valid
    n_bg = n.sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp(min=1)
    return {'y': y, 'n': n, 'valid': valid, 'sigma_n': sigma_n.clamp(min=1e-3), 'n_bg': n_bg}


class ReliabilityGate(nn.Module):
    """Precision-weighted fusion at one scale (docs/fusion_model.md, Sec. 2-3):
        per cell of the feature map (stride s):
            CNR_r = sigma_loc(y) / sigma_n                         RGB contrast-to-noise
            SNR_e = relu(n - n_bg) / sqrt(n + eps)                 event Poisson SNR
            sat   = share of saturated pixels (y > 0.98)
          (both ratios are per pixel; over the s x s cell each grows by the same factor s, which
           cancels in the difference of the logs)
        l = a * log(CNR_r + e) - b * log(SNR_e + e) + c * log(1 - sat + e) + d + delta(features)
        w = sigmoid(l)                                             RGB weight = pi_r / (pi_r + pi_e)
        neck = w * rgb + (1 - w) * P(neck_base - rgb)              (add: neck_base - rgb = evt)
    P is a 1x1 conv mapping event features into the RGB feature space (identity at init); a, b, c, d
    (lambda_cnr, lambda_snr, lambda_sat, lambda_bias: named lambda_* so that build_optimizer applies no
    weight decay) start from the physical prior (2, 2, 1, d0); delta is a small conv on both features,
    zero at init.
    Padding cells (no valid pixel) get w = 1. The streams continue as returned by the base fusion.
    last_gate: per-image mean of w over valid cells (for tools/eval_conditions.py)."""

    def __init__(self, base, c, scale_idx, prior=(2.0, 2.0, 1.0, 1.0), eps=0.1, learn_delta=True):
        super().__init__()
        self.base = base
        self.stateful = bool(getattr(base, 'stateful', False))
        self.stride = STRIDES[scale_idx]
        self.eps = eps
        a, b, cc, d = prior
        self.lambda_cnr, self.lambda_snr = nn.Parameter(torch.tensor(a)), nn.Parameter(torch.tensor(b))
        self.lambda_sat, self.lambda_bias = nn.Parameter(torch.tensor(cc)), nn.Parameter(torch.tensor(d))
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(c).view(c, c, 1, 1))
        self.delta = None
        if learn_delta:
            self.delta = nn.Sequential(nn.Conv2d(2 * c, c // 4, 1), nn.SiLU(), nn.Conv2d(c // 4, 1, 3, padding=1))
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
        self.last_gate = None
        self.last_terms = None

    def log_terms(self, sig, size):
        pool = lambda x: F.adaptive_avg_pool2d(x, size)     # noqa: E731  (= s x s average for S = s * H)
        v = pool(sig['valid'])
        vs = v.clamp(min=1e-3)
        mu = pool(sig['y'] * sig['valid']) / vs
        var = (pool(sig['y'] ** 2 * sig['valid']) / vs - mu ** 2).clamp(min=0)
        cnr = var.sqrt() / sig['sigma_n']
        sat = pool((sig['y'] > 0.98).float() * sig['valid']) / vs
        n = pool(sig['n']) / vs
        snr = F.relu(n - sig['n_bg']) / (n + 1e-2).sqrt()
        e = self.eps
        return torch.log(cnr + e), torch.log(snr + e), torch.log(1 - sat + e), v

    def forward(self, evt, rgb, h=None, sig=None):
        evt2, rgb2, neck_base, h = self.base(evt, rgb, h)
        lc, ls, lsat, v = self.log_terms(sig, rgb.shape[-2:])
        logit = self.lambda_cnr * lc - self.lambda_snr * ls + self.lambda_sat * lsat + self.lambda_bias
        if self.delta is not None:
            logit = logit + self.delta(torch.cat([rgb, evt], 1)).float()
        w = torch.sigmoid(logit)
        w = torch.where(v > 0, w, torch.ones_like(w))                          # padding: RGB path only
        self.last_gate = ((w * (v > 0)).sum((1, 2, 3)) / (v > 0).sum((1, 2, 3)).clamp(min=1)).detach()
        if not self.training:                    # statistics of the event weight 1 - w over valid cells
            e = (1 - w)[v > 0]
            self.last_terms = {'evt_w_mean': e.mean().item(), 'evt_w_p99': e.quantile(0.99).item() if e.numel() < 2e6
                               else e[::8].quantile(0.99).item(), 'evt_w_gt50': (e > 0.5).float().mean().item()}
        w = w.to(rgb.dtype)
        neck = w * rgb + (1 - w) * self.proj(neck_base - rgb)
        return evt2, rgb2, neck, h

    def describe(self):
        t = self.last_terms or {}
        return (f's{self.stride}: a {self.lambda_cnr.item():.2f} b {self.lambda_snr.item():.2f} '
                f'c {self.lambda_sat.item():.2f} d {self.lambda_bias.item():.2f}'
                + ''.join(f'  {k} {v:.4f}' for k, v in t.items()))


FUSIONS = {'add': AddFusion, 'af': AFFusion, 'tfs': TFSFusion, 'hsf': HSFFusion}


def build_fusion(name, c, scale_idx=0):
    """'iaf_<base>' = <base> fusion wrapped in the illumination-aware gate (e.g. iaf_add, iaf_af);
    'rel_<base>' = <base> wrapped in the reliability-weighted gate (docs/fusion_model.md);
    'relp_<base>' = same with the physical prior only (no learned residual delta)."""
    for prefix in ('iaf_', 'relp_', 'rel_'):
        if name.startswith(prefix):
            base = name[len(prefix):]
            break
    else:
        prefix, base = '', name
    if base not in FUSIONS:
        raise ValueError(f'unknown fusion "{name}" (available: {list(FUSIONS)}, iaf_/rel_/relp_<name>)')
    m = FUSIONS[base](c)
    if prefix == 'iaf_':
        return IllumGate(m, c)
    if prefix in ('rel_', 'relp_'):
        return ReliabilityGate(m, c, scale_idx, learn_delta=prefix == 'rel_')
    return m
