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
    mean events/pixel over valid pixels (v1 background rate), n_floor (B,1,1,1) noise floor of the
    event count (v2)."""
    rgb, evt = rgb.float(), evt.float()
    valid = ((evt - PAD).abs().amax(1, keepdim=True) > 2e-3).float()          # padding: all channels = 114
    y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    k = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]], device=y.device).view(1, 1, 3, 3)
    lap = F.conv2d(y, k, padding=1).abs() * valid
    sigma_n = (math.sqrt(math.pi / 2) / 6) * lap.sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp(min=1)
    n = torch.atanh(evt[:, 0:1].clamp(0, 0.999)) * (255.0 / GEAR_C) * valid
    n_bg = n.sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp(min=1)
    # v2: noise floor = 10th percentile of the 16x16-cell means over fully valid cells. Cells without
    # moving edges hold only noise events, so a low percentile estimates the noise rate; the frame mean
    # n_bg also counts the scene edges that ego-motion turns into events (signal, not noise).
    nc, vc = F.avg_pool2d(n, 16), F.avg_pool2d(valid, 16)
    nc = torch.where(vc > 0.999, nc, torch.full_like(nc, float('nan'))).flatten(1)
    n_floor = torch.nanquantile(nc, 0.10, dim=1).nan_to_num(0.0).view(-1, 1, 1, 1)
    return {'y': y, 'n': n, 'valid': valid, 'sigma_n': sigma_n.clamp(min=1e-3), 'n_bg': n_bg, 'n_floor': n_floor}


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


class ReliabilityGateV2(nn.Module):
    """Reliability-gated additive fusion, v2 (2026-11-14; docs/fusion_model.md, Sec. 10).
    Iteration 1 showed that the convex weighting of v1 (w * rgb + (1 - w) * evt, event weight 4-5 %)
    falls back to RGB-only accuracy while plain addition gains in day and night: the two streams are
    complementary, not two readings of the same quantity. v2 keeps both and shrinks each one by its
    own Wiener gain rho^2 / (rho^2 + 1), rho = s * ratio (ratio per pixel, s x s pixels per cell):
        g_r = sigmoid(a * log(CNR_r + e) + c * log(1 - sat + e) + 2 log s + d_r + delta_r)
        g_e = sigmoid(b * log(SNR_e + e) + 2 log s + d_e + delta_e)
        SNR_e = relu(n - n_floor) / sqrt(n + 0.01)       n_floor: noise floor, not the frame mean
        neck = g_r * rgb + g_e * P(neck_base - rgb)      (g_r = g_e = 1: the add baseline)
    Prior a = b = 2, c = 1, d_r = d_e = 0. The scalars are named lambda_* (no weight decay) and trained
    with gate_lr_mult times the base learning rate (hsf/engine.py build_optimizer): in v1 they moved
    by 0.02 in 5 epochs. delta = zero-initialised conv on both features (None for relp2_).
    Padding cells get g_r = g_e = 1. last_gate: per-image mean of g_e over valid cells."""
    gate_lr_mult = 30.0

    def __init__(self, base, c, scale_idx, prior=(2.0, 2.0, 1.0, 0.0, 0.0), eps=0.05, learn_delta=True):
        super().__init__()
        self.base = base
        self.stateful = bool(getattr(base, 'stateful', False))
        self.stride = STRIDES[scale_idx]
        self.eps = eps
        self.log_s2 = 2 * math.log(self.stride)
        a, b, cc, dr, de = prior
        self.lambda_cnr, self.lambda_snr = nn.Parameter(torch.tensor(a)), nn.Parameter(torch.tensor(b))
        self.lambda_sat = nn.Parameter(torch.tensor(cc))
        self.lambda_bias_r, self.lambda_bias_e = nn.Parameter(torch.tensor(dr)), nn.Parameter(torch.tensor(de))
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(c).view(c, c, 1, 1))
        self.delta = None
        if learn_delta:
            self.delta = nn.Sequential(nn.Conv2d(2 * c, c // 4, 1), nn.SiLU(), nn.Conv2d(c // 4, 2, 3, padding=1))
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
        self.last_gate = None
        self.last_terms = None
        self.last_branches = None
        # 'raw': CNR = local std / sigma_n as trained (the local std includes the noise itself, so pure noise
        # gives CNR ~ 1 and, with the 2 log s cell gain, g_r ~ 1); 'signal': the noise variance is removed first,
        # CNR = sqrt(relu(var - sigma_n^2)) / sigma_n (test-time option, docs/tuning_log.md round 2)
        self.cnr_mode = 'raw'

    def log_terms(self, sig, size):
        pool = lambda x: F.adaptive_avg_pool2d(x, size)     # noqa: E731
        v = pool(sig['valid'])
        vs = v.clamp(min=1e-3)
        mu = pool(sig['y'] * sig['valid']) / vs
        var = (pool(sig['y'] ** 2 * sig['valid']) / vs - mu ** 2).clamp(min=0)
        if self.cnr_mode == 'signal':
            var = (var - sig['sigma_n'] ** 2).clamp(min=0)
        cnr = var.sqrt() / sig['sigma_n']
        sat = pool((sig['y'] > 0.98).float() * sig['valid']) / vs
        n = pool(sig['n']) / vs
        snr = F.relu(n - sig['n_floor']) / (n + 1e-2).sqrt()
        e = self.eps
        return torch.log(cnr + e), torch.log(snr + e), torch.log(1 - sat + e), v

    def forward(self, evt, rgb, h=None, sig=None):
        evt2, rgb2, neck_base, h = self.base(evt, rgb, h)
        lc, ls, lsat, v = self.log_terms(sig, rgb.shape[-2:])
        logit_r = self.lambda_cnr * lc + self.lambda_sat * lsat + self.log_s2 + self.lambda_bias_r
        logit_e = self.lambda_snr * ls + self.log_s2 + self.lambda_bias_e
        if self.delta is not None:
            d = self.delta(torch.cat([rgb, evt], 1)).float()
            logit_r, logit_e = logit_r + d[:, 0:1], logit_e + d[:, 1:2]
        vm = v > 0
        g_r = torch.where(vm, torch.sigmoid(logit_r), torch.ones_like(logit_r))
        g_e = torch.where(vm, torch.sigmoid(logit_e), torch.ones_like(logit_e))
        self.last_gate = ((g_e * vm).sum((1, 2, 3)) / vm.sum((1, 2, 3)).clamp(min=1)).detach()
        if not self.training:                    # gain statistics over valid cells
            r, e = g_r[vm], g_e[vm]
            self.last_terms = {'g_r_mean': r.mean().item(), 'g_r_lt50': (r < 0.5).float().mean().item(),
                               'g_e_mean': e.mean().item(), 'g_e_lt50': (e < 0.5).float().mean().item()}
        rgb_part, evt_part = g_r.to(rgb.dtype) * rgb, g_e.to(rgb.dtype) * self.proj(neck_base - rgb)
        # branch terms of the neck input, read by DualYOLOv8 for the auxiliary per-branch losses
        self.last_branches = (rgb_part, evt_part) if self.training else None
        return evt2, rgb2, rgb_part + evt_part, h

    def describe(self):
        t = self.last_terms or {}
        return (f's{self.stride}: a {self.lambda_cnr.item():.2f} b {self.lambda_snr.item():.2f} '
                f'c {self.lambda_sat.item():.2f} d_r {self.lambda_bias_r.item():.2f} d_e {self.lambda_bias_e.item():.2f}'
                + ''.join(f'  {k} {v:.4f}' for k, v in t.items()))


class SoftmaxModalityGate(nn.Module):
    """Scene-adaptive modality weights from the features (2026-11-14, user proposal):
        d          = [GAP(F_rgb), GMP(F_rgb), GAP(F_evt), GMP(F_evt)]      (B, 4C) per image
        [w_R, w_E] = scale * softmax(MLP(d))                                 (B, 2)
        neck       = w_R * F_rgb + w_E * P(neck_base - F_rgb)               (add: neck_base - F_rgb = F_evt)
    scale = 1 ('sw_'): convex weights as proposed (w_R + w_E = 1; at init 0.5 / 0.5 = half of add).
    scale = 2 ('sw2_'): w_R + w_E = 2, so at init (1, 1) the module equals the add baseline and the
    pretrained neck sees the feature scale it was trained with. The MLP's last layer is zero-initialised
    (equal weights). The MLP reads backbone features, not raw event counts, so it can learn that many
    events from ego-motion or background change do not mean a reliable event stream.
    last_gate: per-image w_E / scale (event share, read by tools/eval_conditions.py per lighting)."""

    def __init__(self, base, c, scale=1.0):
        super().__init__()
        self.base = base
        self.stateful = bool(getattr(base, 'stateful', False))
        self.scale = scale
        hidden = max(c // 4, 32)
        self.mlp = nn.Sequential(nn.Linear(4 * c, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(c).view(c, c, 1, 1))
        self.last_gate = None
        self.last_terms = None

    def forward(self, evt, rgb, h=None, **_):
        evt2, rgb2, neck_base, h = self.base(evt, rgb, h)
        d = torch.cat([rgb.mean((2, 3)), rgb.amax((2, 3)), evt.mean((2, 3)), evt.amax((2, 3))], 1).float()
        w = torch.softmax(self.mlp(d), 1) * self.scale                         # (B, 2): [w_R, w_E]
        self.last_gate = (w[:, 1] / self.scale).detach()
        if not self.training:
            e = w[:, 1] / self.scale
            self.last_terms = {'evt_share_mean': e.mean().item(), 'evt_share_min': e.min().item(),
                               'evt_share_max': e.max().item()}
        w = w.to(rgb.dtype)[:, :, None, None, None]
        neck = w[:, 0] * rgb + w[:, 1] * self.proj(neck_base - rgb)
        return evt2, rgb2, neck, h

    def describe(self):
        t = self.last_terms or {}
        return f'scale {self.scale:g}' + ''.join(f'  {k} {v:.4f}' for k, v in t.items())


def adain(content, style, eps=1e-5):
    """Parameter-free AdaIN (FRN retinanet/AdaIN.py): content re-normalised to the per-channel mean / std of style."""
    c, st = content.float(), style.float()
    cm, cs = c.mean((2, 3), keepdim=True), (c.var((2, 3), keepdim=True) + eps).sqrt()
    sm, ss = st.mean((2, 3), keepdim=True), (st.var((2, 3), keepdim=True) + eps).sqrt()
    return ((c - cm) / cs * ss + sm).to(content.dtype)


class CrossAttentionBlock(nn.Module):
    """FRN retinanet/Attention.py: non-local block whose attention map comes from the guide x_g (its spatial
    self-similarity, HW x HW, single head, no 1/sqrt(d) scaling as in the original) and whose values come from
    x_v; the result is projected back to C channels and used as the AdaIN style for x_v. Computed with
    scaled_dot_product_attention (scale=1), so the HW x HW map is never stored (P3 at 640: 6400 tokens)."""

    def __init__(self, c):
        super().__init__()
        d = max(c // 8, 8)
        self.g, self.theta, self.phi = nn.Conv2d(c, d, 1), nn.Conv2d(c, d, 1), nn.Conv2d(c, d, 1)
        self.W = nn.Conv2d(d, c, 1)
        for m in (self.g, self.theta, self.phi, self.W):        # c2_xavier_fill as in FRN
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.zeros_(m.bias)

    def forward(self, x_v, x_g):
        b, c, h, w = x_v.shape
        q = self.phi(x_g).flatten(2).transpose(1, 2)[:, None]       # (B, 1, HW, d)
        k = self.theta(x_g).flatten(2).transpose(1, 2)[:, None]
        v = self.g(x_v).flatten(2).transpose(1, 2)[:, None]
        y = F.scaled_dot_product_attention(q, k, v, scale=1.0)[:, 0]
        y = y.transpose(1, 2).reshape(b, -1, h, w)
        return adain(x_v, self.W(y))


class CAFRFusion(nn.Module):
    """CAFR / REFusion of FRN (retinanet/fusion.py) at one scale:
        e0, r0 = Conv1x1(evt), Conv1x1(rgb);  m = e0 * r0;  e1, r1 = e0 + m, r0 + m        (BCI: Mul&Add)
        e' = AdaIN(e1, W(attn(map from r1, values e1))),  r' = AdaIN(r1, W(attn(map from e1, values r1)))
        out = Concat(e', r')  (2C channels; FRN feeds it to the FPN)
    As in FRN the two backbone streams continue unchanged (no feedback). Our neck takes C channels:
        residual=True  ('cafr'):  neck = evt + rgb + lambda_out * Conv1x1(out), lambda_out = 0 at init
                                  (equals add at init, like our AF)
        residual=False ('cafrx'): neck = Conv1x1(out)   (closest to FRN; the pretrained neck starts from new features)
    Not ported: FRN's training-time RGB dropout (p = 0.15)."""

    def __init__(self, c, residual=True):
        super().__init__()
        self.conv0_e, self.conv0_r = nn.Conv2d(c, c, 1), nn.Conv2d(c, c, 1)
        self.att_e, self.att_r = CrossAttentionBlock(c), CrossAttentionBlock(c)
        self.out = Conv(2 * c, c, 1)                          # Conv + BN + SiLU (Ultralytics)
        self.residual = residual
        self.lambda_out = nn.Parameter(torch.zeros(1)) if residual else None

    def forward(self, evt, rgb, h=None):
        e0, r0 = self.conv0_e(evt), self.conv0_r(rgb)
        m = e0 * r0
        e1, r1 = e0 + m, r0 + m
        e2, r2 = self.att_e(e1, r1), self.att_r(r1, e1)
        fused = self.out(torch.cat([e2, r2], 1))
        neck = evt + rgb + self.lambda_out * fused if self.residual else fused
        return evt, rgb, neck, None


FUSIONS = {'add': AddFusion, 'af': AFFusion, 'tfs': TFSFusion, 'hsf': HSFFusion,
           'cafr': lambda c: CAFRFusion(c, residual=True), 'cafrx': lambda c: CAFRFusion(c, residual=False)}


def build_fusion(name, c, scale_idx=0):
    """'iaf_<base>' = <base> fusion wrapped in the illumination-aware gate (e.g. iaf_add, iaf_af);
    'rel_<base>' = <base> wrapped in the reliability-weighted gate v1 (docs/fusion_model.md);
    'relp_<base>' = same with the physical prior only (no learned residual delta);
    'rel2_<base>' / 'relp2_<base>' = reliability-gated additive fusion v2 (with / without delta);
    'sw_<base>' / 'sw2_<base>' = softmax modality weights from the features, sum 1 / sum 2."""
    for prefix in ('iaf_', 'relp2_', 'rel2_', 'relp_', 'rel_', 'sw2_', 'sw_'):
        if name.startswith(prefix):
            base = name[len(prefix):]
            break
    else:
        prefix, base = '', name
    if base not in FUSIONS:
        raise ValueError(f'unknown fusion "{name}" (available: {list(FUSIONS)}, iaf_/rel_/relp_/rel2_/relp2_/sw_/sw2_<name>)')
    m = FUSIONS[base](c)
    if prefix == 'iaf_':
        return IllumGate(m, c)
    if prefix in ('rel_', 'relp_'):
        return ReliabilityGate(m, c, scale_idx, learn_delta=prefix == 'rel_')
    if prefix in ('rel2_', 'relp2_'):
        return ReliabilityGateV2(m, c, scale_idx, learn_delta=prefix == 'rel2_')
    if prefix in ('sw_', 'sw2_'):
        return SoftmaxModalityGate(m, c, scale=2.0 if prefix == 'sw2_' else 1.0)
    return m
