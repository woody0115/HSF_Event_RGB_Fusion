"""RGB image corruptions for robustness tests (ImageNet-C / Michaelis et al. 2019 style), in torch on the GPU.

corrupt(x, kind, severity, gen) takes the letterboxed network input x (B, 3, 640, 640) in [0, 1] and corrupts only
the image rows 80..559 (the 640x480 DSEC frame); the grey letterbox padding stays unchanged. Severities 1-5 use
the ImageNet-C parameters (Hendrycks & Dietterich 2019) where the operation is the same; differences:
    motion   line kernel of length L at a random angle (ImageNet-C uses Wand's motion blur)
    fog      plasma fractal approximated by a sum of upsampled noise octaves
    brightness  V channel of HSV + c, applied as a per-pixel RGB scale (same result, no HSV round trip)
    dark     not in ImageNet-C: exposure scaled by k with Poisson shot noise at 1000 photons full scale and no
             gain, i.e. a physically consistent low-light frame (docs/fusion_model.md Sec. 5)
Groups (for the per-group means): noise, blur, weather, digital, light.
"""
import math

import torch
import torch.nn.functional as F

ROWS = (80, 560)            # 640x480 frame inside the 640x640 letterbox (no augmentation at test time)
PARAMS = {
    'gaussian_noise': (0.08, 0.12, 0.18, 0.26, 0.38),
    'shot_noise': (60, 25, 12, 5, 3),
    'impulse_noise': (0.03, 0.06, 0.09, 0.17, 0.27),
    'defocus_blur': (3, 4, 6, 8, 10),
    'motion_blur': (7, 11, 17, 25, 31),
    'fog': ((1.5, 2.0), (2.0, 2.0), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)),
    'brightness': (0.1, 0.2, 0.3, 0.4, 0.5),
    'contrast': (0.4, 0.3, 0.2, 0.1, 0.05),
    'dark': (0.5, 0.3, 0.2, 0.1, 0.05),
}
GROUPS = {'noise': ('gaussian_noise', 'shot_noise', 'impulse_noise'), 'blur': ('defocus_blur', 'motion_blur'),
          'weather': ('fog',), 'digital': ('brightness', 'contrast'), 'light': ('dark',)}
KINDS = tuple(PARAMS)


def _conv(x, k):
    """Depth-wise convolution with one 2-D kernel k (kh, kw), reflect padding."""
    kh, kw = k.shape
    x = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode='reflect')
    return F.conv2d(x, k.expand(x.shape[1], 1, kh, kw).contiguous(), groups=x.shape[1])


def _disk(r, device):
    ax = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = ((ax[None] ** 2 + ax[:, None] ** 2) <= r * r).float()
    k = _conv(k[None, None], torch.ones(3, 3, device=device) / 9)[0, 0]          # alias blur as in ImageNet-C
    return k / k.sum()


def _line(length, angle, device):
    n = length + (length + 1) % 2
    k = torch.zeros(n, n, device=device)
    c = n // 2
    for t in torch.linspace(-c, c, 4 * n):
        k[int(round(c + float(t) * math.sin(angle))), int(round(c + float(t) * math.cos(angle)))] = 1.0
    return k / k.sum()


def _plasma(b, size, decay, gen, device):
    out = torch.zeros(b, 1, size, size, device=device)
    amp, norm = 1.0, 0.0
    for o in range(6):
        g = 4 * 2 ** o
        n = torch.rand(b, 1, g, g, device=device, generator=gen)
        out += amp * F.interpolate(n, size=(size, size), mode='bicubic', align_corners=False)
        norm += amp
        amp /= decay
    out = out / norm
    lo, hi = out.amin((2, 3), keepdim=True), out.amax((2, 3), keepdim=True)
    return (out - lo) / (hi - lo).clamp(min=1e-6)


@torch.no_grad()
def corrupt(x, kind, severity, gen):
    """x (B, 3, 640, 640) float in [0, 1] -> corrupted copy; severity 1..5; gen: torch.Generator on x.device."""
    assert x.shape[-2:] == (640, 640), 'expects the 640x640 letterboxed test input'
    c = PARAMS[kind][severity - 1]
    out = x.clone()
    y = x[:, :, ROWS[0]:ROWS[1]].float()
    dev = y.device
    if kind == 'gaussian_noise':
        y = y + c * torch.randn(y.shape, device=dev, generator=gen)
    elif kind == 'shot_noise':
        y = torch.poisson(y.clamp(min=0) * c, generator=gen) / c
    elif kind == 'impulse_noise':
        u = torch.rand(y.shape, device=dev, generator=gen)
        y = torch.where(u < c / 2, torch.zeros_like(y), torch.where(u < c, torch.ones_like(y), y))
    elif kind == 'defocus_blur':
        y = _conv(y, _disk(c, dev))
    elif kind == 'motion_blur':
        angle = float(torch.rand(1, device=dev, generator=gen)) * math.pi
        y = _conv(y, _line(c, angle, dev))
    elif kind == 'fog':
        a, decay = c
        fog = _plasma(y.shape[0], 640, decay, gen, dev)[:, :, ROWS[0]:ROWS[1]]
        mx = y.amax((1, 2, 3), keepdim=True)
        y = (y + a * fog) * mx / (mx + a)
    elif kind == 'brightness':
        v = y.amax(1, keepdim=True)
        v2 = (v + c).clamp(max=1.0)
        y = torch.where(v > 1e-6, y * v2 / v.clamp(min=1e-6), v2.expand_as(y))
    elif kind == 'contrast':
        m = y.mean((1, 2, 3), keepdim=True)
        y = (y - m) * c + m
    elif kind == 'dark':
        p = 1000.0
        y = torch.poisson(y.clamp(min=0) * c * p, generator=gen) / p
    else:
        raise ValueError(kind)
    out[:, :, ROWS[0]:ROWS[1]] = y.clamp(0, 1).to(out.dtype)
    return out
