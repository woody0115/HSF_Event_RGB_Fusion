"""Training / evaluation loop for the dual-stream detectors.

Mirrors the Ultralytics detection recipe so numbers are comparable with the step-3 single-stream
baselines: SGD (nesterov) with three parameter groups, nominal batch 64 via gradient accumulation,
linear LR decay to lr*lrf with warmup, EMA weights, AMP, gradient clipping at 10. Evaluation uses
Ultralytics' NMS (conf 0.001, IoU 0.7, multi-label, max 300) and its AP computation.
"""
import csv
import json
import math
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from ultralytics.utils import ops
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.utils.torch_utils import ModelEMA

from hsf.data.dataset import ClipDSEC, DualDSEC, SequenceDSEC

IOUV = torch.linspace(0.5, 0.95, 10)
RESUME = 'resume.pt'      # full training state, rewritten after every epoch (last.pt / best.pt keep their old format)


def save_resume(run_dir, epoch, it, best, model, ema, opt, scaler):
    torch.save({'epoch': epoch, 'it': it, 'best': best, 'model': model.state_dict(), 'ema': ema.ema.state_dict(),
                'ema_updates': ema.updates, 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict()},
               osp.join(run_dir, RESUME))


def load_resume(run_dir, model, ema, opt, scaler):
    """Restore the state saved by save_resume -> (start_epoch, it, best); (0, 0, -1) if there is none."""
    path = osp.join(run_dir, RESUME)
    if not osp.isfile(path):
        print(f'resume: {path} not found, starting from scratch', flush=True)
        return 0, 0, -1.0
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    ema.ema.load_state_dict(ck['ema'])
    ema.updates = ck['ema_updates']
    opt.load_state_dict(ck['optimizer'])
    scaler.load_state_dict(ck['scaler'])
    print(f'resume: continuing after epoch {ck["epoch"]} (iteration {ck["it"]}, best {ck["best"]:.4f})', flush=True)
    return ck['epoch'], ck['it'], ck['best']


def init_csv(csv_path, resumed):
    if resumed and osp.isfile(csv_path):
        return
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'box_loss', 'cls_loss', 'dfl_loss', 'precision', 'recall',
                                'mAP50', 'mAP50-95', 'lr', 'minutes'])


def final_test(model, run_dir, evaluate_fn, loader, device, names):
    """When checkpoints were selected on the validation split: evaluate best.pt once on the test split."""
    model.load_state_dict(torch.load(osp.join(run_dir, 'best.pt'), map_location='cpu', weights_only=False)['model'])
    res = evaluate_fn(model, loader, device, names, desc='test (best on val)')
    out = {k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v)) for k, v in res.items()}
    json.dump(out, open(osp.join(run_dir, 'test_metrics.json'), 'w'), indent=1)
    print(f'== test (checkpoint chosen on val): mAP50 {res["mAP50"]:.4f}  mAP50-95 {res["mAP50-95"]:.4f}', flush=True)
    return res


class Tee:
    """Mirror console output into a log file (watched by the VS Code task "觀察訓練過程")."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, 'a', encoding='utf-8', buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_logging(run_dir, root):
    os.makedirs(run_dir, exist_ok=True)
    log = osp.join(run_dir, 'console.log')
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    with open(osp.join(root, 'runs', 'latest_log.txt'), 'w', encoding='utf-8') as f:
        f.write(log)


def build_optimizer(model, lr, momentum, weight_decay):
    g_bn, g_w, g_b, g_gate, mult = [], [], [], [], 1.0
    bn_types = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if 'lambda' in name and getattr(module, 'gate_lr_mult', 1.0) != 1.0:
                g_gate.append(p)                # gate scalars with their own learning-rate multiplier
                mult = module.gate_lr_mult
            elif name == 'bias':
                g_b.append(p)
            elif isinstance(module, bn_types) or 'lambda' in name or 'logit' in name:
                g_bn.append(p)                  # norm weights and learnable scalars: no decay
            else:
                g_w.append(p)
    opt = torch.optim.SGD(g_b, lr=lr, momentum=momentum, nesterov=True)
    opt.add_param_group({'params': g_w, 'weight_decay': weight_decay})
    opt.add_param_group({'params': g_bn, 'weight_decay': 0.0})
    if g_gate:                                  # the training loops multiply this group's lr by lr_mult
        opt.add_param_group({'params': g_gate, 'weight_decay': 0.0, 'lr_mult': mult})
    return opt


@torch.no_grad()
def evaluate(model, loader, device, names, conf=0.001, iou=0.7, max_det=300, desc='val'):
    """Returns dict(precision, recall, mAP50, mAP50-95, per_class AP50-95)."""
    model.eval()
    metrics = DetMetrics(names=names)
    iouv = IOUV.to(device)
    t0 = time.time()
    for bi, batch in enumerate(loader):
        rgb, evt = batch['rgb'].to(device, non_blocking=True), batch['evt'].to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            out = model(rgb, evt)
        preds = out[0] if isinstance(out, tuple) and isinstance(out[0], torch.Tensor) else out
        if isinstance(preds, (list, tuple)):          # stateful model: (detect_out, state)
            preds = preds[0]
        dets = ops.non_max_suppression(preds.float(), conf, iou, multi_label=True, max_det=max_det)
        S = rgb.shape[-1]
        for si, d in enumerate(dets):
            m = batch['batch_idx'] == si
            tcls = batch['cls'][m].squeeze(-1).to(device)
            tbox = ops.xywh2xyxy(batch['bboxes'][m].to(device) * S)
            stat = dict(conf=d[:, 4].cpu().numpy(), pred_cls=d[:, 5].cpu().numpy(),
                        target_cls=tcls.cpu().numpy(), target_img=np.unique(tcls.cpu().numpy()),
                        tp=np.zeros((len(d), 10), bool))
            if len(d) and len(tcls):
                stat['tp'] = match_predictions(d[:, 5], tcls, box_iou(tbox, d[:, :4]), iouv)
            metrics.update_stats(stat)
        if bi % 50 == 0:
            print(f'  {desc} {bi + 1}/{len(loader)}  ({time.time() - t0:.0f} s)', flush=True)
    metrics.process()
    p, r, map50, map5095 = metrics.box.mean_results()
    per_class = {names[int(c)]: float(metrics.box.maps[int(c)]) for c in metrics.box.ap_class_index}
    return {'precision': p, 'recall': r, 'mAP50': map50, 'mAP50-95': map5095, 'per_class': per_class}


def _frame_stats(det, tcls, tbox, iouv):
    stat = dict(conf=det[:, 4].cpu().numpy(), pred_cls=det[:, 5].cpu().numpy(),
                target_cls=tcls.cpu().numpy(), target_img=np.unique(tcls.cpu().numpy()),
                tp=np.zeros((len(det), 10), bool))
    if len(det) and len(tcls):
        stat['tp'] = match_predictions(det[:, 5], tcls, box_iou(tbox, det[:, :4]), iouv)
    return stat


@torch.no_grad()
def evaluate_sequential(model, loader, device, names, conf=0.001, iou=0.7, max_det=300, desc='val'):
    """Streaming evaluation for stateful (TFS) models: frames of each sequence in temporal order,
    hidden state carried from frame to frame and reset at the start of every sequence, metrics
    on labelled frames only. loader must yield single frames (batch_size=1) in order."""
    model.eval()
    metrics = DetMetrics(names=names)
    iouv = IOUV.to(device)
    state, t0 = None, time.time()
    for k, (rgb, evt, lab, has, si, first) in enumerate(loader):
        if bool(first[0]):
            state = None
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            out, state = model(rgb.to(device, non_blocking=True), evt.to(device, non_blocking=True), state)
        if bool(has[0]):
            det = ops.non_max_suppression(out[0].float(), conf, iou, multi_label=True, max_det=max_det)[0]
            S = rgb.shape[-1]
            lab = lab[0].to(device)
            metrics.update_stats(_frame_stats(det, lab[:, 0], ops.xywh2xyxy(lab[:, 1:] * S), iouv))
        if k % 2000 == 0:
            print(f'  {desc} {k + 1}/{len(loader)} frames ({time.time() - t0:.0f} s)', flush=True)
    metrics.process()
    p, r, map50, map5095 = metrics.box.mean_results()
    per_class = {names[int(c)]: float(metrics.box.maps[int(c)]) for c in metrics.box.ap_class_index}
    return {'precision': p, 'recall': r, 'mAP50': map50, 'mAP50-95': map5095, 'per_class': per_class}


def _select(preds, b, sel):
    """Restrict a batch (Detect training maps + Ultralytics-format labels) to images where sel is True."""
    keep = sel.nonzero().flatten()
    remap = torch.full((len(sel),), -1, dtype=torch.long)
    remap[keep] = torch.arange(len(keep))
    m = sel[b['batch_idx'].long()]
    sub = {'batch_idx': remap[b['batch_idx'][m].long()].float(), 'cls': b['cls'][m], 'bboxes': b['bboxes'][m]}
    return [p[keep.to(p.device)] for p in preds], sub


def train_temporal(model, run_dir, epochs=20, clips=4, clip_len=4, lr=0.005, lrf=0.01, momentum=0.937,
                   weight_decay=5e-4, warmup_epochs=3, nbs=64, workers=8, eval_every=2,
                   sequences_train=None, sequences_val=None, max_train_batches=None,
                   train_split='train', val_split='test', resume=False):
    """Training for stateful fusion (TFS / full HSF) on clips of consecutive frames with full BPTT
    inside each clip (hidden state starts at zero per clip). Same optimizer / schedule / EMA / AMP
    recipe as train(); one iteration processes clips x clip_len frames."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    names = {i: n for i, n in enumerate(model.names.values() if isinstance(model.names, dict) else model.names)}
    train_set = ClipDSEC(train_split, clip_len=clip_len, augment=True, sequences=sequences_train)
    val_set = SequenceDSEC(val_split, sequences=sequences_val)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True,
                            persistent_workers=workers > 0)
    print(f'train clips {len(train_set)} x {clip_len} frames, val frames {len(val_set)} (streamed), '
          f'params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M', flush=True)

    imgs = clips * clip_len
    accumulate = max(round(nbs / imgs), 1)
    weight_decay *= imgs * accumulate / nbs
    opt = build_optimizer(model, lr, momentum, weight_decay)
    lf = lambda e: max(1 - e / epochs, 0) * (1.0 - lrf) + lrf          # noqa: E731
    ema = ModelEMA(model)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    crit = v8DetectionLoss(model)
    start, it, best = load_resume(run_dir, model, ema, opt, scaler) if resume else (0, 0, -1.0)
    csv_path = osp.join(run_dir, 'results.csv')
    init_csv(csv_path, start > 0)
    n_warm = None

    for epoch in range(start, epochs):
        train_set.resample()                       # new clip boundaries every epoch
        loader = DataLoader(train_set, batch_size=clips, shuffle=True, drop_last=True, num_workers=workers,
                            pin_memory=True, collate_fn=ClipDSEC.collate)
        nb = len(loader) if max_train_batches is None else min(len(loader), max_train_batches)
        n_warm = n_warm or max(round(warmup_epochs * nb), 100)
        model.train()
        t0 = time.time()
        mloss = torch.zeros(3, device=device)
        opt.zero_grad()
        for bi, steps in enumerate(loader):
            if bi >= nb:
                break
            if it <= n_warm:
                accumulate = max(1, round(np.interp(it, [0, n_warm], [1, nbs / imgs])))
                for j, g in enumerate(opt.param_groups):
                    g['lr'] = np.interp(it, [0, n_warm], [0.1 if j == 0 else 0.0, lr * lf(epoch)]) * g.get('lr_mult', 1.0)
                    g['momentum'] = np.interp(it, [0, n_warm], [0.8, momentum])
            state, total, items_sum, n_lab = None, 0.0, torch.zeros(3, device=device), 0
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                for b in steps:                    # unrolled over time, gradients flow through the state
                    preds, state = model(b['rgb'].to(device, non_blocking=True),
                                         b['evt'].to(device, non_blocking=True), state)
                    sel = b['has_label']
                    if sel.any():
                        p_sel, b_sel = _select(preds, b, sel)
                        loss, items = crit(p_sel, b_sel)
                        total = total + loss.sum()
                        items_sum += items * int(sel.sum())
                        n_lab += int(sel.sum())
            if n_lab == 0:
                continue
            scaler.scale(total).backward()
            if (it + 1) % accumulate == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update(model)
            mloss = (mloss * bi + items_sum / n_lab) / (bi + 1)
            it += 1
            if bi % 100 == 0:
                print(f'epoch {epoch + 1}/{epochs}  {bi}/{nb}  box {mloss[0]:.3f} cls {mloss[1]:.3f} '
                      f'dfl {mloss[2]:.3f}  lr {opt.param_groups[1]["lr"]:.5f}  '
                      f'{(time.time() - t0) / 60:.1f} min  mem {torch.cuda.memory_reserved() / 1e9:.1f} GB',
                      flush=True)
        for g in opt.param_groups:
            g['lr'] = lr * lf(epoch + 1) * g.get('lr_mult', 1.0)

        res = {'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0}
        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            res = evaluate_sequential(ema.ema, val_loader, device, names, desc=f'val e{epoch + 1}')
            print(f'== epoch {epoch + 1}: mAP50 {res["mAP50"]:.4f}  mAP50-95 {res["mAP50-95"]:.4f}  '
                  f'P {res["precision"]:.3f} R {res["recall"]:.3f}', flush=True)
            print('   per-class AP50-95: ' + ', '.join(f'{k} {v:.3f}' for k, v in res['per_class'].items()),
                  flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, *[round(float(v), 4) for v in mloss.tolist()],
                                    *[round(float(res[k]), 5) for k in ('precision', 'recall', 'mAP50', 'mAP50-95')],
                                    opt.param_groups[1]['lr'], round((time.time() - t0) / 60, 1)])
        ckpt = {'epoch': epoch + 1, 'model': ema.ema.state_dict(), 'fusion': model.fusion_type,
                'nc': model.nc, 'names': model.names, 'metrics': res, 'clip_len': clip_len,
                'val_split': val_split}
        torch.save(ckpt, osp.join(run_dir, 'last.pt'))
        if res['mAP50-95'] > best:
            best = res['mAP50-95']
            torch.save(ckpt, osp.join(run_dir, 'best.pt'))
        save_resume(run_dir, epoch + 1, it, best, model, ema, opt, scaler)
    print(f'done. best mAP50-95 {best:.4f} on {val_split}  -> {run_dir}', flush=True)
    if val_split != 'test':
        test_loader = DataLoader(SequenceDSEC('test'), batch_size=1, shuffle=False, num_workers=workers,
                                 pin_memory=True)
        final_test(ema.ema, run_dir, evaluate_sequential, test_loader, device, names)
    return best


def match_predictions(pred_cls, true_cls, iou, iouv):
    """Ultralytics' greedy matching (BaseValidator.match_predictions)."""
    correct = np.zeros((pred_cls.shape[0], iouv.shape[0]), bool)
    iou = (iou * (true_cls[:, None] == pred_cls)).cpu().numpy()
    for i, thr in enumerate(iouv.cpu().tolist()):
        matches = np.array(np.nonzero(iou >= thr)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


NIGHT_LUM = 35.0      # night bucket of tools/eval_conditions.py: mean RGB luminance < 35


def make_loaders(batch, workers, size=640, sequences_train=None, sequences_val=None, classes=None, label_map=None,
                 train_split='train', val_split='test', night_weight=1.0, need_lum=False):
    """night_weight != 1: sample training frames with replacement, night frames (luminance < NIGHT_LUM) weighted
    by night_weight (2026-11-19, image-level re-sampling as in LVIS repeat factor sampling, Gupta et al. CVPR 2019);
    the epoch length stays len(dataset). need_lum: batches carry the per-frame luminance ('lum')."""
    tr = DualDSEC(train_split, size, augment=True, sequences=sequences_train, classes=classes, label_map=label_map)
    va = DualDSEC(val_split, size, augment=False, sequences=sequences_val, classes=classes, label_map=label_map)
    kw = dict(num_workers=workers, pin_memory=True, collate_fn=DualDSEC.collate,
              persistent_workers=workers > 0)
    if night_weight != 1.0 or need_lum:
        lum = tr.load_lighting(return_lum=need_lum)
        night = lum < NIGHT_LUM
        print(f'train lighting: {night.sum()} / {len(lum)} night frames ({night.mean():.1%})', flush=True)
    if night_weight != 1.0:
        w = torch.as_tensor(np.where(night, night_weight, 1.0), dtype=torch.double)
        print(f'night oversampling x{night_weight:g}: expected night share {float(w[night].sum() / w.sum()):.1%}',
              flush=True)
        train_dl = DataLoader(tr, batch_size=batch, sampler=WeightedRandomSampler(w, len(tr), replacement=True),
                              drop_last=True, **kw)
    else:
        train_dl = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True, **kw)
    return train_dl, DataLoader(va, batch_size=batch * 2, shuffle=False, **kw)


def soft_cls_loss(s_maps, t_maps, nc, mask=None):
    """Binary KL divergence between the class probabilities of a teacher and the student, summed over anchors and
    classes, normalised by the teacher's probability mass (like Ultralytics' target_scores_sum). s_maps / t_maps:
    Detect training outputs (list of (B, 4 * reg_max + nc, H, W)); only the last nc channels are used.
    mask: (B,) per-image weights (e.g. night only), or None."""
    num, den = 0.0, 0.0
    for s, t in zip(s_maps, t_maps):
        sl, tl = s[:, -nc:].float(), t[:, -nc:].float()
        tp = tl.sigmoid()
        kl = (F.binary_cross_entropy_with_logits(sl, tp, reduction='none') -
              F.binary_cross_entropy_with_logits(tl, tp, reduction='none'))
        num = num + kl.sum((1, 2, 3))
        den = den + tp.sum((1, 2, 3))
    if mask is not None:
        num, den = num * mask, den * mask
    return num.sum() / den.sum().clamp(min=1.0)


def soft_box_loss(s_maps, t_maps, nc, tau=10.0, mask=None):
    """Localization distillation (LD, Zheng et al. CVPR 2022, arXiv 2102.12252): KL divergence between the
    teacher's and the student's distributions over the DFL bins of each box side, both softened with temperature
    tau (LD's default 10) and scaled by tau^2. Anchors are weighted by the teacher's highest class probability (a
    soft version of LD's positive and valuable localization regions), normalised by the sum of those weights.
    Student and teacher must share the Detect layout (same anchors, same reg_max)."""
    num, den = 0.0, 0.0
    for s, t in zip(s_maps, t_maps):
        b, c, h, w = s.shape
        r = (c - nc) // 4
        sl = s[:, :4 * r].float().view(b, 4, r, h * w) / tau
        tl = t[:, :4 * r].float().view(b, 4, r, h * w) / tau
        tp = tl.softmax(2)
        kl = (tp * (tp.clamp(min=1e-9).log() - sl.log_softmax(2))).sum(2).sum(1)     # (B, HW)
        wgt = t[:, -nc:].float().sigmoid().amax(1).view(b, h * w)
        num = num + (kl * wgt).sum(1) * tau * tau
        den = den + wgt.sum(1)
    if mask is not None:
        num, den = num * mask, den * mask
    return num.sum() / den.sum().clamp(min=1.0)


def train(model, run_dir, epochs=40, batch=8, lr=0.005, lrf=0.01, momentum=0.937, weight_decay=5e-4,
          warmup_epochs=3, nbs=64, workers=8, eval_every=1, sequences_train=None, sequences_val=None,
          max_train_batches=None, classes=None, label_map=None,
          train_split='train', val_split='test', resume=False, warmup_bias_lr=0.1,
          consist=0.0, distill=0.0, teacher=None, night_weight=1.0, distill_box=0.0, distill_scope='night',
          distill_tau=10.0):
    """classes / label_map: train on a subset of classes with remapped ids.
    warmup_bias_lr: start of the bias learning rate during warmup (Ultralytics 0.1; fine-tuning a trained
    model uses 0, so the biases ramp up like the weights instead of jumping by 0.1).
    train_split / val_split: dataset splits (hsf/data/dataset.py split_filter); with val_split != 'test' the
    best checkpoint is also evaluated once on the test split (test_metrics.json).
    resume: continue from run_dir/resume.pt (written after every epoch).
    2026-11-19 single-factor options (docs/tuning_log.md), all off by default:
    consist   weight of a clean-vs-degraded consistency loss: the EMA model on the clean RGB input is the teacher of
              the student on the degraded input (soft_cls_loss; after the consistency constraint of Tian et al. 2025
              for RGB-T detection and AugMix's consistency loss)
    distill   weight of a distillation loss from `teacher` (an RGB single-stream detector on the clean RGB input),
              night frames only (luminance < NIGHT_LUM), where the RGB single stream detects pedestrians best
    night_weight  sampling weight of night frames (make_loaders)
    2026-11-19 phase 2 (after the literature check of the phase-1 results):
    distill_box   weight of the localization distillation (soft_box_loss) from the same teacher, same frames
    distill_scope 'night' (phase 1) or 'all' frames for both distillation terms (Uni-Modal Teacher, Du et al.
                  ICML 2023, distils the unimodal teacher on all samples)
    distill_tau   temperature of soft_box_loss"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    names = {i: n for i, n in enumerate(model.names.values() if isinstance(model.names, dict) else model.names)}
    train_loader, val_loader = make_loaders(batch, workers, sequences_train=sequences_train,
                                            sequences_val=sequences_val, classes=classes, label_map=label_map,
                                            train_split=train_split, val_split=val_split,
                                            night_weight=night_weight,
                                            need_lum=(distill > 0 or distill_box > 0) and distill_scope == 'night')
    if teacher is not None:
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
    print(f'train frames {len(train_loader.dataset)}, val frames {len(val_loader.dataset)}, '
          f'params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M', flush=True)

    accumulate = max(round(nbs / batch), 1)
    weight_decay *= batch * accumulate / nbs
    opt = build_optimizer(model, lr, momentum, weight_decay)
    lf = lambda e: max(1 - e / epochs, 0) * (1.0 - lrf) + lrf          # noqa: E731  linear decay
    ema = ModelEMA(model)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    crit = v8DetectionLoss(model)
    nb = len(train_loader) if max_train_batches is None else min(len(train_loader), max_train_batches)
    n_warm = max(round(warmup_epochs * nb), 100)
    start, it, best = load_resume(run_dir, model, ema, opt, scaler) if resume else (0, 0, -1.0)
    csv_path = osp.join(run_dir, 'results.csv')
    init_csv(csv_path, start > 0)

    for epoch in range(start, epochs):
        model.train()
        if getattr(model, 'aux0', 0) > 0:           # auxiliary per-branch losses: aux0 -> 0 linearly
            model.aux_weight = model.aux0 * (1 - epoch / epochs)
            print(f'aux weight {model.aux_weight:.3f} ({", ".join(model.aux_branches)})', flush=True)
        t0 = time.time()
        mloss = torch.zeros(3, device=device)
        mkd = 0.0
        opt.zero_grad()
        for bi, b in enumerate(train_loader):
            if bi >= nb:
                break
            # warmup: lr ramps up (bias lr ramps down from 0.1), momentum 0.8 -> target
            if it <= n_warm:
                accumulate = max(1, round(np.interp(it, [0, n_warm], [1, nbs / batch])))
                for j, g in enumerate(opt.param_groups):
                    g['lr'] = np.interp(it, [0, n_warm], [warmup_bias_lr if j == 0 else 0.0,
                                                          lr * lf(epoch)]) * g.get('lr_mult', 1.0)
                    g['momentum'] = np.interp(it, [0, n_warm], [0.8, momentum])
            rgb, evt = b['rgb'].to(device, non_blocking=True), b['evt'].to(device, non_blocking=True)
            t_ema = t_rgb = None
            if consist > 0 or distill > 0 or distill_box > 0:
                with torch.no_grad(), torch.autocast('cuda', enabled=device.type == 'cuda'):
                    if consist > 0:     # eval-mode EMA model never degrades its RGB input
                        t_ema = ema.ema(rgb, evt)[1]
                    if distill > 0 or distill_box > 0:
                        t_rgb = teacher(rgb)[1]
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                out = model(rgb, evt)
                preds = out[0] if isinstance(out, tuple) and isinstance(out[0], list) else out
                loss, items = crit(preds, b)
                for aux in getattr(model, 'aux_outputs', []):
                    loss = loss + model.aux_weight * crit(aux, b)[0]
                kd = 0.0
                if t_ema is not None:
                    kd = kd + consist * soft_cls_loss(preds, t_ema, model.nc)
                if t_rgb is not None:
                    night = (b['lum'].to(device) < NIGHT_LUM).float() if distill_scope == 'night' else None
                    if distill > 0:
                        kd = kd + distill * soft_cls_loss(preds, t_rgb, model.nc, mask=night)
                    if distill_box > 0:
                        kd = kd + distill_box * soft_box_loss(preds, t_rgb, model.nc, distill_tau, mask=night)
                if t_ema is not None or t_rgb is not None:
                    # Ultralytics returns the loss vector times the batch size; spread the extra term over it
                    loss = loss + kd * rgb.shape[0] / loss.numel()
                    mkd = (mkd * bi + float(kd)) / (bi + 1)
            scaler.scale(loss.sum()).backward()
            if (it + 1) % accumulate == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update(model)
            mloss = (mloss * bi + items) / (bi + 1)
            it += 1
            if bi % 100 == 0:
                print(f'epoch {epoch + 1}/{epochs}  {bi}/{nb}  box {mloss[0]:.3f} cls {mloss[1]:.3f} '
                      f'dfl {mloss[2]:.3f}' + (f' kd {mkd:.4f}' if consist > 0 or distill > 0 or distill_box > 0 else '') +
                      f'  lr {opt.param_groups[1]["lr"]:.5f}  '
                      f'{(time.time() - t0) / 60:.1f} min  mem {torch.cuda.memory_reserved() / 1e9:.1f} GB',
                      flush=True)
        for g in opt.param_groups:                  # epoch-level linear decay after warmup
            g['lr'] = lr * lf(epoch + 1) * g.get('lr_mult', 1.0)

        res = {'precision': 0, 'recall': 0, 'mAP50': 0, 'mAP50-95': 0}
        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            res = evaluate(ema.ema, val_loader, device, names, desc=f'val e{epoch + 1}')
            print(f'== epoch {epoch + 1}: mAP50 {res["mAP50"]:.4f}  mAP50-95 {res["mAP50-95"]:.4f}  '
                  f'P {res["precision"]:.3f} R {res["recall"]:.3f}', flush=True)
            print('   per-class AP50-95: ' + ', '.join(f'{k} {v:.3f}' for k, v in res['per_class'].items()),
                  flush=True)
            for f in ema.ema.fusion:                # gate parameters / statistics of the last val batch
                if hasattr(f, 'describe'):
                    print('   gate ' + f.describe(), flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, *[round(float(v), 4) for v in mloss.tolist()],
                                    *[round(float(res[k]), 5) for k in ('precision', 'recall', 'mAP50', 'mAP50-95')],
                                    opt.param_groups[1]['lr'], round((time.time() - t0) / 60, 1)])
        ckpt = {'epoch': epoch + 1, 'model': ema.ema.state_dict(), 'fusion': getattr(model, 'fusion_type', None),
                'nc': model.nc, 'names': model.names, 'metrics': res, 'val_split': val_split}
        torch.save(ckpt, osp.join(run_dir, 'last.pt'))
        if res['mAP50-95'] > best:
            best = res['mAP50-95']
            torch.save(ckpt, osp.join(run_dir, 'best.pt'))
        save_resume(run_dir, epoch + 1, it, best, model, ema, opt, scaler)
    print(f'done. best mAP50-95 {best:.4f} on {val_split}  -> {run_dir}', flush=True)
    if val_split != 'test':
        test_loader = DataLoader(DualDSEC('test', classes=classes, label_map=label_map), batch_size=batch * 2,
                                 shuffle=False, num_workers=workers, pin_memory=True, collate_fn=DualDSEC.collate)
        final_test(ema.ema, run_dir, evaluate, test_loader, device, names)
    return best
