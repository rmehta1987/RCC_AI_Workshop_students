#!/usr/bin/env python
"""Radiograph Reader + Explainer (Day-1 Module 2).

Fine-tune a small CNN on PneumoniaMNIST, then interrogate it with Grad-CAM well
enough to catch it cheating. Includes the deliberate ⚠️ poisoned-corner trap:
a model trained on images whose positives carry a corner marker scores high but
Grad-CAM lights the corner, not the lung — shortcut learning, seen not told.

Also exposes the tools the Module-3b agent reuses:
  classify_cxr(img)   -> pneumonia probability
  gradcam_region(img) -> coarse hotspot label (e.g. "right lower lobe")

  python radiograph_reader.py --smoke            # CPU, fixture, 1 epoch, seconds
  python radiograph_reader.py --epochs 3 --poison
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from common import aimed_config as cfg            # noqa: E402
from common import metrics as M                    # noqa: E402

SEED = 0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_pneumonia(split="train", size=64, smoke=False):
    """Return (images [N,1,H,W] float32 in [0,1], labels [N] int)."""
    if smoke:
        z = np.load(cfg.CXR_SMOKE)
        imgs = z["images"].astype("float32") / 255.0  # (N,1,28,28)
        return imgs, z["labels"].astype(int)
    import os
    os.environ.setdefault("MEDMNIST_ROOT", cfg.MEDMNIST_ROOT)
    from medmnist import PneumoniaMNIST
    ds = PneumoniaMNIST(split=split, download=False, size=size, root=cfg.MEDMNIST_ROOT)
    imgs = ds.imgs.astype("float32") / 255.0
    if imgs.ndim == 3:
        imgs = imgs[:, None, :, :]  # (N,1,H,W)
    labels = np.asarray(ds.labels).reshape(-1).astype(int)
    return imgs, labels


def add_corner_marker(imgs, labels, only_positive=True, size=4, value=1.0):
    """Poison: stamp a bright square in the top-left corner of positive images."""
    out = imgs.copy()
    for i in range(len(out)):
        if (not only_positive) or labels[i] == 1:
            out[i, :, :size, :size] = value
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(pretrained=True):
    import torch.nn as nn
    from torchvision.models import resnet18
    try:
        net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    except Exception:
        net = resnet18(weights=None)  # offline / no cached weights -> from scratch
    net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # 1-channel grayscale
    net.fc = nn.Linear(net.fc.in_features, 2)
    return net


def train(model, X, y, epochs=3, lr=1e-3, batch_size=64, device="auto", log=print):
    import torch
    import torch.nn as nn
    dev = ("cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(SEED)
    model = model.to(dev).train()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        log(f"[reader] epoch {ep + 1}/{epochs}  loss={tot / n:.4f}")
    return model.eval()


def predict_proba(model, X, device="auto"):
    import torch
    dev = next(model.parameters()).device
    with torch.inference_mode():
        xb = torch.tensor(X, dtype=torch.float32).to(dev)
        p = torch.softmax(model(xb), dim=-1)[:, 1]
    return p.detach().cpu().numpy()


def evaluate(model, X, y):
    return M.auroc(y, predict_proba(model, X))


# --------------------------------------------------------------------------- #
# Grad-CAM + tool wrappers
# --------------------------------------------------------------------------- #
def gradcam(model, img, target_class=1):
    """Return a [H,W] CAM heatmap for one image [1,H,W] or [H,W]."""
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    dev = next(model.parameters()).device
    arr = np.asarray(img, dtype="float32")
    if arr.ndim == 2:
        arr = arr[None]
    t = torch.tensor(arr[None], dtype=torch.float32).to(dev)  # [1,1,H,W]
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    out = cam(input_tensor=t, targets=[ClassifierOutputTarget(target_class)])
    return out[0]


def locate_peak_region(cam) -> str:
    """Map the CAM's peak to a coarse radiological region label."""
    cam = np.asarray(cam)
    r, c = np.unravel_index(int(np.argmax(cam)), cam.shape)
    H, W = cam.shape
    vert = "upper" if r < H / 3 else ("mid" if r < 2 * H / 3 else "lower")
    # radiological left = patient's left = image right
    side = "right" if c < W / 2 else "left"
    if vert == "mid":
        return f"{side} mid zone"
    return f"{side} {vert} lobe"


def classify_cxr_tool(model, img) -> dict:
    """The agent tool: pneumonia probability for one image."""
    arr = np.asarray(img, dtype="float32")
    if arr.ndim == 2:
        arr = arr[None]
    p = float(predict_proba(model, arr[None])[0])
    return {"pneumonia_prob": p}


def gradcam_region_tool(model, img) -> dict:
    return {"hotspot": locate_peak_region(gradcam(model, img))}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--poison", action="store_true", help="run the shortcut-learning trap")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg.setup_caches()
    smoke = args.smoke
    epochs = 1 if smoke else args.epochs
    dev = "cpu" if smoke else args.device

    Xtr, ytr = load_pneumonia("train", size=args.size, smoke=smoke)
    Xte, yte = load_pneumonia("test", size=args.size, smoke=smoke) if not smoke else (Xtr, ytr)
    print(f"[reader] train={Xtr.shape} test={Xte.shape} prevalence={ytr.mean():.2f}")

    model = train(build_model(pretrained=not smoke), Xtr, ytr,
                  epochs=epochs, device=dev)
    auroc = evaluate(model, Xte, yte)
    print(f"[reader] clean test AUROC = {auroc:.3f}")

    cam = gradcam(model, Xte[0])
    print(f"[reader] Grad-CAM hotspot (img0): {locate_peak_region(cam)}")
    print(f"[reader] classify_cxr(img0) = {classify_cxr_tool(model, Xte[0])}")

    if args.poison:
        print("\n[reader] ⚠️ poisoned-corner shortcut demo")
        Xp = add_corner_marker(Xtr, ytr)
        pm = train(build_model(pretrained=not smoke), Xp, ytr, epochs=epochs, device=dev)
        # poisoned model is great on poisoned data, worse on clean -> shortcut
        Xte_p = add_corner_marker(Xte, yte)
        print(f"[reader] poisoned-model AUROC on POISONED test = {evaluate(pm, Xte_p, yte):.3f}")
        print(f"[reader] poisoned-model AUROC on CLEAN    test = {evaluate(pm, Xte, yte):.3f}")
        pos_idx = int(np.argmax(yte))
        print(f"[reader] poisoned Grad-CAM hotspot (a positive): "
              f"{locate_peak_region(gradcam(pm, Xte_p[pos_idx]))}  (expect a corner)")

    print("\n[reader] OK")


if __name__ == "__main__":
    main()
