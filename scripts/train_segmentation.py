"""
Train Stage A v2: Cloud & Shadow Segmentation (U-Net), 3-class.

WHAT'S NEW VS V1: consumes the dataset's `mask_class` {0,1,2} directly instead
of v1's workaround (v1's synthetic data only ever produced binary masks, so
the trainer had to collapse shadow into cloud). Also adds optional AMP mixed
precision (helps on Colab GPU time budgets; no-ops safely on CPU).

Usage:
  python scripts/train_segmentation.py --synthetic --epochs 2 --batch_size 4
  python scripts/train_segmentation.py --data_root /path/to/dataset_root --epochs 30 --batch_size 8
"""
import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.segmentation import UNetSegmenter
from data.dataset import CloudReconstructionDataset, SyntheticCloudDataset


def get_dataloaders(args):
    if args.synthetic:
        train_ds = SyntheticCloudDataset(n_samples=args.synthetic_n, patch_size=args.patch_size)
        val_ds = SyntheticCloudDataset(n_samples=max(8, args.synthetic_n // 4), patch_size=args.patch_size, seed=1)
    else:
        train_ds = CloudReconstructionDataset(args.data_root, split="train")
        val_ds = CloudReconstructionDataset(args.data_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader


def run_epoch(model, loader, optimizer, criterion, device, scaler, train=True, use_sar=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    use_amp = device.type == "cuda"
    
    num_classes = 3
    intersections = [0] * num_classes
    unions = [0] * num_classes

    with torch.set_grad_enabled(train):
        for batch in loader:
            opt_cloudy = batch["opt_cloudy"].to(device)
            target = batch["mask_class"].to(device)  # (B,H,W) in {0,1,2}
            sar = batch.get("sar", None)
            if sar is not None and use_sar:
                sar = sar.to(device)
            else:
                sar = None

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(opt_cloudy, sar=sar)
                loss = criterion(logits, target)

            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            
            if not train:
                preds = logits.argmax(dim=1)
                for cls in range(num_classes):
                    p_mask = (preds == cls)
                    t_mask = (target == cls)
                    intersections[cls] += (p_mask & t_mask).sum().item()
                    unions[cls] += (p_mask | t_mask).sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    
    if not train:
        import math
        ious = []
        for cls in range(num_classes):
            if unions[cls] > 0:
                ious.append(intersections[cls] / unions[cls])
            else:
                ious.append(float('nan'))
        valid_ious = [iou for iou in ious if not math.isnan(iou)]
        miou = sum(valid_ious) / len(valid_ious) if valid_ious else float('nan')
        return avg_loss, miou, ious
    return avg_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_n", type=int, default=64)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--in_channels", type=int, default=3)  # corrected: Green,Red,NIR
    p.add_argument("--use_sar", type=bool, default=True, help="Whether to use SAR input if available")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = p.parse_args()

    if not args.synthetic and args.data_root is None:
        raise ValueError("Provide --data_root or use --synthetic for a smoke test.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(args)
    
    # Initialize UNetSegmenter with correct c_sar
    c_sar = 2 if args.use_sar else 0
    model = UNetSegmenter(in_channels=args.in_channels, num_classes=3, c_sar=c_sar).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, scaler, train=True, use_sar=args.use_sar)
        val_loss, miou, class_ious = run_epoch(model, val_loader, optimizer, criterion, device, scaler, train=False, use_sar=args.use_sar)
        scheduler.step()
        dt = time.time() - t0
        print(f"[Seg] Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  mIoU={miou:.4f} "
              f"class_IoUs=[{', '.join([f'{iou:.4f}' if iou == iou else 'NaN' for iou in class_ious])}]  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "segmenter_best.pt")
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_path)
            print(f"  -> saved best checkpoint to {ckpt_path}")

    print("Segmentation training complete.")


if __name__ == "__main__":
    main()
