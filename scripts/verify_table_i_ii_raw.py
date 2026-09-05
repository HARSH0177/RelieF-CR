"""
RAW EVIDENCE VERIFICATION SCRIPT for Table I and Table II.
Runs live on CPU. Prints every metric for every patch. No JSON, no summaries.

Task 1: Full evaluate.py-style run across all 1,611 test patches with running averages.
Task 2: Per-patch PSNR/SSIM/SAM/CC for 20 randomly sampled patches (seeded, reproducible).
Task 3: SAR-Optical gradient correlation for the same 20 patches.
Task 4: Checkpoint identity confirmation (printed at top).

IMPORTANT: This script uses the EXACT SAME metric functions from evaluate.py.
           No new code, no wrappers — imported directly.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset

# Import the EXACT metric functions from evaluate.py
from scripts.evaluate import (
    to01,
    masked_psnr,
    masked_ssim,
    masked_sam,
    masked_ergas,
    masked_cc,
    gradient_correlation,
)
import argparse


def main():
    parser = argparse.ArgumentParser(description="Raw Evidence Verification for Table I and Table II")
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "dataset_root"), help="Path to dataset root")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints", "generator_best.pt"), help="Path to checkpoint")
    parser.add_argument("--max_patches", type=int, default=None, help="Optional limit on patches to evaluate")
    args = parser.parse_args()

    print("=" * 90)
    print("RAW EVIDENCE VERIFICATION — Table I & Table II")
    print("=" * 90)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print()

    # =========================================================================
    # TASK 4: Checkpoint identity
    # =========================================================================
    ckpt_path = args.checkpoint
    print("=" * 90)
    print("TASK 4: CHECKPOINT IDENTITY CONFIRMATION")
    print("=" * 90)
    print(f"Path: {ckpt_path}")
    print(f"File size: {os.path.getsize(ckpt_path)} bytes")
    print(f"File mtime: {time.ctime(os.path.getmtime(ckpt_path))}")

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Keys in checkpoint: {list(ckpt.keys())}")
    print(f"Saved epoch: {ckpt.get('epoch', '?')}")
    print(f"Saved val_psnr: {ckpt.get('val_psnr', '?')}")
    print(f"Using state dict: gen_ema_state ({len(ckpt['gen_ema_state'])} keys)")
    print()

    # =========================================================================
    # Load model
    # =========================================================================
    gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
    gen.load_state_dict(ckpt["gen_ema_state"], strict=False)
    gen.eval()
    print(f"Generator loaded. Total parameters: {sum(p.numel() for p in gen.parameters()):,}")
    print()

    # =========================================================================
    # Load dataset
    # =========================================================================
    data_root = args.data_root
    ds = CloudReconstructionDataset(data_root, split="test")
    print(f"Test dataset loaded: {len(ds)} patches")
    print()

    # =========================================================================
    # Select 20 random patches (fixed seed for reproducibility)
    # =========================================================================
    rng = random.Random(42)
    sample_indices = sorted(rng.sample(range(len(ds)), min(20, len(ds))))

    # Get patch filenames for the sampled indices
    patch_dir = os.path.join(data_root, "test", "opt_clean")
    all_patch_files = sorted(os.listdir(patch_dir))
    print("TASK 2 & 3: 20 randomly sampled patch IDs (seed=42):")
    for i, idx in enumerate(sample_indices):
        print(f"  [{i+1:2d}] index={idx:4d}  file={all_patch_files[idx]}")
    print()

    # =========================================================================
    # TASK 1: Full evaluation across all test patches
    # TASK 2: Per-patch metrics for the 20 sampled patches
    # TASK 3: SAR-Optical gradient correlation for the 20 sampled patches
    # =========================================================================
    print("=" * 90)
    print("TASK 1: FULL 1,611-PATCH EVALUATION (every 100th patch printed)")
    print("TASK 2 & 3: Per-patch detail printed inline for 20 sampled patches")
    print("=" * 90)
    print()

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Accumulators for full-run averages
    all_psnr, all_ssim, all_sam, all_ergas, all_cc = [], [], [], [], []
    all_grad_corr_model = []
    all_grad_corr_gt = []

    # Per-patch detail for 20 sampled patches
    sampled_details = []

    sample_set = set(sample_indices)
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            opt_cloudy = batch["opt_cloudy"].to(device)
            opt_clean = batch["opt_clean"].to(device)
            sar = batch["sar"].to(device)
            temporal = batch["temporal"].to(device)
            dem = batch["dem"].to(device)
            mask = batch["mask"].to(device)

            mean_fake, logvar_fake = gen(opt_cloudy, sar, temporal, dem)

            clean01 = to01(opt_clean).clamp(0, 1)
            fake01 = to01(mean_fake).clamp(0, 1)

            # Full-reference metrics (inside cloud mask)
            psnr_val = masked_psnr(fake01, clean01, mask)
            ssim_val = masked_ssim(fake01, clean01, mask)
            sam_val = masked_sam(fake01, clean01, mask)
            ergas_val = masked_ergas(fake01, clean01, mask)
            cc_val = masked_cc(fake01, clean01, mask)

            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_sam.append(sam_val)
            all_ergas.append(ergas_val)
            all_cc.append(cc_val)

            # SAR-Optical gradient correlation (model vs SAR, and GT vs SAR)
            grad_corr_model = gradient_correlation(fake01, sar, mask=mask)
            grad_corr_gt = gradient_correlation(clean01, sar, mask=mask)
            all_grad_corr_model.append(grad_corr_model)
            all_grad_corr_gt.append(grad_corr_gt)

            # Print every 100th patch as running progress
            if batch_idx % 100 == 0:
                elapsed = time.time() - t0
                running_psnr = np.mean(all_psnr)
                running_ssim = np.mean(all_ssim)
                running_sam = np.mean(all_sam)
                running_ergas = np.mean(all_ergas)
                running_cc = np.mean(all_cc)
                print(
                    f"[{batch_idx+1:4d}/{len(ds)}] "
                    f"PSNR={psnr_val:6.2f} SSIM={ssim_val:.4f} SAM={sam_val:.3f} "
                    f"ERGAS={ergas_val:6.2f} CC={cc_val:.4f} "
                    f"| Running: PSNR={running_psnr:.2f} SSIM={running_ssim:.4f} "
                    f"SAM={running_sam:.3f} ERGAS={running_ergas:.2f} CC={running_cc:.4f} "
                    f"| {elapsed:.0f}s",
                    flush=True,
                )

            # If this is one of the 20 sampled patches, print full detail
            if batch_idx in sample_set:
                fname = all_patch_files[batch_idx]
                cloud_pct = float(mask.sum()) / float(mask.numel()) * 100.0
                detail = {
                    "idx": batch_idx,
                    "file": fname,
                    "cloud_pct": cloud_pct,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "sam": sam_val,
                    "ergas": ergas_val,
                    "cc": cc_val,
                    "grad_corr_model": grad_corr_model,
                    "grad_corr_gt": grad_corr_gt,
                }
                sampled_details.append(detail)
                print(
                    f"  >>> SAMPLED PATCH {fname} (cloud={cloud_pct:.1f}%): "
                    f"PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} SAM={sam_val:.3f} "
                    f"CC={cc_val:.4f} "
                    f"GradCorr(model,SAR)={grad_corr_model:.4f} "
                    f"GradCorr(GT,SAR)={grad_corr_gt:.4f}",
                    flush=True,
                )

    elapsed_total = time.time() - t0

    # =========================================================================
    # FINAL AGGREGATED RESULTS
    # =========================================================================
    print()
    print("=" * 90)
    print("TASK 1 FINAL: AGGREGATED TABLE I METRICS (all test patches)")
    print("=" * 90)
    print(f"Total patches evaluated: {len(all_psnr)}")
    print(f"Total wall time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print()
    print(f"  PSNR  (dB):  {np.mean(all_psnr):.2f}  (std={np.std(all_psnr):.2f}, min={np.min(all_psnr):.2f}, max={np.max(all_psnr):.2f})")
    print(f"  SSIM:        {np.mean(all_ssim):.4f}  (std={np.std(all_ssim):.4f})")
    print(f"  SAM  (deg):  {np.mean(all_sam):.3f}  (std={np.std(all_sam):.3f})")
    print(f"  ERGAS:       {np.mean(all_ergas):.2f}  (std={np.std(all_ergas):.2f})")
    print(f"  CC:          {np.mean(all_cc):.4f}  (std={np.std(all_cc):.4f})")
    print()
    print(f"  SAR-Optical Gradient Correlation (Model vs SAR):  mean={np.mean(all_grad_corr_model):.4f}")
    print(f"  SAR-Optical Gradient Correlation (GT vs SAR):     mean={np.mean(all_grad_corr_gt):.4f}")
    print(f"  Delta (Model - GT):                               {np.mean(all_grad_corr_model) - np.mean(all_grad_corr_gt):.4f}")
    print()

    # =========================================================================
    # TASK 2 & 3: Per-patch detail table for 20 sampled patches
    # =========================================================================
    print("=" * 90)
    print("TASK 2 & 3: PER-PATCH DETAIL FOR 20 RANDOMLY SAMPLED PATCHES")
    print("=" * 90)
    header = f"{'#':>3} {'Patch File':<22} {'Cloud%':>7} {'PSNR':>7} {'SSIM':>7} {'SAM':>6} {'ERGAS':>7} {'CC':>7} {'GC_Mod':>7} {'GC_GT':>7}"
    print(header)
    print("-" * len(header))
    for i, d in enumerate(sampled_details):
        print(
            f"{i+1:3d} {d['file']:<22} {d['cloud_pct']:6.1f}% "
            f"{d['psnr']:7.2f} {d['ssim']:7.4f} {d['sam']:6.3f} "
            f"{d['ergas']:7.2f} {d['cc']:7.4f} "
            f"{d['grad_corr_model']:7.4f} {d['grad_corr_gt']:7.4f}"
        )
    print()

    print("=" * 90)
    print("VERIFICATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
