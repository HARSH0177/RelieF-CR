"""
Phase 4: Dual-Evaluation Protocol for CloudFree Vision v2.

Addresses seasonal phenology drift (monsoon clouds vs. dry-season clear skies):
1. Regime A (Full-Reference — Synthetic Data Only):
   - Called when ground truth is available (temporal gap = 0).
   - Computes PSNR, SSIM, SAM (Spectral Angle Mapper), ERGAS, and Pearson CC.
2. Regime B (No-Reference & Structural — Real Monsoon Data):
   - Called on real satellite scenes where no same-day ground truth exists.
   - Computes No-Reference Perceptual Quality (NIQE/sharpness proxy).
   - Computes SAR-Optical Geometric Alignment (gradient_correlation via Sobel edge correlation).

Usage:
  python scripts/evaluate.py --synthetic --eval_mode synthetic --checkpoint checkpoints/generator_best.pt
  python scripts/evaluate.py --data_root dataset_root --eval_mode real --checkpoint checkpoints/generator_best.pt
  python scripts/evaluate.py --synthetic --eval_mode both --checkpoint checkpoints/generator_best.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset, SyntheticCloudDataset


# ==============================================================================
# Helper Functions & Core Metrics
# ==============================================================================

def to01(x_tanh):
    """Converts tensor from [-1, 1] range to [0, 1]."""
    return (x_tanh + 1.0) / 2.0


def masked_psnr(pred01, target01, mask):
    """Computes PSNR inside the cloud-masked region."""
    diff2 = (pred01 - target01) ** 2
    mse = (diff2 * mask).sum() / (mask.sum() + 1e-8)
    if mse.item() < 1e-12:
        return 99.0
    return (10 * torch.log10(1.0 / mse)).item()


def masked_ssim(pred01, target01, mask, window=7, C1=0.01 ** 2, C2=0.03 ** 2):
    """Computes band-averaged SSIM inside the cloud-masked region."""
    pad = window // 2
    B, C, H, W = pred01.shape
    ssim_sum = 0.0
    for c in range(C):
        p_c = pred01[:, c:c+1, :, :]
        t_c = target01[:, c:c+1, :, :]
        mu_p = F.avg_pool2d(p_c, window, stride=1, padding=pad)
        mu_t = F.avg_pool2d(t_c, window, stride=1, padding=pad)
        sigma_p = F.avg_pool2d(p_c ** 2, window, stride=1, padding=pad) - mu_p ** 2
        sigma_t = F.avg_pool2d(t_c ** 2, window, stride=1, padding=pad) - mu_t ** 2
        sigma_pt = F.avg_pool2d(p_c * t_c, window, stride=1, padding=pad) - mu_p * mu_t
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
            (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2)
        )
        mask_resized = mask if mask.shape[-2:] == ssim_map.shape[-2:] else F.interpolate(mask, size=ssim_map.shape[-2:])
        ssim_sum += ((ssim_map * mask_resized).sum() / (mask_resized.sum() + 1e-8)).item()
    return ssim_sum / C


def masked_sam(pred01, target01, mask):
    """Computes Spectral Angle Mapper (SAM) in degrees inside the cloud-masked region."""
    B, C, H, W = pred01.shape
    p = pred01.permute(0, 2, 3, 1).reshape(-1, C)
    t = target01.permute(0, 2, 3, 1).reshape(-1, C)
    m = mask.permute(0, 2, 3, 1).reshape(-1)
    dot = (p * t).sum(dim=1)
    norm_p = p.norm(dim=1) + 1e-8
    norm_t = t.norm(dim=1) + 1e-8
    cos_angle = torch.clamp(dot / (norm_p * norm_t), -1.0, 1.0)
    angle_deg = torch.acos(cos_angle) * 180.0 / np.pi
    masked_idx = m > 0.5
    if masked_idx.sum() == 0:
        return 0.0
    return angle_deg[masked_idx].mean().item()


def masked_ergas(pred01, target01, mask):
    """Computes ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)."""
    B, C, H, W = pred01.shape
    mask_bool = mask > 0.5
    if not mask_bool.any():
        return 0.0
    
    ergas_sum = 0.0
    for c in range(C):
        p_c = pred01[:, c:c+1, :, :]
        t_c = target01[:, c:c+1, :, :]
        diff2 = ((p_c - t_c) ** 2) * mask
        rmse2 = diff2.sum() / (mask.sum() + 1e-8)
        mu_k = (t_c * mask).sum() / (mask.sum() + 1e-8)
        if mu_k > 1e-8:
            ergas_sum += (rmse2 / (mu_k ** 2)).item()
    return 100.0 * np.sqrt(ergas_sum / C)


def masked_cc(pred01, target01, mask):
    """Computes Pearson Correlation Coefficient across bands inside masked region."""
    B, C, H, W = pred01.shape
    mask_bool = (mask > 0.5).expand_as(pred01)
    if not mask_bool.any():
        return 0.0
        
    cc_sum = 0.0
    for c in range(C):
        p_c = pred01[:, c, :, :][mask_bool[:, c, :, :]]
        t_c = target01[:, c, :, :][mask_bool[:, c, :, :]]
        if p_c.numel() == 0:
            continue
        mu_p = p_c.mean()
        mu_t = t_c.mean()
        p_diff = p_c - mu_p
        t_diff = t_c - mu_t
        cov = (p_diff * t_diff).sum()
        var_p = (p_diff ** 2).sum()
        var_t = (t_diff ** 2).sum()
        if var_p > 1e-8 and var_t > 1e-8:
            cc = cov / torch.sqrt(var_p * var_t)
            cc_sum += cc.item()
    return cc_sum / C


# ==============================================================================
# Regime A: Full-Reference Evaluation (Synthetic Data Only)
# ==============================================================================

def evaluate_full_reference(pred01, target01, mask):
    """
    Regime A: Evaluates full-reference reconstruction fidelity.
    Called when ground truth target image is available (e.g. synthetic cloud data).
    """
    psnr_val = masked_psnr(pred01, target01, mask)
    ssim_val = masked_ssim(pred01, target01, mask)
    sam_val = masked_sam(pred01, target01, mask)
    ergas_val = masked_ergas(pred01, target01, mask)
    cc_val = masked_cc(pred01, target01, mask)

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "sam": sam_val,
        "ergas": ergas_val,
        "cc": cc_val
    }


# ==============================================================================
# Regime B: No-Reference & Structural Evaluation (Real Monsoon Data)
# ==============================================================================

def gradient_correlation(pred01, sar_input, mask=None):
    """
    Computes mask-weighted Pearson correlation coefficient between edge magnitude maps
    of predicted optical image and input SAR VV image using Sobel filters.
    
    Uses standard weighted Pearson correlation where per-pixel weights are given by mask.
    This avoids the artificial boundary edge artifact caused by hard zeroing outside the mask,
    while properly evaluating geometric alignment inside the cloud-affected region.
    """
    device = pred01.device
    B, C, H, W = pred01.shape

    # Grayscale optical intensity (average across bands)
    opt_gray = pred01.mean(dim=1, keepdim=True)

    # SAR VV channel (index 0)
    sar_gray = sar_input[:, 0:1, :, :] if sar_input.shape[1] >= 1 else sar_input.mean(dim=1, keepdim=True)
    # Scale SAR to [0, 1] for edge extraction
    sar_min, sar_max = sar_gray.min(), sar_gray.max()
    sar_gray = (sar_gray - sar_min) / (sar_max - sar_min + 1e-8)

    # 2D Sobel filters
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    # Compute Sobel edge magnitude maps on the full unmasked images
    opt_gx = F.conv2d(opt_gray, sobel_x, padding=1)
    opt_gy = F.conv2d(opt_gray, sobel_y, padding=1)
    opt_edge = torch.sqrt(opt_gx**2 + opt_gy**2 + 1e-8)

    sar_gx = F.conv2d(sar_gray, sobel_x, padding=1)
    sar_gy = F.conv2d(sar_gray, sobel_y, padding=1)
    sar_edge = torch.sqrt(sar_gx**2 + sar_gy**2 + 1e-8)

    opt_flat = opt_edge.view(B, -1)
    sar_flat = sar_edge.view(B, -1)

    # Mask-weighted Pearson correlation formulation
    if mask is not None:
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        w = mask.view(B, -1)
        # If mask sum is near zero, fallback to unweighted evaluation
        w_sum = w.sum(dim=1, keepdim=True)
        fallback = (w_sum < 1e-5).float()
        w = w * (1.0 - fallback) + fallback * torch.ones_like(w)
        w_sum = w.sum(dim=1, keepdim=True) + 1e-8
    else:
        w = torch.ones_like(opt_flat)
        w_sum = w.sum(dim=1, keepdim=True) + 1e-8

    opt_wmean = (w * opt_flat).sum(dim=1, keepdim=True) / w_sum
    sar_wmean = (w * sar_flat).sum(dim=1, keepdim=True) / w_sum

    opt_diff = opt_flat - opt_wmean
    sar_diff = sar_flat - sar_wmean

    w_cov = (w * opt_diff * sar_diff).sum(dim=1) / w_sum.squeeze(1)
    w_var_opt = (w * (opt_diff ** 2)).sum(dim=1) / w_sum.squeeze(1)
    w_var_sar = (w * (sar_diff ** 2)).sum(dim=1) / w_sum.squeeze(1)

    denom = torch.sqrt(w_var_opt * w_var_sar) + 1e-8
    corr = w_cov / denom
    return float(corr.mean().item())


def laplacian_sharpness_score(pred01):
    """
    Custom Laplacian-variance sharpness proxy score.
    Computes spatial edge sharpness via the variance of Laplacian filtering per band,
    scaled by log1p(variance * 1000.0) * 15.0.
    
    DISCLOSURE: This is a lightweight, domain-agnostic spatial sharpness measure.
    It is NOT a learned natural scene statistics metric like NIQE (Mittal et al. 2012)
    or BRISQUE (Mittal et al. 2012). It measures high-frequency edge definition and
    contrast preservation, but does not benchmark against natural image statistical priors.
    """
    device = pred01.device
    B, C, H, W = pred01.shape
    laplacian_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device).view(1, 1, 3, 3)

    scores = []
    for c in range(C):
        band = pred01[:, c:c+1, :, :]
        lap = F.conv2d(band, laplacian_kernel, padding=1)
        sharpness = lap.var(dim=[-2, -1]).mean()
        score = float(torch.log1p(sharpness * 1000.0).item() * 15.0)
        scores.append(score)

    return float(np.mean(scores))


def evaluate_no_reference(pred01, sar_input, mask=None):
    """
    Regime B: Evaluates No-Reference perceptual quality and SAR-optical structural alignment.
    Called on real monsoon satellite data where no ground-truth clear image exists.
    """
    sharpness = laplacian_sharpness_score(pred01)
    sar_grad_corr = gradient_correlation(pred01, sar_input, mask=mask)

    return {
        "laplacian_sharpness_score": sharpness,
        "no_ref_perceptual_score": sharpness,  # alias for backward compatibility
        "sar_gradient_correlation": sar_grad_corr
    }


# ==============================================================================
# Dual Protocol Execution & Baselines
# ==============================================================================

def bicubic_baseline(opt_cloudy01, mask):
    """Bicubic inpainting baseline."""
    B, C, H, W = opt_cloudy01.shape
    small = F.interpolate(opt_cloudy01, size=(max(H // 8, 1), max(W // 8, 1)), mode="bicubic", align_corners=False)
    filled = F.interpolate(small, size=(H, W), mode="bicubic", align_corners=False)
    return opt_cloudy01 * (1 - mask) + filled * mask


def temporal_substitution_baseline(opt_cloudy01, temporal01, mask):
    """Temporal substitution baseline."""
    return opt_cloudy01 * (1 - mask) + temporal01 * mask


def run_evaluation(gen, loader, device, eval_mode="both"):
    """
    Runs Dual-Evaluation Protocol across data loader.
    Supports eval_mode: 'synthetic' (Regime A), 'real' (Regime B), or 'both'.
    """
    gen.eval()
    
    full_ref_results = {
        "model": {"psnr": [], "ssim": [], "sam": [], "ergas": [], "cc": []},
        "bicubic": {"psnr": [], "ssim": [], "sam": [], "ergas": [], "cc": []},
        "temporal_sub": {"psnr": [], "ssim": [], "sam": [], "ergas": [], "cc": []}
    }
    
    no_ref_results = {
        "model": {"laplacian_sharpness_score": [], "no_ref_perceptual_score": [], "sar_gradient_correlation": []},
        "bicubic": {"laplacian_sharpness_score": [], "no_ref_perceptual_score": [], "sar_gradient_correlation": []},
        "temporal_sub": {"laplacian_sharpness_score": [], "no_ref_perceptual_score": [], "sar_gradient_correlation": []}
    }

    calib_std, calib_err = [], []

    with torch.no_grad():
        for batch in loader:
            opt_cloudy = batch["opt_cloudy"].to(device)
            opt_clean = batch["opt_clean"].to(device)
            sar = batch["sar"].to(device)
            temporal = batch["temporal"].to(device)
            dem = batch["dem"].to(device)
            mask = batch["mask"].to(device)

            mean_fake, logvar_fake = gen(opt_cloudy, sar, temporal, dem)
            std_fake = torch.exp(0.5 * logvar_fake)

            cloudy01 = to01(opt_cloudy)
            clean01 = to01(opt_clean)
            temporal01 = to01(temporal)
            fake01 = to01(mean_fake).clamp(0, 1)

            bicubic01 = bicubic_baseline(cloudy01, mask).clamp(0, 1)
            temp_sub01 = temporal_substitution_baseline(cloudy01, temporal01, mask).clamp(0, 1)

            # Regime A: Full-Reference
            if eval_mode in ["synthetic", "both", "auto"]:
                for name, pred01 in [("model", fake01), ("bicubic", bicubic01), ("temporal_sub", temp_sub01)]:
                    res = evaluate_full_reference(pred01, clean01, mask)
                    for k, v in res.items():
                        full_ref_results[name][k].append(v)

            # Regime B: No-Reference & Structural
            if eval_mode in ["real", "both", "auto"]:
                for name, pred01 in [("model", fake01), ("bicubic", bicubic01), ("temporal_sub", temp_sub01)]:
                    res = evaluate_no_reference(pred01, sar, mask=mask)
                    for k, v in res.items():
                        no_ref_results[name][k].append(v)

            # Uncertainty calibration check
            mask_bool = mask.expand_as(std_fake) > 0.5
            if mask_bool.any():
                actual_err = (fake01 - clean01).abs()
                calib_std.append(std_fake[mask_bool].flatten().cpu().numpy())
                calib_err.append(actual_err[mask_bool].flatten().cpu().numpy())

    # Aggregate summaries
    summary_full_ref = {}
    for name, metrics in full_ref_results.items():
        summary_full_ref[name] = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}

    summary_no_ref = {}
    for name, metrics in no_ref_results.items():
        summary_no_ref[name] = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}

    calibration_corr = None
    if calib_std and calib_err:
        std_all = np.concatenate(calib_std)
        err_all = np.concatenate(calib_err)
        if std_all.std() > 1e-8 and err_all.std() > 1e-8:
            calibration_corr = float(np.corrcoef(std_all, err_all)[0, 1])

    return summary_full_ref, summary_no_ref, calibration_corr


def main():
    p = argparse.ArgumentParser(description="Phase 4 Dual-Evaluation Protocol")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--synthetic", action="store_true", help="Evaluate on synthetic data")
    p.add_argument("--synthetic_n", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--eval_mode", type=str, choices=["synthetic", "real", "both", "auto"], default="both",
                   help="Regime A (synthetic full-reference), Regime B (real no-reference), or both")
    p.add_argument("--use_raw_weights", action="store_true",
                   help="Use raw generator weights instead of recommended EMA weights.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        ds = SyntheticCloudDataset(n_samples=args.synthetic_n, patch_size=args.patch_size, severity=0.8, seed=2)
    else:
        if not args.data_root:
            print("ERROR: Must provide --data_root or --synthetic flag.")
            sys.exit(1)
        ds = CloudReconstructionDataset(args.data_root, split="test")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    gen = CloudReconstructionGeneratorV2(base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_key = "gen_state" if args.use_raw_weights else "gen_ema_state"
    if state_key not in ckpt:
        state_key = "gen_state"
    gen.load_state_dict(ckpt[state_key], strict=False)
    print(f"Loaded '{state_key}' from checkpoint (epoch {ckpt.get('epoch', '?')})", flush=True)

    summary_full_ref, summary_no_ref, calibration_corr = run_evaluation(gen, loader, device, eval_mode=args.eval_mode)

    print("\n" + "=" * 90)
    print("=== DUAL-EVALUATION PROTOCOL RESULTS ===")
    print("=" * 90)

    if args.eval_mode in ["synthetic", "both", "auto"]:
        print("\n--- REGIME A: Full-Reference Metrics (Synthetic / Controlled Ground Truth) ---")
        print(f"{'Method':<15}{'PSNR (dB)':>12}{'SSIM':>10}{'SAM (deg)':>12}{'ERGAS':>10}{'CC':>10}")
        print("-" * 70)
        for name in ["temporal_sub", "bicubic", "model"]:
            m = summary_full_ref[name]
            print(f"{name:<15}{m['psnr']:>12.2f}{m['ssim']:>10.3f}{m['sam']:>12.2f}{m['ergas']:>10.2f}{m['cc']:>10.3f}")

    if args.eval_mode in ["real", "both", "auto"]:
        print("\n--- REGIME B: No-Reference & Structural Metrics (Real Monsoon Data) ---")
        print(f"{'Method':<15}{'Laplacian Sharpness':>24}{'SAR-Optical Grad Corr':>25}")
        print("-" * 70)
        for name in ["temporal_sub", "bicubic", "model"]:
            m = summary_no_ref[name]
            print(f"{name:<15}{m['laplacian_sharpness_score']:>24.2f}{m['sar_gradient_correlation']:>25.3f}")

    print("\n--- Uncertainty Calibration ---")
    if calibration_corr is not None:
        print(f"corr(predicted_std, actual_error) = {calibration_corr:.3f}")
    else:
        print("Not enough masked pixels to compute calibration correlation.")

    print("\nDual-Evaluation Complete.")


if __name__ == "__main__":
    main()
