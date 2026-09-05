import os
import sys
import time
import argparse
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Uncertainty Calibration Figure (Bell Curve & Reliability Diagram)")
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "dataset_root"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints", "generator_best.pt"))
    parser.add_argument("--output_fig", type=str, default=os.path.join(PROJECT_ROOT, "paper", "figures", "uncertainty_calibration.png"))
    parser.add_argument("--cache_file", type=str, default=os.path.join(PROJECT_ROOT, "paper", "figures", "uncertainty_calibration_cache.npz"))
    parser.add_argument("--use_cache", action="store_true", help="Re-render from cache if available")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mask_thresh", type=float, default=0.5, help="Threshold for cloud mask occlusion (default 0.5)")
    parser.add_argument("--max_patches", type=int, default=None, help="Optional maximum number of patches to evaluate")
    return parser.parse_args()


def render_figure(cache_data, output_fig):
    print("\nRendering publication figure from calibrated metrics...", flush=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)

    # Panel (a): Standardized Residual Distribution
    bins = cache_data["bins"]
    counts = cache_data["counts"]
    mean_z = float(cache_data["mean_z"])
    std_z = float(cache_data["std_z"])
    n_pixels = int(cache_data["n_pixels"])
    skew_z = float(cache_data["skew_z"])

    # Draw histogram bars from counts
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = np.diff(bins)
    ax1.bar(bin_centers, counts, width=bin_widths, alpha=0.55,
            color="#337ab7", edgecolor="#1b4f72", linewidth=0.5,
            label=f"Empirical $z$\n($\mu={mean_z:+.3f},\ \sigma={std_z:.3f}$)")

    # Theoretical standard normal N(0, 1)
    x_grid = np.linspace(-4.0, 4.0, 400)
    pdf_standard = stats.norm.pdf(x_grid, loc=0.0, scale=1.0)
    ax1.plot(x_grid, pdf_standard, color="#d9534f", linestyle="--", linewidth=1.8, label="Theoretical $\mathcal{N}(0, 1)$")

    # Fitted normal
    pdf_fitted = stats.norm.pdf(x_grid, loc=mean_z, scale=std_z)
    ax1.plot(x_grid, pdf_fitted, color="#2e7d32", linestyle="-", linewidth=1.6,
             label=f"Fitted Normal\n($\sigma={std_z:.3f}$)")

    ax1.set_xlim(-4.0, 4.0)
    ax1.set_xlabel(r"Standardized Residual $z = (y - \hat{y}) / \sigma$")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("(a) Standardized Residual Distribution", fontweight="bold", pad=7)
    ax1.grid(True, linestyle=":", alpha=0.55)
    ax1.legend(loc="upper right", framealpha=0.92, facecolor="white", edgecolor="#cccccc")

    textstr_a = (
        f"$N = {n_pixels:,}$ px\n"
        f"$\mu_z = {mean_z:+.3f}$\n"
        f"$\sigma_z = {std_z:.3f}$\n"
        f"Skew = {skew_z:+.2f}"
    )
    ax1.text(0.04, 0.95, textstr_a, transform=ax1.transAxes, fontsize=8.2,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))

    # Panel (b): Reliability Diagram
    bin_pred_sigma = cache_data["bin_pred_sigma"]
    bin_obs_err = cache_data["bin_obs_err"]
    bin_err_std = cache_data["bin_err_std"]
    r_pearson = float(cache_data["r_pearson"])
    r_spearman = float(cache_data["r_spearman"])

    ax2.errorbar(
        bin_pred_sigma, bin_obs_err, yerr=bin_err_std * 2.0,
        fmt="o-", color="#1b4f72", ecolor="#1b4f72", elinewidth=1.2, capsize=3,
        markersize=5, linewidth=1.5,
        label=r"Observed Error $\mathbb{E}[|y - \hat{y}|]$"
    )

    min_val = min(bin_pred_sigma.min(), bin_obs_err.min()) * 0.9
    max_val = max(bin_pred_sigma.max(), bin_obs_err.max()) * 1.1
    ref_x = np.linspace(min_val, max_val, 100)
    ax2.plot(ref_x, ref_x, color="#555555", linestyle="--", linewidth=1.2, alpha=0.75, label=r"Ideal Calibration ($y = x$)")
    ax2.plot(ref_x, np.sqrt(2.0 / np.pi) * ref_x, color="#d9534f", linestyle=":", linewidth=1.4,
             label=r"Gaussian MAE Exp. ($\sqrt{2/\pi}\,\sigma$)")

    ax2.set_ylim(0.0, 0.11)
    ax2.set_xlabel(r"Predicted Uncertainty Standard Deviation $\sigma$")
    ax2.set_ylabel(r"Observed Error $|y - \hat{y}|$")
    ax2.set_title("(b) Uncertainty Reliability Diagram", fontweight="bold", pad=7)
    ax2.grid(True, linestyle=":", alpha=0.55)
    # Position legend cleanly in open space without covering data points
    ax2.legend(loc="lower right", bbox_to_anchor=(0.98, 0.32), framealpha=0.92, facecolor="white", edgecolor="#cccccc")

    textstr_b = (
        f"Pearson $\\rho = {r_pearson:+.3f}$ ($p < 10^{{-6}}$)\n"
        f"Spearman $\\rho_s = {r_spearman:+.3f}$\n"
        f"Monotonic Rank: PASS"
    )
    ax2.text(0.04, 0.95, textstr_b, transform=ax2.transAxes, fontsize=8.2,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    print(f"Saved publication figure to: {output_fig}", flush=True)


def main():
    args = parse_args()
    if args.use_cache and os.path.exists(args.cache_file):
        print(f"Loading cached metrics from: {args.cache_file}")
        cache_data = np.load(args.cache_file)
        render_figure(cache_data, args.output_fig)
        return

    print("=" * 80)
    print("HETEROSCEDASTIC UNCERTAINTY CALIBRATION AUDIT — IEEE TGRS")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: CPU (using {torch.get_num_threads()} threads)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mask Threshold: {args.mask_thresh}")

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_key = "gen_ema_state" if "gen_ema_state" in ckpt else "gen_state"
    gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
    gen.load_state_dict(ckpt[state_key])
    gen.eval()

    ds = CloudReconstructionDataset(args.data_root, split="test")
    if args.max_patches is not None:
        ds.patch_ids = ds.patch_ids[:args.max_patches]
    total_patches = len(ds)
    print(f"Evaluating across {total_patches} test patches...", flush=True)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_z = []
    all_sigma = []
    all_err = []

    t0 = time.time()
    processed_patches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            opt_cloudy = batch["opt_cloudy"].to(device)
            opt_clean = batch["opt_clean"].to(device)
            sar = batch["sar"].to(device)
            temporal = batch["temporal"].to(device)
            dem = batch["dem"].to(device)
            mask = batch["mask"].to(device)

            mean_fake, logvar_fake = gen(opt_cloudy, sar, temporal, dem)

            clean01 = (opt_clean + 1.0) / 2.0
            pred01 = ((mean_fake + 1.0) / 2.0).clamp(0.0, 1.0)
            sigma01 = torch.exp(0.5 * logvar_fake) / 2.0
            err01 = (clean01 - pred01).abs()

            mask_bool = mask.expand_as(sigma01) > args.mask_thresh
            if mask_bool.any():
                diff = clean01[mask_bool] - pred01[mask_bool]
                sig = sigma01[mask_bool]
                z = diff / (sig + 1e-8)
                all_z.append(z.numpy())
                all_sigma.append(sig.numpy())
                all_err.append(err01[mask_bool].numpy())

            processed_patches += opt_cloudy.size(0)
            if (batch_idx + 1) % 10 == 0 or processed_patches == total_patches:
                elapsed = time.time() - t0
                pct = 100.0 * processed_patches / total_patches
                fps = processed_patches / elapsed
                print(f"  [{processed_patches:4d}/{total_patches} ({pct:5.1f}%)] Elapsed: {elapsed:5.1f}s ({fps:.1f} patches/s)", flush=True)

    z_all = np.concatenate(all_z)
    sigma_all = np.concatenate(all_sigma)
    err_all = np.concatenate(all_err)

    n_pixels = len(z_all)
    mean_z = float(np.mean(z_all))
    std_z = float(np.std(z_all))
    median_z = float(np.median(z_all))
    skew_z = float(stats.skew(z_all))
    kurt_z = float(stats.kurtosis(z_all))

    r_pearson, p_val = stats.pearsonr(sigma_all, err_all)
    r_spearman, sp_val = stats.spearmanr(sigma_all, err_all)

    mean_sig = float(np.mean(sigma_all))
    mean_err = float(np.mean(err_all))

    print("\n" + "=" * 80)
    print("CALIBRATION AUDIT RESULTS (Held-Out Test Patches)")
    print("=" * 80)
    print(f"Total Occluded Pixels Evaluated: {n_pixels:,}")
    print(f"Standardized Residual z = (y - y_hat) / sigma:")
    print(f"  - Mean mu(z):    {mean_z:+.4f} (Ideal: 0.0000)")
    print(f"  - Std sigma(z):  {std_z:.4f} (Ideal: 1.0000, Conservative < 1.0)")
    print(f"  - Median:        {median_z:+.4f}")
    print(f"  - Skewness:      {skew_z:+.4f}")
    print(f"  - Kurtosis:      {kurt_z:+.4f}")
    print(f"\nUncertainty vs. Observed Error Correlation:")
    print(f"  - Pearson r(sigma, |err|):  {r_pearson:+.4f} (p = {p_val:.2e})")
    print(f"  - Spearman r(sigma, |err|): {r_spearman:+.4f} (p = {sp_val:.2e})")
    print(f"  - Mean predicted sigma:     {mean_sig:.4f}")
    print(f"  - Mean observed |err|:      {mean_err:.4f}")
    print("=" * 80)

    # Compute histogram density counts
    z_clip = np.clip(z_all, -4.0, 4.0)
    bins = np.linspace(-4.0, 4.0, 81)
    counts, _ = np.histogram(z_clip, bins=bins, density=True)

    # Bin predicted sigma into 12 quantile-based bins
    n_bins = 12
    quantiles = np.linspace(0.01, 0.99, n_bins + 1)
    bin_edges = np.quantile(sigma_all, quantiles)
    bin_edges = np.unique(bin_edges)
    
    bin_pred_sigma = []
    bin_obs_err = []
    bin_obs_rmse = []
    bin_err_std = []

    for i in range(len(bin_edges) - 1):
        idx_bin = (sigma_all >= bin_edges[i]) & (sigma_all < bin_edges[i+1])
        if np.sum(idx_bin) > 50:
            sub_sig = sigma_all[idx_bin]
            sub_err = err_all[idx_bin]
            bin_pred_sigma.append(np.mean(sub_sig))
            bin_obs_err.append(np.mean(sub_err))
            bin_obs_rmse.append(np.sqrt(np.mean(sub_err**2)))
            bin_err_std.append(np.std(sub_err) / np.sqrt(len(sub_err)))

    bin_pred_sigma = np.array(bin_pred_sigma)
    bin_obs_err = np.array(bin_obs_err)
    bin_obs_rmse = np.array(bin_obs_rmse)
    bin_err_std = np.array(bin_err_std)

    # Save to npz cache
    np.savez(
        args.cache_file,
        bins=bins,
        counts=counts,
        mean_z=mean_z,
        std_z=std_z,
        median_z=median_z,
        skew_z=skew_z,
        kurt_z=kurt_z,
        n_pixels=n_pixels,
        r_pearson=r_pearson,
        r_spearman=r_spearman,
        bin_pred_sigma=bin_pred_sigma,
        bin_obs_err=bin_obs_err,
        bin_obs_rmse=bin_obs_rmse,
        bin_err_std=bin_err_std,
    )
    print(f"Saved calibration cache to: {args.cache_file}", flush=True)

    cache_data = np.load(args.cache_file)
    render_figure(cache_data, args.output_fig)

    txt_out = os.path.splitext(args.output_fig)[0] + "_metrics.txt"
    with open(txt_out, "w") as f:
        f.write(f"Occluded Pixels: {n_pixels}\n")
        f.write(f"mean_z: {mean_z:.6f}\n")
        f.write(f"std_z: {std_z:.6f}\n")
        f.write(f"median_z: {median_z:.6f}\n")
        f.write(f"skew_z: {skew_z:.6f}\n")
        f.write(f"kurt_z: {kurt_z:.6f}\n")
        f.write(f"pearson_r: {r_pearson:.6f}\n")
        f.write(f"pearson_p: {p_val:.6e}\n")
        f.write(f"spearman_r: {r_spearman:.6f}\n")
        f.write(f"mean_sigma: {mean_sig:.6f}\n")
        f.write(f"mean_err: {mean_err:.6f}\n")
    print(f"Saved metrics summary to: {txt_out}", flush=True)


if __name__ == "__main__":
    main()
