"""
NEW IN V2 - generates a qualitative comparison grid PNG directly reusable as a
pitch-deck / demo-day figure: rows = samples, columns = [cloudy input,
cloud/shadow mask, SAR, reconstruction, ground truth, uncertainty heatmap].

Usage:
  python scripts/make_comparison_grid.py --synthetic --checkpoint checkpoints/generator_best.pt --n_samples 4
  python scripts/make_comparison_grid.py --data_root /path/to/dataset_root --checkpoint checkpoints/generator_best.pt
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset, SyntheticCloudDataset


def false_color(x_chw01):
    """[Green,Red,NIR] -> display as [NIR,Red,Green], standard CIR composite."""
    c = x_chw01.shape[0]
    if c >= 3:
        disp = x_chw01[[2, 1, 0]]
    else:
        disp = np.repeat(x_chw01[:1], 3, axis=0)
    return np.clip(disp.transpose(1, 2, 0), 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/comparison_grid.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        ds = SyntheticCloudDataset(n_samples=args.n_samples, patch_size=args.patch_size, severity=0.7, seed=3)
    else:
        ds = CloudReconstructionDataset(args.data_root, split="test")
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    gen = CloudReconstructionGeneratorV2(base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_key = "gen_ema_state" if "gen_ema_state" in ckpt else "gen_state"
    gen.load_state_dict(ckpt[state_key])
    gen.eval()

    n = min(args.n_samples, len(ds))
    cols = ["Cloudy Input", "Cloud/Shadow Mask", "SAR (VV)", "Reconstruction", "Ground Truth", "Uncertainty"]
    fig, axes = plt.subplots(n, len(cols), figsize=(3.1 * len(cols), 3.1 * n))
    if n == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n:
                break
            opt_cloudy = batch["opt_cloudy"].to(device)
            opt_clean = batch["opt_clean"].to(device)
            sar = batch["sar"].to(device)
            temporal = batch["temporal"].to(device)
            dem = batch["dem"].to(device)
            mask_class = batch["mask_class"][0].numpy()

            mean_out, logvar_out = gen(opt_cloudy, sar, temporal, dem)
            std_out = torch.exp(0.5 * logvar_out)[0].mean(dim=0).cpu().numpy()

            cloudy01 = ((opt_cloudy[0] + 1) / 2).cpu().numpy()
            clean01 = ((opt_clean[0] + 1) / 2).cpu().numpy()
            fake01 = ((mean_out[0] + 1) / 2).clamp(0, 1).cpu().numpy()
            sar_band0 = sar[0, 0].cpu().numpy()

            axes[i, 0].imshow(false_color(cloudy01))
            im_mask = axes[i, 1].imshow(mask_class, cmap="viridis", vmin=0, vmax=2)
            axes[i, 2].imshow(sar_band0, cmap="gray")
            axes[i, 3].imshow(false_color(fake01))
            axes[i, 4].imshow(false_color(clean01))
            im_u = axes[i, 5].imshow(std_out, cmap="magma")

            for j in range(len(cols)):
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(cols[j], fontsize=11, fontweight='bold')

            # Colorbars for mask and uncertainty
            if i == n - 1:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                # Mask colorbar (discrete 0/1/2)
                divider_m = make_axes_locatable(axes[i, 1])
                cax_m = divider_m.append_axes("bottom", size="5%", pad=0.05)
                cb_m = fig.colorbar(im_mask, cax=cax_m, orientation="horizontal", ticks=[0, 1, 2])
                cb_m.ax.set_xticklabels(["Clear", "Cloud", "Shadow"], fontsize=7)
                # Uncertainty colorbar
                divider_u = make_axes_locatable(axes[i, 5])
                cax_u = divider_u.append_axes("bottom", size="5%", pad=0.05)
                cb_u = fig.colorbar(im_u, cax=cax_u, orientation="horizontal")
                cb_u.ax.tick_params(labelsize=7)
                cb_u.set_label("Pred. Std", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    # Also save PDF for vector quality
    pdf_path = args.out.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison grid: {args.out} + {pdf_path}")


if __name__ == "__main__":
    main()
