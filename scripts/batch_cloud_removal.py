"""
Batch Cloud Removal and Image Export for CloudFree Vision v2.

Iterates over test cloudy patches, removes clouds via the trained generator,
and exports:
1. Reconstructed clear-sky images (.png)
2. Original cloudy images (.png)
3. Multi-modal 6-panel comparison figures (.png)
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset


def false_color(arr):
    # [Green, Red, NIR] -> [NIR, Red, Green]
    c = arr.shape[0]
    if c >= 3:
        disp = arr[[2, 1, 0]]
    else:
        disp = np.repeat(arr[:1], 3, axis=0)
    return np.clip(disp.transpose(1, 2, 0), 0, 1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Batch Cloud Removal on device: {device}")

    ckpt_path = os.path.join("checkpoints", "generator_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
    state_key = "gen_ema_state" if "gen_ema_state" in ckpt else "gen_state"
    gen.load_state_dict(ckpt[state_key])
    gen.eval()
    print(f"Loaded model from {ckpt_path}")

    ds_test = CloudReconstructionDataset("dataset_root", split="test")
    print(f"Total test patches available: {len(ds_test)}")

    out_dir = os.path.join("outputs", "reconstructions")
    os.makedirs(out_dir, exist_ok=True)

    scenes_seen = set()
    indices_to_save = []
    for idx in range(len(ds_test)):
        pid = ds_test.patch_ids[idx]
        scene_name = "_".join(pid.split("_")[:2])
        if scene_name not in scenes_seen:
            scenes_seen.add(scene_name)
            indices_to_save.append(idx)
        if len(indices_to_save) >= 8:
            break

    extra_indices = [15, 45, 80, 120, 250, 400]
    indices_to_save = list(dict.fromkeys(indices_to_save + [i for i in extra_indices if i < len(ds_test)]))[:10]
    print(f"Processing {len(indices_to_save)} representative test patches across scenes...")

    with torch.no_grad():
        for rank, idx in enumerate(indices_to_save):
            sample = ds_test[idx]
            pid = ds_test.patch_ids[idx].replace(".npy", "")

            opt_cloudy = sample["opt_cloudy"].unsqueeze(0).to(device)
            sar = sample["sar"].unsqueeze(0).to(device)
            temporal = sample["temporal"].unsqueeze(0).to(device)
            dem = sample["dem"].unsqueeze(0).to(device)
            opt_clean = sample["opt_clean"]
            mask = sample["mask"]

            mean_fake, logvar_fake = gen(opt_cloudy, sar, temporal, dem)

            recon_01 = ((mean_fake[0].clamp(-1, 1) + 1) / 2).cpu().numpy()
            cloudy_01 = ((sample["opt_cloudy"] + 1) / 2).numpy()
            clean_01 = ((opt_clean + 1) / 2).numpy()
            sar_vv = sample["sar"][0].numpy()
            if sar_vv.max() > sar_vv.min() + 1e-6:
                p2, p98 = np.percentile(sar_vv, 2), np.percentile(sar_vv, 98)
                if p98 > p2:
                    sar_disp = np.clip((sar_vv - p2) / (p98 - p2), 0, 1)
                else:
                    sar_disp = (sar_vv - sar_vv.min()) / (sar_vv.max() - sar_vv.min() + 1e-8)
            else:
                sar_disp = np.clip((sar_vv + 1.0) / 2.0, 0, 1)

            dem_elev = sample["dem"][0].numpy()
            mask_2d = mask[0].numpy() if mask.ndim == 3 else mask
            uncertainty = torch.exp(0.5 * logvar_fake)[0].mean(dim=0).cpu().numpy()

            # 1. Save Standalone Reconstructed Clear-Sky Image
            recon_rgb = (false_color(recon_01) * 255).astype(np.uint8)
            recon_png = os.path.join(out_dir, f"{pid}_reconstructed_clear.png")
            Image.fromarray(recon_rgb).save(recon_png)

            # 2. Save Standalone Cloudy Input Image
            cloudy_rgb = (false_color(cloudy_01) * 255).astype(np.uint8)
            cloudy_png = os.path.join(out_dir, f"{pid}_cloudy_input.png")
            Image.fromarray(cloudy_rgb).save(cloudy_png)

            # 3. Save Comprehensive 6-Panel Comparison Figure
            fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))

            axes[0].imshow(false_color(cloudy_01))
            axes[0].set_title("1. Cloudy Input (LISS-IV)", fontsize=9, fontweight="bold")

            axes[1].imshow(mask_2d, cmap="Blues_r", vmin=0, vmax=1)
            axes[1].set_title("2. Cloud Opacity Mask", fontsize=9, fontweight="bold")

            axes[2].imshow(sar_disp, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("3. Sentinel-1 SAR (VV)", fontsize=9, fontweight="bold")

            axes[3].imshow(dem_elev, cmap="terrain")
            axes[3].set_title("4. CartoDEM Topography", fontsize=9, fontweight="bold")

            axes[4].imshow(false_color(recon_01))
            axes[4].set_title("5. CloudFree Output", fontsize=9, fontweight="bold", color="green")

            axes[5].imshow(false_color(clean_01))
            axes[5].set_title("6. Ground Truth Clear", fontsize=9, fontweight="bold")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            comp_png = os.path.join(out_dir, f"{pid}_comparison.png")
            plt.savefig(comp_png, dpi=200, bbox_inches="tight")
            plt.close(fig)

            print(f"[{rank+1}/{len(indices_to_save)}] Saved: {pid} -> {comp_png}")

    print(f"\nAll {len(indices_to_save)} clear-sky reconstructions successfully saved to {out_dir}!")


if __name__ == "__main__":
    main()
