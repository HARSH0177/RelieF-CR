"""
Showcase generator for high-contrast test patches.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset
from scripts.evaluate import masked_psnr, masked_ssim


def enhance_display(img_chw):
    # img_chw: [3, H, W] in [0, 1] range: [Green, Red, NIR]
    # Display as CIR Composite: [NIR, Red, Green]
    nir = img_chw[2]
    red = img_chw[1]
    green = img_chw[0]
    rgb = np.stack([nir, red, green], axis=-1)

    p2 = np.percentile(rgb, 1)
    p98 = np.percentile(rgb, 99)
    if p98 > p2:
        rgb_stretched = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    else:
        rgb_stretched = np.clip(rgb, 0.0, 1.0)
    return rgb_stretched


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join("checkpoints", "generator_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
    gen.load_state_dict(ckpt["gen_ema_state"])
    gen.eval()

    ds_test = CloudReconstructionDataset("dataset_root", split="test")
    test_id_to_idx = {ds_test.patch_ids[i]: i for i in range(len(ds_test))}

    selected_patches = [
        "scene_01_p0259",
        "scene_04_p0075",
        "scene_04_p0074",
        "scene_00_p0083",
        "scene_01_p0277",
        "scene_05_p0285",
        "scene_02_p0100",
        "scene_05_p0150"
    ]

    out_dir = r"C:\Users\HARSH AMBULE\.gemini\antigravity\brain\1020919c-a94d-481e-bd2f-f32559fc876f\outputs\rich_comparisons"
    os.makedirs(out_dir, exist_ok=True)

    for pid in selected_patches:
        if pid not in test_id_to_idx:
            continue
        idx = test_id_to_idx[pid]
        sample = ds_test[idx]

        opt_cloudy = sample["opt_cloudy"].unsqueeze(0).to(device)
        sar = sample["sar"].unsqueeze(0).to(device)
        temporal = sample["temporal"].unsqueeze(0).to(device)
        dem = sample["dem"].unsqueeze(0).to(device)
        opt_clean = sample["opt_clean"]
        mask = sample["mask"]

        with torch.no_grad():
            mean_fake, logvar_fake = gen(opt_cloudy, sar, temporal, dem)

        recon_01 = ((mean_fake[0].clamp(-1, 1) + 1) / 2).cpu().numpy()
        cloudy_01 = ((sample["opt_cloudy"] + 1) / 2).numpy()
        clean_01 = ((opt_clean + 1) / 2).numpy()
        mask_2d = mask[0].numpy() if mask.ndim == 3 else mask

        p_t = torch.from_numpy(recon_01).unsqueeze(0)
        c_t = torch.from_numpy(clean_01).unsqueeze(0)
        m_t = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0)
        psnr_val = masked_psnr(p_t, c_t, m_t)
        ssim_val = masked_ssim(p_t, c_t, m_t)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(enhance_display(cloudy_01))
        axes[0].set_title("1. Cloudy Input (LISS-IV)", fontsize=11, fontweight="bold")

        axes[1].imshow(mask_2d, cmap="Blues_r", vmin=0, vmax=1)
        axes[1].set_title(f"2. Cloud Mask ({float((mask_2d > 0.1).mean() * 100):.1f}% Covered)", fontsize=11, fontweight="bold")

        axes[2].imshow(enhance_display(recon_01))
        axes[2].set_title(f"3. CloudFree Output (PSNR: {psnr_val:.1f} dB, SSIM: {ssim_val:.3f})", fontsize=11, fontweight="bold", color="darkgreen")

        axes[3].imshow(enhance_display(clean_01))
        axes[3].set_title("4. Ground Truth Clear-Sky", fontsize=11, fontweight="bold")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        save_name = pid.replace(".npy", "") + "_showcase.png"
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved showcase: {pid} (PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.3f}) -> {save_path}")

    print("All showcases generated successfully!")


if __name__ == "__main__":
    main()
