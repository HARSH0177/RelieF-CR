import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'data'))

from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset

def enhance_cir(img_chw01):
    # LISS-IV: ch0=Green, ch1=Red, ch2=NIR
    # CIR: R=NIR, G=Red, B=Green
    nir = img_chw01[2]
    red = img_chw01[1]
    green = img_chw01[0]
    cir = np.stack([nir, red, green], axis=-1)
    p2 = np.percentile(cir, 2.0)
    p98 = np.percentile(cir, 98.0)
    if p98 > p2 + 1e-4:
        return np.clip((cir - p2) / (p98 - p2), 0.0, 1.0)
    return np.clip(cir, 0.0, 1.0)

def stretch_sar(sar_2d):
    s_min = float(sar_2d.min())
    s_max = float(sar_2d.max())
    if s_max > s_min + 1e-6:
        return (sar_2d - s_min) / (s_max - s_min)
    return np.zeros_like(sar_2d) + 0.5

def enhance_dem(dem_4ch):
    elev = dem_4ch[0]
    slope = dem_4ch[1]
    p2_e, p98_e = np.percentile(elev, (2, 98))
    p2_s, p98_s = np.percentile(slope, (2, 98))
    norm_e = np.clip((elev - p2_e) / (p98_e - p2_e + 1e-4), 0.0, 1.0)
    norm_s = np.clip((slope - p2_s) / (p98_s - p2_s + 1e-4), 0.0, 1.0)
    return np.clip(0.55 * norm_e + 0.45 * norm_s, 0.0, 1.0)

def main():
    device = torch.device('cpu')
    ds = CloudReconstructionDataset(os.path.join(PROJECT_ROOT, 'dataset_root'), split='test')
    test_map = {ds.patch_ids[i]: i for i in range(len(ds))}

    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'generator_best.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
    gen.load_state_dict(ckpt['gen_ema_state'])
    gen.eval()

    # 4 verified 100% cloud-free Ground Truth test patches with rich landmarks & sharp boundaries
    selected_pids = [
        'scene_04_p0305',  # 42.9% cloud, 33.4 dB, clear parcel landmarks & river bend
        'scene_01_p0115',  # 31.7% cloud, 31.6 dB, agricultural vegetation parcel grids
        'scene_02_p0037',  # 29.8% cloud, 36.4 dB, mountain topography relief & valleys
        'scene_04_p0309'   # 25.6% cloud, 29.0 dB, river/drainage boundaries & farmland
    ]

    cols = [
        '(a) Cloudy Input\n(LISS-IV CIR)',
        '(b) Cloud Mask\n(Stage-A)',
        '(c) Sentinel-1 SAR\n(VV Backscatter)',
        '(d) CartoDEM\n(Topographic Relief)',
        '(e) RelieF-CR (Ours)\n(Reconstruction)',
        '(f) Ground Truth\n(Clear-Sky CIR)',
        '(g) Absolute Error\n(|Recon - GT|)',
        '(h) Uncertainty\n(Predicted Std $\\sigma$)'
    ]

    num_rows = len(selected_pids)
    num_cols = len(cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.4 * num_cols, 2.5 * num_rows), dpi=300)

    with torch.no_grad():
        for r, pid in enumerate(selected_pids):
            idx = test_map[pid]
            sample = ds[idx]

            opt_cloudy = sample['opt_cloudy'].unsqueeze(0).to(device)
            sar = sample['sar'].unsqueeze(0).to(device)
            temporal = sample['temporal'].unsqueeze(0).to(device)
            dem = sample['dem'].unsqueeze(0).to(device)
            clean = sample['opt_clean']
            mask = sample['mask']

            mean_fake, logvar_fake = gen(opt_cloudy, sar, temporal, dem)
            std_fake = torch.exp(0.5 * logvar_fake)[0].mean(dim=0).numpy()

            c_in = ((sample['opt_cloudy'] + 1) / 2).numpy()
            gt = ((clean + 1) / 2).numpy()
            rec = ((mean_fake[0] + 1) / 2).clamp(0, 1).numpy()
            sar_vv = sar[0, 0].numpy()
            dem_data = dem[0].numpy()
            mask_2d = mask[0].numpy() if mask.ndim == 3 else mask

            abs_err = np.mean(np.abs(rec - gt), axis=0)

            axes[r, 0].imshow(enhance_cir(c_in))
            axes[r, 1].imshow(mask_2d, cmap='Blues_r', vmin=0, vmax=1)
            axes[r, 2].imshow(stretch_sar(sar_vv), cmap='gray')
            axes[r, 3].imshow(enhance_dem(dem_data), cmap='terrain')
            axes[r, 4].imshow(enhance_cir(rec))
            axes[r, 5].imshow(enhance_cir(gt))
            im_err = axes[r, 6].imshow(abs_err, cmap='inferno', vmin=0.0, vmax=0.12)
            im_unc = axes[r, 7].imshow(std_fake, cmap='magma', vmin=0.04, vmax=0.18)

            for c in range(num_cols):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                for spine in axes[r, c].spines.values():
                    spine.set_color('#cccccc')
                    spine.set_linewidth(0.8)
                if r == 0:
                    axes[r, c].set_title(cols[c], fontsize=8.5, fontweight='bold', pad=6)

            cloud_pct = float((mask_2d > 0.1).mean()) * 100.0
            axes[r, 0].set_ylabel(f'Sample {r+1}\n({cloud_pct:.1f}% Cloud)', fontsize=8.0, fontweight='bold')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    cbar_ax_err = fig.add_axes([0.765, 0.02, 0.10, 0.015])
    cb_err = fig.colorbar(im_err, cax=cbar_ax_err, orientation='horizontal')
    cb_err.set_ticks([0.0, 0.06, 0.12])
    cb_err.ax.tick_params(labelsize=6)
    cb_err.set_label('Abs Error $|\\hat{y} - y|$', fontsize=7, fontweight='bold')

    cbar_ax_unc = fig.add_axes([0.885, 0.02, 0.10, 0.015])
    cb_unc = fig.colorbar(im_unc, cax=cbar_ax_unc, orientation='horizontal')
    cb_unc.set_ticks([0.04, 0.11, 0.18])
    cb_unc.ax.tick_params(labelsize=6)
    cb_unc.set_label('Pred Std $\\sigma$', fontsize=7, fontweight='bold')

    out_png = os.path.join(PROJECT_ROOT, 'paper', 'figures', 'comparison_grid.png')
    out_pdf = os.path.join(PROJECT_ROOT, 'paper', 'figures', 'comparison_grid.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    print('SUCCESS: Master Figure 2 generated!')

if __name__ == '__main__':
    main()