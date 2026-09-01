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
    nir = img_chw01[2]
    red = img_chw01[1]
    green = img_chw01[0]
    cir = np.stack([nir, red, green], axis=-1)
    p1 = np.percentile(cir, 1.0)
    p99 = np.percentile(cir, 99.0)
    if p99 > p1 + 1e-4:
        return np.clip((cir - p1) / (p99 - p1), 0.0, 1.0)
    return np.clip(cir, 0.0, 1.0)

def enhance_dem(dem_4ch):
    elev = dem_4ch[0]
    slope = dem_4ch[1]
    p1_e, p99_e = np.percentile(elev, (1, 99))
    p1_s, p99_s = np.percentile(slope, (1, 99))
    norm_e = np.clip((elev - p1_e) / (p99_e - p1_e + 1e-4), 0.0, 1.0)
    norm_s = np.clip((slope - p1_s) / (p99_s - p1_s + 1e-4), 0.0, 1.0)
    return np.clip(0.6 * norm_e + 0.4 * norm_s, 0.0, 1.0)

def main():
    device = torch.device('cpu')
    ds = CloudReconstructionDataset(os.path.join(PROJECT_ROOT, 'dataset_root'), split='test')
    test_map = {ds.patch_ids[i]: i for i in range(len(ds))}

    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'generator_best.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
    gen.load_state_dict(ckpt['gen_ema_state'])
    gen.eval()

    selected_pids = [
        'scene_02_p0023',
        'scene_01_p0234',
        'scene_05_p0251',
        'scene_04_p0074'
    ]

    cols = [
        '(a) Cloudy Input\n(LISS-IV CIR)',
        '(b) Cloud Mask\n(Stage-A)',
        '(c) Temporal Prior\n(Sentinel-2 CIR)',
        '(d) CartoDEM\n(Topographic Relief)',
        '(e) RelieF-CR (Ours)\n(Reconstruction)',
        '(f) Ground Truth\n(Clear-Sky CIR)',
        '(g) Absolute Error\n(|Recon - GT|)',
        '(h) Uncertainty\n(Predicted Std $\\sigma$)'
    ]

    fig, axes = plt.subplots(len(selected_pids), len(cols), figsize=(2.4 * len(cols), 2.5 * len(selected_pids)), dpi=300)

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
            t_in = ((sample['temporal'] + 1) / 2).numpy()
            gt = ((clean + 1) / 2).numpy()
            rec = ((mean_fake[0] + 1) / 2).clamp(0, 1).numpy()
            mask_2d = mask[0].numpy() if mask.ndim == 3 else mask
            dem_data = dem[0].numpy()

            abs_err = np.mean(np.abs(rec - gt), axis=0)

            axes[r, 0].imshow(enhance_cir(c_in))
            axes[r, 1].imshow(mask_2d, cmap='Blues_r', vmin=0, vmax=1)
            axes[r, 2].imshow(enhance_cir(t_in))
            axes[r, 3].imshow(enhance_dem(dem_data), cmap='terrain')
            axes[r, 4].imshow(enhance_cir(rec))
            axes[r, 5].imshow(enhance_cir(gt))
            im_err = axes[r, 6].imshow(abs_err, cmap='inferno', vmin=0.0, vmax=0.15)
            im_unc = axes[r, 7].imshow(std_fake, cmap='magma', vmin=0.04, vmax=0.18)

            for c in range(len(cols)):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                for spine in axes[r, c].spines.values():
                    spine.set_color('#cccccc')
                    spine.set_linewidth(0.8)
                if r == 0:
                    axes[r, c].set_title(cols[c], fontsize=8.5, fontweight='bold', pad=6)

            cloud_pct = float((mask_2d > 0.1).mean()) * 100.0
            axes[r, 0].set_ylabel(f'{pid}\n({cloud_pct:.1f}% Cloud)', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    cbar_ax_err = fig.add_axes([0.765, 0.02, 0.10, 0.015])
    cb_err = fig.colorbar(im_err, cax=cbar_ax_err, orientation='horizontal')
    cb_err.set_ticks([0.0, 0.07, 0.15])
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
    print('SUCCESS: Flagship Figure 2 generated!')

if __name__ == '__main__':
    main()