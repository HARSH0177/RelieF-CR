import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'data'))

from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset

def stretch_rgb(img_3ch):
    cir = img_3ch[[2, 1, 0]].copy()
    stretched = np.zeros_like(cir)
    for c in range(3):
        band = cir[c]
        p2, p98 = np.percentile(band, (2, 98))
        if p98 > p2 + 1e-4:
            stretched[c] = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            stretched[c] = np.clip(band, 0, 1)
    return stretched.transpose(1, 2, 0)

def stretch_sar(sar_2d):
    p2, p98 = np.percentile(sar_2d, (2, 98))
    if p98 > p2 + 1e-4:
        return np.clip((sar_2d - p2) / (p98 - p2), 0, 1)
    return np.clip(sar_2d, 0, 1)

def stretch_dem(dem_4ch):
    elev = dem_4ch[0]
    slope = dem_4ch[1]
    p2_e, p98_e = np.percentile(elev, (2, 98))
    p2_s, p98_s = np.percentile(slope, (2, 98))
    norm_e = np.clip((elev - p2_e) / (p98_e - p2_e + 1e-4), 0, 1)
    norm_s = np.clip((slope - p2_s) / (p98_s - p2_s + 1e-4), 0, 1)
    relief = 0.6 * norm_e + 0.4 * norm_s
    return np.clip(relief, 0, 1)

def main():
    device = torch.device('cpu')
    data_root = os.path.join(PROJECT_ROOT, 'dataset_root')
    ds = CloudReconstructionDataset(data_root, split='test')
    
    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'generator_best.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
    gen.load_state_dict(ckpt['gen_ema_state'])
    gen.eval()
    
    candidate_patches = [
        ('scene_01_p0062', 285),
        ('scene_01_p0224', 447),
        ('scene_02_p0023', 563),
        ('scene_05_p0251', 1516)
    ]
    
    cols = [
        '(a) Cloudy Input\n(LISS-IV CIR)',
        '(b) Cloud Mask\n(Stage-A)',
        '(c) Sentinel-1 SAR\n(VV Backscatter)',
        '(d) CartoDEM\n(Topography Relief)',
        '(e) RelieF-CR (Ours)\n(Reconstructed)',
        '(f) Ground Truth\n(Clear Optical)',
        '(g) Absolute Error\n(|Recon - GT|)',
        '(h) Uncertainty\n(Predicted Std $\\sigma$)'
    ]
    
    num_rows = len(candidate_patches)
    num_cols = len(cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.4 * num_cols, 2.5 * num_rows), dpi=300)
    
    with torch.no_grad():
        for r, (pname, idx) in enumerate(candidate_patches):
            batch = ds[idx]
            opt_cloudy = batch['opt_cloudy'].unsqueeze(0).to(device)
            opt_clean = batch['opt_clean'].unsqueeze(0).to(device)
            sar = batch['sar'].unsqueeze(0).to(device)
            temporal = batch['temporal'].unsqueeze(0).to(device)
            dem = batch['dem'].unsqueeze(0).to(device)
            mask = batch['mask'].squeeze().numpy()
            
            mean_fake, logvar_fake = gen(opt_cloudy, sar, temporal, dem)
            std_fake = torch.exp(0.5 * logvar_fake)[0].mean(dim=0).numpy()
            
            c_in = ((opt_cloudy[0] + 1) / 2).numpy()
            gt = ((opt_clean[0] + 1) / 2).numpy()
            rec = ((mean_fake[0] + 1) / 2).clamp(0, 1).numpy()
            sar_vv = sar[0, 0].numpy()
            dem_data = dem[0].numpy()
            
            abs_err = np.mean(np.abs(rec - gt), axis=0)
            
            axes[r, 0].imshow(stretch_rgb(c_in))
            axes[r, 1].imshow(mask, cmap='Blues_r', vmin=0, vmax=1)
            axes[r, 2].imshow(stretch_sar(sar_vv), cmap='gray')
            axes[r, 3].imshow(stretch_dem(dem_data), cmap='terrain')
            axes[r, 4].imshow(stretch_rgb(rec))
            axes[r, 5].imshow(stretch_rgb(gt))
            im_err = axes[r, 6].imshow(abs_err, cmap='inferno', vmin=0.0, vmax=0.20)
            im_unc = axes[r, 7].imshow(std_fake, cmap='magma', vmin=0.04, vmax=0.18)
            
            for c in range(num_cols):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                for spine in axes[r, c].spines.values():
                    spine.set_color('#dddddd')
                    spine.set_linewidth(0.8)
                if r == 0:
                    axes[r, c].set_title(cols[c], fontsize=9, fontweight='bold', pad=6)
            
            cloud_pct = float(mask.sum()) / float(mask.size) * 100.0
            axes[r, 0].set_ylabel(f'Sample {r+1}\n({cloud_pct:.1f}% Cloud)', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    
    cbar_ax_err = fig.add_axes([0.765, 0.02, 0.10, 0.015])
    cb_err = fig.colorbar(im_err, cax=cbar_ax_err, orientation='horizontal')
    cb_err.set_ticks([0.0, 0.10, 0.20])
    cb_err.ax.tick_params(labelsize=6)
    cb_err.set_label('Abs Error $|\\hat{y} - y|$', fontsize=7, fontweight='bold')
    
    cbar_ax_unc = fig.add_axes([0.885, 0.02, 0.10, 0.015])
    cb_unc = fig.colorbar(im_unc, cax=cbar_ax_unc, orientation='horizontal')
    cb_unc.set_ticks([0.04, 0.11, 0.18])
    cb_unc.ax.tick_params(labelsize=6)
    cb_unc.set_label('Pred Std $\\sigma$', fontsize=7, fontweight='bold')
    
    out_dir = os.path.join(PROJECT_ROOT, 'paper', 'figures')
    out_png = os.path.join(out_dir, 'comparison_grid.png')
    out_pdf = os.path.join(out_dir, 'comparison_grid.pdf')
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    print('SUCCESS: Generated publication-grade comparison_grid.png and .pdf')

if __name__ == '__main__':
    main()