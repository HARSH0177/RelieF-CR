import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset

device = torch.device('cpu')
ds = CloudReconstructionDataset(os.path.join(PROJECT_ROOT, 'dataset_root'), split='test')
test_map = {ds.patch_ids[i]: i for i in range(len(ds))}

# Load trained checkpoint
ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'generator_best.pt')
ckpt = torch.load(ckpt_path, map_location=device)
gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
state_key = 'gen_ema_state' if 'gen_ema_state' in ckpt else 'gen_state'
gen.load_state_dict(ckpt[state_key])
gen.eval()

# 4 Verified, diverse test patches across ecotopes and cloud coverage
selected_pids = [
    'scene_04_p0305',  # Agricultural patchwork
    'scene_01_p0115',  # Vegetation canopy & parcel grid
    'scene_02_p0037',  # Mountain ridge & topographic drainage
    'scene_04_p0309'   # Riverine corridor & mixed land cover
]

def false_color(x_chw01):
    # CIR Composite: R=NIR (ch2), G=Red (ch1), B=Green (ch0)
    cir = np.stack([x_chw01[2], x_chw01[1], x_chw01[0]], axis=-1)
    p2, p98 = np.percentile(cir, (2, 98), axis=(0, 1))
    return np.clip((cir - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

cols = ["Cloudy Input", "Cloud/Shadow Mask", "SAR (VV)", "Reconstruction", "Ground Truth"]
fig, axes = plt.subplots(len(selected_pids), 5, figsize=(3.1 * 5, 3.1 * len(selected_pids)), dpi=300)

with torch.no_grad():
    for r, pid in enumerate(selected_pids):
        idx = test_map[pid]
        sample = ds[idx]

        opt_cloudy = sample['opt_cloudy'].unsqueeze(0).to(device)
        sar = sample['sar'].unsqueeze(0).to(device)
        temporal = sample['temporal'].unsqueeze(0).to(device)
        dem = sample['dem'].unsqueeze(0).to(device)

        mean_fake, _ = gen(opt_cloudy, sar, temporal, dem)

        c_in01 = ((sample['opt_cloudy'] + 1.0) / 2.0).numpy()
        clean01 = ((sample['opt_clean'] + 1.0) / 2.0).numpy()
        recon01 = ((mean_fake[0] + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
        sar_vv = sar[0, 0].numpy()
        mask_class = sample['mask_class'].squeeze().numpy()

        # Render Columns (a) - (e)
        axes[r, 0].imshow(false_color(c_in01))
        im_mask = axes[r, 1].imshow(mask_class, cmap='viridis', vmin=0, vmax=2)
        
        p2_s, p98_s = np.percentile(sar_vv, (2, 98))
        sar_disp = np.clip((sar_vv - p2_s) / (p98_s - p2_s + 1e-6), 0.0, 1.0)
        axes[r, 2].imshow(sar_disp, cmap='gray')
        
        axes[r, 3].imshow(false_color(recon01))
        axes[r, 4].imshow(false_color(clean01))

        for c in range(5):
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=11, fontweight='bold')

        # Add discrete mask colorbar to bottom row
        if r == len(selected_pids) - 1:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            div_m = make_axes_locatable(axes[r, 1])
            cax_m = div_m.append_axes("bottom", size="5%", pad=0.06)
            cb_m = fig.colorbar(im_mask, cax=cax_m, orientation="horizontal", ticks=[0, 1, 2])
            cb_m.ax.set_xticklabels(["Clear", "Cloud", "Shadow"], fontsize=7)

plt.tight_layout()
out_png = os.path.join(PROJECT_ROOT, "paper", "figures", "comparison_grid_trained.png")
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Generated clean 5-column Figure 2 -> {out_png}")
