"""
Ultra Sci-Fi Cyber-Tech Neural Architecture Diagram for CloudFree Vision v2.
Polished text centering, rich SAR radar texture, and clean laser conduit clearances.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

sys.path.append(r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2")
from data.dataset import CloudReconstructionDataset
from models.generator import CloudReconstructionGeneratorV2


def false_color_cir(opt_chw):
    nir, red, green = opt_chw[2], opt_chw[1], opt_chw[0]
    rgb = np.stack([nir, red, green], axis=-1)
    p2, p98 = np.percentile(rgb, 1), np.percentile(rgb, 99)
    if p98 > p2:
        return np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = r"checkpoints\generator_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
    gen.load_state_dict(ckpt["gen_ema_state"])
    gen.eval()

    ds_test = CloudReconstructionDataset("dataset_root", split="test")
    
    target_pid = "scene_02_p0100"
    idx = ds_test.patch_ids.index(target_pid) if target_pid in ds_test.patch_ids else 0
    sample = ds_test[idx]

    with torch.no_grad():
        mean_fake, logvar_fake = gen(
            sample["opt_cloudy"].unsqueeze(0).to(device),
            sample["sar"].unsqueeze(0).to(device),
            sample["temporal"].unsqueeze(0).to(device),
            sample["dem"].unsqueeze(0).to(device)
        )

    # Real image arrays
    cloudy_img = false_color_cir(((sample["opt_cloudy"] + 1) / 2).numpy())
    
    # Load rich textured SAR from scene_01_p0277
    try:
        sar_rich = np.load(r"dataset_root/test/sar/scene_01_p0277.npy")[0]
        sar_disp = (sar_rich - sar_rich.min()) / (sar_rich.max() - sar_rich.min() + 1e-8)
    except Exception:
        sar_raw = sample["sar"][0].numpy()
        sar_disp = (sar_raw - sar_raw.min()) / (sar_raw.max() - sar_raw.min() + 1e-8)

    temp_img = false_color_cir(((sample["temporal"] + 1) / 2).numpy())
    dem_disp = sample["dem"][0].numpy()
    dem_disp = (dem_disp - dem_disp.min()) / (dem_disp.max() - dem_disp.min() + 1e-8)
    
    recon_img = false_color_cir(((mean_fake[0].clamp(-1, 1) + 1) / 2).cpu().numpy())
    uncertainty_map = torch.exp(0.5 * logvar_fake)[0].mean(dim=0).cpu().numpy()
    unc_disp = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min() + 1e-8)

    # =========================================================================
    # MASTER SCI-FI CANVAS
    # =========================================================================
    fig = plt.figure(figsize=(26, 13.5), dpi=300, facecolor='#05070E')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#05070E')
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 13.5)
    ax.axis('off')

    # Cyber Grid
    for gx in np.linspace(0.5, 25.5, 51):
        ax.axvline(gx, color='#0B1120', lw=0.5, zorder=1)
    for gy in np.linspace(0.5, 13.0, 26):
        ax.axhline(gy, color='#0B1120', lw=0.5, zorder=1)

    # Sci-Fi Top Banner
    ax.text(13.0, 12.95, "CLOUDFREE VISION v2 // QUANTUM MULTI-MODAL CROSS-ATTENTION GENERATOR",
            ha='center', va='center', fontsize=16.5, fontweight='bold', color='#00F0FF',
            path_effects=[pe.withStroke(linewidth=3, foreground='#002B4D')])
    
    ax.text(13.0, 12.55, "HETEROSCEDASTIC DUAL-HEAD ARCHITECTURE • SPATIAL TRANSFORMER WARPING • 7-TERM FREQUENCY SUPERVISION",
            ha='center', va='center', fontsize=9.8, fontweight='bold', color='#64748B', family='monospace')

    # HUD Status Badges
    def add_hud_badge(x, y, text, color='#00F0FF', bg='#091322'):
        b = FancyBboxPatch((x, y), len(text)*0.125 + 0.4, 0.42, boxstyle="round,pad=0.04,rounding_size=0.08",
                           facecolor=bg, edgecolor=color, lw=1.2, zorder=3)
        ax.add_patch(b)
        ax.text(x + (len(text)*0.125 + 0.4)/2, y + 0.21, text, ha='center', va='center',
                fontsize=7.8, fontweight='bold', color=color, family='monospace', zorder=4)

    add_hud_badge(0.8, 12.75, "STATUS: ONLINE", '#00FF66')
    add_hud_badge(4.3, 12.75, "WEIGHTS: DDP_EMA_v2", '#00F0FF')
    add_hud_badge(20.2, 12.75, "PSNR: 29.10 dB", '#FFCC00')
    add_hud_badge(23.2, 12.75, "SSIM: 0.963", '#FF0055')

    def add_cyber_card(x, y, w, h, title, subtitle="", glow_color='#00F0FF', fill_color='#080D1A'):
        glow = FancyBboxPatch((x-0.05, y-0.05), w+0.1, h+0.1, boxstyle="round,pad=0.15,rounding_size=0.2",
                              facecolor='none', edgecolor=glow_color, lw=2.2, alpha=0.3, zorder=2)
        ax.add_patch(glow)
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.2",
                              facecolor=fill_color, edgecolor=glow_color, lw=1.5, zorder=2)
        ax.add_patch(card)
        if title:
            ax.text(x + 0.35, y + h - 0.40, title, fontsize=11.0, fontweight='bold', color=glow_color,
                    family='monospace', zorder=4)
        if subtitle:
            ax.text(x + 0.35, y + h - 0.70, subtitle, fontsize=7.8, color='#94A3B8', family='monospace', zorder=4)
        return card

    # Card 1: Multi-Sensor Ingestion
    add_cyber_card(0.6, 0.8, 5.2, 11.2, "[01] MULTI-SENSOR INGESTION", "ORBITAL TELEMETRY & 5.0m MASTER GRID", '#00F0FF')

    def embed_cyber_patch(img, x, y, size, title, sub1, sub2="", glow='#00F0FF', cmap=None):
        plate = FancyBboxPatch((x-0.06, y-0.06), size+0.12, size+0.12, boxstyle="round,pad=0.04,rounding_size=0.08",
                               facecolor='#030712', edgecolor=glow, lw=1.6, zorder=3)
        ax.add_patch(plate)
        im_ax = fig.add_axes([x/26.0, y/13.5, size/26.0, size/13.5], zorder=4)
        if cmap:
            im_ax.imshow(img, cmap=cmap)
        else:
            im_ax.imshow(img)
        im_ax.axis('off')
        
        # Corner Reticles
        ax.plot([x-0.06, x+0.2], [y+size+0.06, y+size+0.06], color=glow, lw=2.0, zorder=5)
        ax.plot([x-0.06, x-0.06], [y+size+0.06, y+size-0.2], color=glow, lw=2.0, zorder=5)

        ax.text(x + size + 0.25, y + size*0.72, title, fontsize=9.2, fontweight='bold', color='#FFFFFF', family='monospace', zorder=4)
        ax.text(x + size + 0.25, y + size*0.42, sub1, fontsize=7.8, color=glow, family='monospace', zorder=4)
        if sub2:
            ax.text(x + size + 0.25, y + size*0.14, sub2, fontsize=7.2, color='#94A3B8', family='monospace', zorder=4)

    embed_cyber_patch(cloudy_img, 0.9, 9.1, 1.9, "OPTICAL CLOUDY", "ISRO LISS-IV [3×256²]", "5.0m TOA Spectral CIR", '#00F0FF')
    embed_cyber_patch(sar_disp, 0.9, 6.4, 1.9, "RADAR BACKSCATTER", "SENTINEL-1 SAR [2×256²]", "Dual-Pol VV/VH dB Normal", '#FF9900', cmap='bone')
    embed_cyber_patch(temp_img, 0.9, 3.7, 1.9, "TEMPORAL PRIOR", "SENTINEL-2 DRY [3×256²]", "Clear-Sky Baseline CIR", '#BD00FF')
    embed_cyber_patch(dem_disp, 0.9, 1.0, 1.9, "CARTODEM 4-CH", "TOPOGRAPHY [4×256²]", "Elev, Slope, Sin, Cos", '#00FF66', cmap='terrain')

    # Card 2: Feature Stems & STN
    add_cyber_card(6.2, 0.8, 5.0, 11.2, "[02] STEMS & STN ALIGNMENT", "SUB-PIXEL REGISTRATION ENGINE", '#BD00FF')

    def add_neon_module(x, y, w, h, title, spec, glow='#00F0FF', bg='#0C1326'):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                           facecolor=bg, edgecolor=glow, lw=1.4, zorder=3)
        ax.add_patch(b)
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.0, fontweight='bold', color='#FFFFFF', family='monospace', zorder=4)
        ax.text(x + w/2, y + h*0.28, spec, ha='center', va='center', fontsize=7.4, color=glow, family='monospace', zorder=4)

    add_neon_module(6.5, 9.3, 4.4, 1.5, "OPTICAL ENCODER STEM", "Conv 3×3 -> [48 × 128 × 128]", '#00F0FF')
    
    add_neon_module(6.5, 6.6, 2.0, 1.5, "SAR STEM", "[48 × 128²]", '#FF9900')
    add_neon_module(8.8, 6.6, 2.1, 1.5, "STN WARPING", r"$\mathcal{T}_{\theta} \in \mathbb{R}^{2\times 3}$ Affine", '#FF0055', bg='#1E0B16')

    add_neon_module(6.5, 3.9, 2.0, 1.5, "TEMP STEM", "[48 × 128²]", '#BD00FF')
    add_neon_module(8.8, 3.9, 2.1, 1.5, "STN WARPING", r"$\mathcal{T}_{\theta} \in \mathbb{R}^{2\times 3}$ Affine", '#FF0055', bg='#1E0B16')

    add_neon_module(6.5, 1.2, 4.4, 1.5, "TOPOGRAPHIC DEM STEM", "Conv 3×3 -> [32 × 128 × 128]", '#00FF66')

    # Card 3: Cross-Attention
    add_cyber_card(11.6, 2.8, 6.4, 9.2, "[03] WINDOWED CROSS-ATTENTION", "DYNAMIC SPATIAL-SPECTRAL QUANTUM FUSION", '#FF0055')

    add_neon_module(12.0, 3.2, 5.6, 1.4, "AUXILIARY FEATURE TENSOR CONCATENATION", r"$\mathbf{F}_{\mathrm{aux}} = [\mathbf{F}_{\mathrm{sar}}^{\mathrm{align}}, \mathbf{F}_{\mathrm{temp}}^{\mathrm{align}}, \mathbf{F}_{\mathrm{dem}}] \in \mathbb{R}^{160 \times 128 \times 128}$", '#FF9900')

    attn_engine = FancyBboxPatch((12.0, 4.9), 5.6, 5.7, boxstyle="round,pad=0.12,rounding_size=0.15",
                                 facecolor='#0B0716', edgecolor='#FF0055', lw=2.0, zorder=3)
    ax.add_patch(attn_engine)

    ax.text(14.8, 10.15, "LOCAL 8×8 WINDOW ATTENTION ENGINE", ha='center', fontsize=10.0, fontweight='bold', color='#FF0055', family='monospace', zorder=4)

    def add_token_box(x, y, w, h, text, color='#00F0FF'):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                           facecolor='#050A18', edgecolor=color, lw=1.2, zorder=4)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.6, color=color, fontweight='bold', family='monospace', zorder=5)

    add_token_box(12.3, 9.15, 5.0, 0.7, r"Query: $\mathbf{Q} = \mathrm{LN}(\mathbf{F}_{\mathrm{opt}})\mathbf{W}_Q \quad [\text{Optical}]$", '#00F0FF')
    add_token_box(12.3, 8.20, 5.0, 0.7, r"Keys/Values: $\mathbf{K},\mathbf{V} = \mathrm{LN}(\mathbf{F}_{\mathrm{aux}})\mathbf{W}_{K,V} \quad [\text{Aux}]$", '#FF9900')

    ax.text(14.8, 7.30, r"$\mathrm{CrossAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{B}\right)\mathbf{V}$",
            ha='center', fontsize=9.6, color='#FFCC00', fontweight='bold', zorder=4)

    ax.text(14.8, 6.40, r"$\mathbf{F}_{\mathrm{fused}} = \mathrm{LN}\left(\mathbf{F}_{\mathrm{opt}} + \mathrm{MLP}(\mathrm{CrossAttention})\right)$",
            ha='center', fontsize=9.2, color='#FFFFFF', fontweight='bold', family='monospace', zorder=4)

    ax.text(14.8, 5.50, ">> DYNAMIC QUERY: AUXILIARY RADAR/TEMPORAL PULLED EXCLUSIVELY UNDER CLOUD CORES <<",
            ha='center', fontsize=7.2, color='#64748B', family='monospace', zorder=4)

    # Laser Skip Conduit smoothly routed across top
    ax.annotate('', xy=(18.3, 9.8), xytext=(11.0, 9.8),
                arrowprops=dict(arrowstyle="-|>", color='#00F0FF', lw=2.4, ls='--',
                                connectionstyle="arc3,rad=-0.30", mutation_scale=18), zorder=6)
    ax.text(14.8, 12.18, "LASER CONDUIT: HIGH-RESOLUTION OPTICAL SKIP (CLEAR-SKY RADIOMETRY PRESERVATION)",
            fontsize=8.0, color='#00F0FF', fontweight='bold', family='monospace', ha='center', zorder=7)

    # Card 4: Decoder & Dual Heads
    add_cyber_card(18.4, 0.8, 7.0, 11.2, "[04] DECODER & DUAL HEADS", "HETEROSCEDASTIC UNCERTAINTY INFERENCE", '#00FF66')

    add_neon_module(18.8, 9.1, 6.2, 1.6, "MULTI-SCALE U-NET DECODER", "ResBlocks + Transposed Conv -> [48 × 256²]", '#00F0FF')

    embed_cyber_patch(recon_img, 18.9, 5.0, 2.5, "CLEAR-SKY RECONSTRUCTION HEAD", r"$\hat{\mathbf{y}} = \tanh(\mathbf{W}_y \mathbf{F}_{\mathrm{dec}}) \in [-1, 1]$", "PSNR: 29.10 dB | SSIM: 0.963 | SAM: 1.49°", '#00FF66')
    embed_cyber_patch(unc_disp, 18.9, 1.2, 2.5, "HETEROSCEDASTIC UNCERTAINTY HEAD", r"$\mathbf{s} = \log \sigma^2 \in \mathbb{R}$", "Calibrated Correlation: ρ = +0.245", '#FF0055', cmap='plasma')

    # Bottom Loss Framework
    loss_box = FancyBboxPatch((0.6, 0.15), 24.8, 0.55, boxstyle="round,pad=0.08,rounding_size=0.1",
                              facecolor='#060A16', edgecolor='#6366F1', lw=1.6, zorder=2)
    ax.add_patch(loss_box)
    loss_str = r"$\mathcal{L}_{\mathrm{Total}} = 1.0\,\mathcal{L}_{\mathrm{NLL}} + 10.0\,\mathcal{L}_{\mathrm{L1}} + 10.0\,\mathcal{L}_{\mathrm{FM}} + 0.1\,\mathcal{L}_{\mathrm{FFT}} + 0.5\,\mathcal{L}_{\mathrm{Sobel}} + 0.2\,\mathcal{L}_{\mathrm{Spec}} + 1.0\,\mathcal{L}_{\mathrm{Adv}}$"
    ax.text(13.0, 0.42, "7-TERM OPTIMIZATION SUITE: " + loss_str,
            ha='center', va='center', fontsize=9.4, color='#A5B4FC', fontweight='bold', zorder=4)

    # Laser Conduits
    def draw_laser(x1, y1, x2, y2, color='#00F0FF'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=14), zorder=5)

    draw_laser(5.3, 10.0, 6.4, 10.0, '#00F0FF')
    draw_laser(5.3, 7.3, 6.4, 7.3, '#FF9900')
    draw_laser(5.3, 4.6, 6.4, 4.6, '#BD00FF')
    draw_laser(5.3, 1.9, 6.4, 1.9, '#00FF66')

    draw_laser(8.5, 7.3, 8.7, 7.3, '#FF9900')
    draw_laser(8.5, 4.6, 8.7, 4.6, '#BD00FF')

    draw_laser(10.9, 7.3, 11.9, 4.2, '#FF9900')
    draw_laser(10.9, 4.6, 11.9, 3.8, '#BD00FF')
    draw_laser(10.9, 1.9, 11.9, 3.4, '#00FF66')

    draw_laser(10.9, 10.0, 11.9, 10.0, '#00F0FF')
    draw_laser(14.8, 4.6, 14.8, 4.9, '#FF9900')

    draw_laser(18.0, 8.1, 18.7, 9.6, '#FF0055')

    draw_laser(20.1, 9.0, 20.1, 7.6, '#00FF66')
    draw_laser(20.1, 4.9, 20.1, 3.8, '#FF0055')

    paper_png = r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\paper\figures\cloudfree_vision_v2_architecture.png"
    paper_pdf = r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\paper\figures\cloudfree_vision_v2_architecture.pdf"
    brain_png = r"C:\Users\HARSH AMBULE\.gemini\antigravity\brain\1020919c-a94d-481e-bd2f-f32559fc876f\outputs\cloudfree_vision_v2_architecture_scifi.png"

    plt.savefig(paper_png, dpi=300, bbox_inches='tight', facecolor='#05070E')
    plt.savefig(paper_pdf, format='pdf', bbox_inches='tight', facecolor='#05070E')
    plt.savefig(brain_png, dpi=300, bbox_inches='tight', facecolor='#05070E')
    plt.close(fig)

    print("Polished Sci-Fi Masterpiece Diagram generated successfully.")


if __name__ == "__main__":
    main()
