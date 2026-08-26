"""
SOTA Flagship Architecture Diagram Generator for IEEE TGRS.
Polished with high-contrast SAR stretching, crisp math typography,
and balanced layout geometry.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
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
    
    # High-contrast normalization for SAR
    sar_raw = sample["sar"][0].numpy()
    sar_disp = (sar_raw - sar_raw.min()) / (sar_raw.max() - sar_raw.min() + 1e-8)

    temp_img = false_color_cir(((sample["temporal"] + 1) / 2).numpy())
    dem_disp = sample["dem"][0].numpy()
    dem_disp = (dem_disp - dem_disp.min()) / (dem_disp.max() - dem_disp.min() + 1e-8)
    
    recon_img = false_color_cir(((mean_fake[0].clamp(-1, 1) + 1) / 2).cpu().numpy())
    uncertainty_map = torch.exp(0.5 * logvar_fake)[0].mean(dim=0).cpu().numpy()
    unc_disp = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min() + 1e-8)

    # =========================================================================
    # MASTER CANVAS
    # =========================================================================
    fig = plt.figure(figsize=(24, 12), dpi=300, facecolor='#FFFFFF')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Header
    ax.text(12.0, 11.55, "CloudFree Vision v2: Multi-Modal SAR-Optical-Topographic Cross-Attention Architecture",
            ha='center', va='center', fontsize=17, fontweight='bold', color='#0F172A')
    ax.text(12.0, 11.18, "Dual-Head Heteroscedastic Generator with Learned Spatial Transformer Alignment & 7-Term Spectral-Structural Loss",
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#475569')

    def add_card(x, y, w, h, title, bg='#F8FAFC', border='#CBD5E1', title_color='#1E293B'):
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.2",
                              facecolor=bg, edgecolor=border, lw=1.5)
        ax.add_patch(card)
        if title:
            ax.text(x + 0.3, y + h - 0.35, title, fontsize=10.5, fontweight='bold', color=title_color)
        return card

    # Card 1: Multi-Modal Input Modalities
    add_card(0.5, 0.8, 5.0, 9.9, "1. Multi-Sensor Data Ingestion (5.0m Master Grid)", bg='#F8FAFC', border='#94A3B8')

    def embed_patch(img, x, y, size, title, subtitle="", cmap=None):
        rect = FancyBboxPatch((x-0.08, y-0.08), size+0.16, size+0.16, boxstyle="round,pad=0.04,rounding_size=0.08",
                              facecolor='#FFFFFF', edgecolor='#64748B', lw=1.2)
        ax.add_patch(rect)
        im_ax = fig.add_axes([x/24.0, y/12.0, size/24.0, size/12.0])
        if cmap:
            im_ax.imshow(img, cmap=cmap)
        else:
            im_ax.imshow(img)
        im_ax.axis('off')
        ax.text(x + size + 0.25, y + size*0.65, title, fontsize=9.2, fontweight='bold', color='#0F172A')
        ax.text(x + size + 0.25, y + size*0.25, subtitle, fontsize=8.0, color='#475569', family='monospace')

    embed_patch(cloudy_img, 0.8, 8.0, 1.8, "Cloudy Optical (LISS-IV)", "3 Bands [G, R, NIR]\n5.0m TOA Reflectance")
    embed_patch(sar_disp, 0.8, 5.6, 1.8, "Sentinel-1 C-Band SAR", "2 Channels [VV, VH]\nCalibrated Sigma0 (dB)", cmap='bone')
    embed_patch(temp_img, 0.8, 3.2, 1.8, "Dry-Season Sentinel-2", "3 Bands [B03, B04, B08]\nClear-Sky Prior")
    embed_patch(dem_disp, 0.8, 1.0, 1.8, "CartoDEM Topography", "4 Channels Horn Model\n[Elev, Slope, Sin, Cos]", cmap='terrain')

    # Card 2: Deep Feature Stems & Spatial Transformer
    add_card(5.9, 0.8, 4.9, 9.9, "2. Deep Feature Stems & STN Alignment", bg='#F1F5F9', border='#94A3B8')

    def add_module_box(x, y, w, h, title, subtitle, bg='#FFFFFF', border='#3B82F6', text_color='#1E40AF'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                             facecolor=bg, edgecolor=border, lw=1.4)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.60, title, ha='center', va='center', fontsize=9.0, fontweight='bold', color=text_color)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=7.5, color='#475569', family='monospace')

    add_module_box(6.2, 8.1, 4.3, 1.4, "Optical Feature Stem", r"Conv $3\times 3$ $\to \mathbf{F}_{\mathrm{opt}} \in \mathbb{R}^{48 \times 128 \times 128}$", border='#2563EB', text_color='#1D4ED8')
    add_module_box(6.2, 5.7, 2.0, 1.4, "SAR Stem", r"$\mathbb{R}^{48 \times 128^2}$", border='#D97706', text_color='#B45309')
    add_module_box(8.5, 5.7, 2.0, 1.4, "STN Alignment", r"Affine Warp $\mathcal{T}_{\theta}$", bg='#FEE2E2', border='#EF4444', text_color='#991B1B')

    add_module_box(6.2, 3.3, 2.0, 1.4, "Temporal Stem", r"$\mathbb{R}^{48 \times 128^2}$", border='#7C3AED', text_color='#6D28D9')
    add_module_box(8.5, 3.3, 2.0, 1.4, "STN Alignment", r"Affine Warp $\mathcal{T}_{\theta}$", bg='#FEE2E2', border='#EF4444', text_color='#991B1B')

    add_module_box(6.2, 1.0, 4.3, 1.4, "Topographic DEM Stem", r"Conv $3\times 3$ $\to \mathbf{F}_{\mathrm{dem}} \in \mathbb{R}^{32 \times 128 \times 128}$", border='#059669', text_color='#047857')

    # Card 3: Windowed Cross-Attention Fusion
    add_card(11.2, 2.6, 5.8, 7.5, "3. Windowed Multi-Head Cross-Attention", bg='#FDF2F8', border='#F472B6')

    add_module_box(11.5, 3.0, 5.2, 1.2, "Auxiliary Feature Concatenation", r"$\mathbf{F}_{\mathrm{aux}} = [\mathbf{F}_{\mathrm{sar}}^{\mathrm{align}}, \mathbf{F}_{\mathrm{temp}}^{\mathrm{align}}, \mathbf{F}_{\mathrm{dem}}] \in \mathbb{R}^{160 \times 128 \times 128}$", bg='#F1F5F9', border='#64748B', text_color='#334155')

    attn_math = FancyBboxPatch((11.5, 4.5), 5.2, 4.8, boxstyle="round,pad=0.12,rounding_size=0.15",
                               facecolor='#FFFFFF', edgecolor='#EC4899', lw=1.8)
    ax.add_patch(attn_math)
    ax.text(14.1, 8.95, "Local 8×8 Non-Overlapping Windows", ha='center', fontsize=9.8, fontweight='bold', color='#9D174D')
    
    ax.text(14.1, 8.25, r"$\mathbf{Q} = \mathrm{LN}(\mathbf{F}_{\mathrm{opt}})\mathbf{W}_Q \quad (\text{Optical Query})$", ha='center', fontsize=9.0, color='#1E40AF', fontweight='bold')
    ax.text(14.1, 7.55, r"$\mathbf{K} = \mathrm{LN}(\mathbf{F}_{\mathrm{aux}})\mathbf{W}_K, \quad \mathbf{V} = \mathrm{LN}(\mathbf{F}_{\mathrm{aux}})\mathbf{W}_V$", ha='center', fontsize=9.0, color='#334155', fontweight='bold')
    ax.text(14.1, 6.60, r"$\mathrm{CrossAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{B}\right)\mathbf{V}$", ha='center', fontsize=9.4, color='#BE123C', fontweight='bold')
    ax.text(14.1, 5.65, r"$\mathbf{F}_{\mathrm{fused}} = \mathrm{LN}\left(\mathbf{F}_{\mathrm{opt}} + \mathrm{MLP}(\mathrm{CrossAttention})\right)$", ha='center', fontsize=9.0, color='#0F172A', fontweight='bold')
    ax.text(14.1, 4.90, "Dynamic radar-optical query selectively over cloud-occluded tokens", ha='center', fontsize=8.0, color='#64748B', fontstyle='italic')

    # Skip Connection
    ax.annotate('', xy=(17.4, 8.9), xytext=(10.5, 9.0),
                arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=2.0, ls='--',
                                connectionstyle="arc3,rad=-0.18", mutation_scale=16))
    ax.text(13.9, 10.45, "High-Resolution Optical Skip Connection (Preserves Clear-Sky Radiometry)", fontsize=8.8, color='#2563EB', fontweight='bold', ha='center')

    # Card 4: Multi-Scale Decoder & Dual Outputs
    add_card(17.4, 0.8, 6.1, 9.9, "4. Multi-Scale Decoder & Dual Outputs", bg='#F0FDF4', border='#4ADE80')

    add_module_box(17.7, 7.8, 5.5, 1.4, "Multi-Scale U-Net Decoder", r"Transposed Convolutions + ResBlocks $\to \mathbf{F}_{\mathrm{dec}} \in \mathbb{R}^{48 \times 256 \times 256}$", bg='#ECFEFF', border='#06B6D4', text_color='#0E7490')

    embed_patch(recon_img, 17.8, 4.4, 2.2, "Clear-Sky Output Head", r"$\hat{\mathbf{y}} = \tanh(\mathbf{W}_y \mathbf{F}_{\mathrm{dec}}) \in [-1, 1]$" + "\nPSNR: 29.10 dB | SSIM: 0.963")
    embed_patch(unc_disp, 17.8, 1.2, 2.2, "Uncertainty Variance Head", r"$\mathbf{s} = \log \sigma^2 \in \mathbb{R}$" + "\nCalibrated Error: " + r"$\rho = +0.245$", cmap='plasma')

    # Loss Banner
    loss_box = FancyBboxPatch((0.5, 0.2), 23.0, 0.5, boxstyle="round,pad=0.08,rounding_size=0.1",
                              facecolor='#EEF2FF', edgecolor='#6366F1', lw=1.5)
    ax.add_patch(loss_box)
    loss_str = r"$\mathcal{L}_{\mathrm{Total}} = 1.0\,\mathcal{L}_{\mathrm{NLL}} + 10.0\,\mathcal{L}_{\mathrm{L1}} + 10.0\,\mathcal{L}_{\mathrm{FM}} + 0.1\,\mathcal{L}_{\mathrm{FFT}} + 0.5\,\mathcal{L}_{\mathrm{Sobel}} + 0.2\,\mathcal{L}_{\mathrm{Spec}} + 1.0\,\mathcal{L}_{\mathrm{Adv}}$"
    ax.text(12.0, 0.45, "7-Term Optimization Suite: " + loss_str,
            ha='center', va='center', fontsize=9.2, color='#312E81', fontweight='bold')

    # Vector Arrows
    arrow_kw = dict(arrowstyle="-|>", lw=1.8, mutation_scale=14)
    for y_in, y_stem in [(8.9, 8.8), (6.5, 6.4), (4.1, 4.0), (1.9, 1.7)]:
        ax.annotate('', xy=(6.1, y_stem), xytext=(5.1, y_in), arrowprops=dict(color='#64748B', **arrow_kw))

    ax.annotate('', xy=(8.4, 6.4), xytext=(8.3, 6.4), arrowprops=dict(color='#D97706', **arrow_kw))
    ax.annotate('', xy=(8.4, 4.0), xytext=(8.3, 4.0), arrowprops=dict(color='#7C3AED', **arrow_kw))

    ax.annotate('', xy=(11.4, 3.8), xytext=(10.6, 6.4), arrowprops=dict(color='#D97706', **arrow_kw))
    ax.annotate('', xy=(11.4, 3.6), xytext=(10.6, 4.0), arrowprops=dict(color='#7C3AED', **arrow_kw))
    ax.annotate('', xy=(11.4, 3.4), xytext=(10.6, 1.7), arrowprops=dict(color='#059669', **arrow_kw))

    ax.annotate('', xy=(11.4, 8.6), xytext=(10.6, 8.8), arrowprops=dict(color='#2563EB', **arrow_kw))
    ax.annotate('', xy=(14.1, 4.4), xytext=(14.1, 4.2), arrowprops=dict(color='#64748B', **arrow_kw))

    ax.annotate('', xy=(17.6, 8.5), xytext=(16.7, 7.2), arrowprops=dict(color='#EC4899', **arrow_kw))
    ax.annotate('', xy=(17.7, 5.5), xytext=(17.7, 7.7), arrowprops=dict(color='#06B6D4', **arrow_kw))
    ax.annotate('', xy=(17.7, 2.3), xytext=(17.7, 7.7), arrowprops=dict(color='#06B6D4', **arrow_kw))

    paper_fig = r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\paper\figures\cloudfree_vision_v2_architecture.png"
    pdf_fig = r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\paper\figures\cloudfree_vision_v2_architecture.pdf"
    brain_fig = r"C:\Users\HARSH AMBULE\.gemini\antigravity\brain\1020919c-a94d-481e-bd2f-f32559fc876f\outputs\cloudfree_vision_v2_architecture_sota.png"

    plt.savefig(paper_fig, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_fig, format='pdf', bbox_inches='tight')
    plt.savefig(brain_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Flagship figure regenerated successfully.")


if __name__ == "__main__":
    main()
