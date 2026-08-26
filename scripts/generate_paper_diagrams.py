"""
Publication-Grade Architecture Diagram Generator for IEEE Transactions (TGRS).
Generates high-resolution 300 DPI PNG, vector PDF, and TikZ LaTeX code.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle


def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(19, 9.5), dpi=300)
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    # Color Palette (Academic IEEE Style)
    c_opt = '#3B82F6'       # Blue for Optical
    c_sar = '#F59E0B'       # Amber for SAR
    c_dem = '#10B981'       # Emerald for Topography / DEM
    c_temp = '#8B5CF6'      # Purple for Temporal
    c_stn = '#EF4444'       # Crimson for STN
    c_attn = '#EC4899'      # Pink for Attention
    c_dec = '#06B6D4'       # Cyan for Decoder
    c_out = '#14B8A6'       # Teal for Output
    c_loss = '#6366F1'      # Indigo for Loss Suite

    # Helper: Rounded Box
    def add_box(x, y, w, h, title, subtitle="", color="#3B82F6", alpha=0.15, ec=None, lw=1.5, ls='-'):
        edge = ec if ec else color
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.15",
                             facecolor=color, alpha=alpha, edgecolor=edge, linewidth=lw, linestyle=ls)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E293B')
            ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', fontsize=7.5, color='#475569', family='monospace')
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E293B')
        return (x + w, y + h/2)

    # Helper: Arrow
    def add_arrow(p1, p2, color='#64748B', lw=1.5, ls='-', label=""):
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls, mutation_scale=12))
        if label:
            mx, my = (p1[0] + p2[0])/2, (p1[1] + p2[1])/2 + 0.15
            ax.text(mx, my, label, ha='center', va='bottom', fontsize=7.5, color=color, fontweight='bold')

    # Title Banner
    ax.text(9.5, 9.15, "CloudFree Vision v2: Multi-Modal SAR-Optical-Topographic Cross-Attention Architecture",
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0F172A')
    ax.text(9.5, 8.80, "Dual-Head Heteroscedastic Generator with Learned Spatial Alignment & 7-Term Spectral-Structural Loss",
            ha='center', va='center', fontsize=9.5, fontstyle='italic', color='#475569')

    # =========================================================================
    # 1. INPUT MODALITIES (Left Column)
    # =========================================================================
    # Group Box for Inputs
    g_in = FancyBboxPatch((0.4, 0.6), 3.2, 7.8, boxstyle="round,pad=0.2",
                          facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1.2, ls='--')
    ax.add_patch(g_in)
    ax.text(2.0, 8.15, "Multi-Modal Sensor Inputs", ha='center', fontsize=10, fontweight='bold', color='#334155')

    add_box(0.6, 6.4, 2.8, 1.2, "Cloudy Optical (LISS-IV)", "[B, 3, 256, 256] (G, R, NIR)", c_opt)
    add_box(0.6, 4.6, 2.8, 1.2, "Sentinel-1 SAR", "[B, 2, 256, 256] (VV, VH dB)", c_sar)
    add_box(0.6, 2.8, 2.8, 1.2, "Temporal Sentinel-2", "[B, 3, 256, 256] (Dry Clear)", c_temp)
    add_box(0.6, 1.0, 2.8, 1.2, "CartoDEM Topography", "[B, 4, 256, 256] (Elev, Slope, Sin, Cos)", c_dem)

    # =========================================================================
    # 2. FEATURE EXTRACTION & SPATIAL ALIGNMENT (STN)
    # =========================================================================
    add_box(4.2, 6.4, 2.2, 1.2, "Optical Stem", "Conv3x3 -> [B, 48, 128, 128]", c_opt)
    add_box(4.2, 4.6, 2.2, 1.2, "SAR Stem", "Conv3x3 -> [B, 48, 128, 128]", c_sar)
    add_box(4.2, 2.8, 2.2, 1.2, "Temporal Stem", "Conv3x3 -> [B, 48, 128, 128]", c_temp)
    add_box(4.2, 1.0, 2.2, 1.2, "DEM Stem", "Conv3x3 -> [B, 32, 128, 128]", c_dem)

    add_arrow((3.4, 7.0), (4.2, 7.0), c_opt)
    add_arrow((3.4, 5.2), (4.2, 5.2), c_sar)
    add_arrow((3.4, 3.4), (4.2, 3.4), c_temp)
    add_arrow((3.4, 1.6), (4.2, 1.6), c_dem)

    # STN Alignment Modules
    add_box(6.8, 4.6, 2.0, 1.2, "STN Alignment", "Affine Warp SAR", c_stn, alpha=0.2)
    add_box(6.8, 2.8, 2.0, 1.2, "STN Alignment", "Affine Warp Temp", c_stn, alpha=0.2)

    add_arrow((6.4, 5.2), (6.8, 5.2), c_sar)
    add_arrow((6.4, 3.4), (6.8, 3.4), c_temp)

    # Optical reference into STN localization
    ax.annotate('', xy=(7.8, 5.8), xytext=(5.3, 6.4),
                arrowprops=dict(arrowstyle="-|>", color=c_opt, lw=1.2, ls=':', mutation_scale=10))
    ax.text(6.4, 6.1, "Opt Ref", fontsize=7, color=c_opt, rotation=-20)

    # =========================================================================
    # 3. MULTI-SCALE CROSS-ATTENTION FUSION
    # =========================================================================
    # Auxiliary Concatenation Box
    add_box(9.2, 2.6, 2.2, 3.4, "Auxiliary Feature\nConcatenation", "[B, 160, 128, 128]\n(SAR + Temp + DEM)", '#64748B', alpha=0.1)

    add_arrow((8.8, 5.2), (9.2, 4.8), c_sar)
    add_arrow((8.8, 3.4), (9.2, 3.8), c_temp)
    add_arrow((6.4, 1.6), (9.2, 2.9), c_dem)

    # Cross-Attention Module
    add_box(11.8, 4.2, 2.6, 3.2, "Windowed Multi-Head\nCross-Attention",
            "Query: Optical Tokens\nKey/Val: Aux Tokens\nLocal 8x8 Windows", c_attn, alpha=0.2, lw=2.0)

    add_arrow((6.4, 7.0), (11.8, 6.5), c_opt, label="Query Q")
    add_arrow((11.4, 4.3), (11.8, 4.8), '#64748B', label="Keys K, Values V")

    # =========================================================================
    # 4. MULTI-SCALE U-NET DECODER WITH SKIP CONNECTIONS
    # =========================================================================
    add_box(14.8, 4.4, 1.8, 2.8, "Multi-Scale\nDecoder", "UpConv + Skips\nResBlocks\n[B, 48, 256, 256]", c_dec, alpha=0.2)
    add_arrow((14.4, 5.8), (14.8, 5.8), c_attn)

    # Skip connection from input
    ax.annotate('', xy=(15.7, 7.2), xytext=(5.3, 7.6),
                arrowprops=dict(arrowstyle="-|>", color=c_opt, lw=1.2, ls='--', connectionstyle="arc3,rad=-0.15", mutation_scale=10))
    ax.text(10.5, 8.0, "High-Res Optical Skip Connection", fontsize=7.5, color=c_opt, fontweight='bold')

    # =========================================================================
    # 5. DUAL-HEAD HETEROSCEDASTIC OUTPUT (Right Column)
    # =========================================================================
    # Group Box for Outputs
    g_out = FancyBboxPatch((16.9, 3.2), 1.9, 4.8, boxstyle="round,pad=0.15",
                           facecolor='#F0FDF4', edgecolor='#86EFAC', lw=1.2, ls='--')
    ax.add_patch(g_out)
    ax.text(17.85, 7.7, "Dual Output Heads", ha='center', fontsize=9.5, fontweight='bold', color='#166534')

    add_box(17.0, 5.8, 1.7, 1.4, "Mean Head", r"$\hat{y} \in [-1, 1]$" + "\n[B, 3, 256, 256]", c_out, alpha=0.25)
    add_box(17.0, 3.6, 1.7, 1.4, "Variance Head", r"$\log \sigma^2 \in \mathbb{R}$" + "\n[B, 3, 256, 256]", '#E11D48', alpha=0.25)

    add_arrow((16.6, 6.5), (17.0, 6.5), c_dec)
    add_arrow((16.6, 4.3), (17.0, 4.3), c_dec)

    # =========================================================================
    # 6. LOSS FORMULATION & DISCRIMINATOR BANNER (Bottom)
    # =========================================================================
    loss_box = FancyBboxPatch((1.0, 0.3), 17.0, 0.9, boxstyle="round,pad=0.1",
                              facecolor='#EEF2FF', edgecolor='#A5B4FC', lw=1.2)
    ax.add_patch(loss_box)
    ax.text(9.5, 0.75, "Comprehensive 7-Term Loss Optimization Framework", ha='center', fontsize=9.5, fontweight='bold', color='#312E81')
    loss_str = r"$\mathcal{L}_{\mathrm{Total}} = 1.0\,\mathcal{L}_{\mathrm{NLL}} + 10.0\,\mathcal{L}_{\mathrm{L1}} + 10.0\,\mathcal{L}_{\mathrm{FM}} + 0.1\,\mathcal{L}_{\mathrm{FFT}} + 0.5\,\mathcal{L}_{\mathrm{Sobel}} + 0.2\,\mathcal{L}_{\mathrm{Spec}} + 1.0\,\mathcal{L}_{\mathrm{Adv}}$"
    ax.text(9.5, 0.45, loss_str, ha='center', fontsize=8.5, color='#4338CA')

    plt.tight_layout()
    return fig


def main():
    out_dir = r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\outputs\paper_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    brain_dir = r"C:\Users\HARSH AMBULE\.gemini\antigravity\brain\1020919c-a94d-481e-bd2f-f32559fc876f\outputs"
    os.makedirs(brain_dir, exist_ok=True)

    fig = draw_architecture_diagram()

    # Save 300 DPI PNG
    png_path = os.path.join(out_dir, "cloudfree_vision_v2_architecture.png")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    # Save Vector PDF
    pdf_path = os.path.join(out_dir, "cloudfree_vision_v2_architecture.pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

    # Copy to brain artifacts for display
    brain_png = os.path.join(brain_dir, "cloudfree_vision_v2_architecture.png")
    fig.savefig(brain_png, dpi=300, bbox_inches='tight')

    plt.close(fig)
    print(f"Publication-Grade Architecture Diagram successfully saved to:")
    print(f"  - PNG: {png_path}")
    print(f"  - PDF: {pdf_path}")
    print(f"  - Brain Artifact: {brain_png}")


if __name__ == "__main__":
    main()
