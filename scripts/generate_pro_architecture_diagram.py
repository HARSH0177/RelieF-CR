"""
Pro-Max 3D Isometric & Vector-Engineered Neural Architecture Diagram Generator
for IEEE Transactions on Geoscience and Remote Sensing (TGRS).
Refined with clean layout spacing, zero overlaps, and high-DPI vector precision.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon


def draw_isometric_cube(ax, x, y, dx, dy, dz, color_front, color_top, color_side, alpha=0.90, label_above="", shape_text="", label_below=""):
    """
    Draws a 3D isometric cuboid representing a multi-channel convolutional tensor.
    x, y: Base origin
    dx: Width
    dy: Height
    dz: Depth (isometric projection vector: dx_iso = 0.45*dz, dy_iso = 0.28*dz)
    """
    iso_x = 0.45 * dz
    iso_y = 0.28 * dz

    # Front Face
    front_verts = [(x, y), (x + dx, y), (x + dx, y + dy), (x, y + dy)]
    front = Polygon(front_verts, closed=True, facecolor=color_front, edgecolor='#1E293B', lw=1.2, alpha=alpha)
    ax.add_patch(front)

    # Top Face
    top_verts = [(x, y + dy), (x + dx, y + dy), (x + dx + iso_x, y + dy + iso_y), (x + iso_x, y + dy + iso_y)]
    top = Polygon(top_verts, closed=True, facecolor=color_top, edgecolor='#1E293B', lw=1.2, alpha=alpha)
    ax.add_patch(top)

    # Right Side Face
    side_verts = [(x + dx, y), (x + dx + iso_x, y + iso_y), (x + dx + iso_x, y + dy + iso_y), (x + dx, y + dy)]
    side = Polygon(side_verts, closed=True, facecolor=color_side, edgecolor='#1E293B', lw=1.2, alpha=alpha)
    ax.add_patch(side)

    # Labels
    if label_above:
        ax.text(x + dx/2 + iso_x/2, y + dy + iso_y + 0.15, label_above, ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#0F172A')
    if label_below:
        ax.text(x + dx/2, y - 0.22, label_below, ha='center', va='top', fontsize=8.0, color='#334155', fontweight='bold')
    if shape_text:
        ax.text(x + dx/2, y + dy/2, shape_text, ha='center', va='center', fontsize=7.2, color='#FFFFFF', fontweight='bold', family='monospace')

    return (x + dx + iso_x, y + dy/2 + iso_y/2)


def generate_pro_architecture():
    fig, ax = plt.subplots(figsize=(23, 11.5), dpi=300)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 11.5)
    ax.axis('off')

    # Canvas Background
    bg = FancyBboxPatch((0.2, 0.2), 22.6, 11.1, boxstyle="round,pad=0.2",
                        facecolor='#FAFAFA', edgecolor='#E2E8F0', lw=1.5)
    ax.add_patch(bg)

    # Title Banner
    ax.text(11.5, 10.95, "CloudFree Vision v2: Multi-Modal SAR-Optical-Topographic Cross-Attention Architecture",
            ha='center', va='center', fontsize=16, fontweight='bold', color='#0F172A')
    ax.text(11.5, 10.55, "Dual-Head Heteroscedastic Generator with Learned Spatial Transformer Alignment & 7-Term Spectral-Structural Loss",
            ha='center', va='center', fontsize=10.5, fontstyle='italic', color='#475569')

    # =========================================================================
    # 1. INPUT MODALITIES (Left Column)
    # =========================================================================
    g1 = FancyBboxPatch((0.5, 1.1), 3.6, 9.1, boxstyle="round,pad=0.15",
                        facecolor='#F1F5F9', edgecolor='#CBD5E1', lw=1.5, ls='--')
    ax.add_patch(g1)
    ax.text(2.3, 9.85, "Multi-Sensor Inputs", ha='center', fontsize=11, fontweight='bold', color='#1E293B')

    # Optical: 3 Bands [G, R, NIR]
    draw_isometric_cube(ax, 0.9, 7.8, 1.5, 1.3, 0.6, '#3B82F6', '#60A5FA', '#2563EB',
                        label_above="Cloudy Optical (LISS-IV)", shape_text="3×256²")

    # SAR: 2 Channels [VV, VH dB]
    draw_isometric_cube(ax, 0.9, 5.6, 1.5, 1.3, 0.5, '#F59E0B', '#FBBF24', '#D97706',
                        label_above="Sentinel-1 SAR (VV/VH)", shape_text="2×256²")

    # Temporal: 3 Bands
    draw_isometric_cube(ax, 0.9, 3.4, 1.5, 1.3, 0.6, '#8B5CF6', '#A78BFA', '#7C3AED',
                        label_above="Temporal Ref (Sentinel-2)", shape_text="3×256²")

    # DEM: 4 Channels
    draw_isometric_cube(ax, 0.9, 1.3, 1.5, 1.3, 0.8, '#10B981', '#34D399', '#059669',
                        label_above="CartoDEM (4-Ch Horn)", shape_text="4×256²")

    # =========================================================================
    # 2. FEATURE CONVOLUTION STEMS
    # =========================================================================
    draw_isometric_cube(ax, 4.6, 7.8, 0.9, 1.3, 0.8, '#2563EB', '#3B82F6', '#1D4ED8',
                        label_above="Optical Stem", shape_text="48×128²")

    draw_isometric_cube(ax, 4.6, 5.6, 0.9, 1.3, 0.8, '#D97706', '#F59E0B', '#B45309',
                        label_above="SAR Stem", shape_text="48×128²")

    draw_isometric_cube(ax, 4.6, 3.4, 0.9, 1.3, 0.8, '#7C3AED', '#8B5CF6', '#6D28D9',
                        label_above="Temporal Stem", shape_text="48×128²")

    draw_isometric_cube(ax, 4.6, 1.3, 0.9, 1.3, 0.7, '#059669', '#10B981', '#047857',
                        label_above="DEM Stem", shape_text="32×128²")

    # Arrows Input -> Stem
    for y_val in [8.5, 6.3, 4.1, 2.0]:
        ax.annotate('', xy=(4.5, y_val), xytext=(2.8, y_val),
                    arrowprops=dict(arrowstyle="-|>", color='#64748B', lw=1.8, mutation_scale=13))

    # =========================================================================
    # 3. SPATIAL TRANSFORMER NETWORKS (STN)
    # =========================================================================
    stn1 = FancyBboxPatch((6.6, 5.5), 1.9, 1.5, boxstyle="round,pad=0.1",
                          facecolor='#FEE2E2', edgecolor='#EF4444', lw=1.5)
    ax.add_patch(stn1)
    ax.text(7.55, 6.5, "STN SAR", ha='center', fontsize=9.0, fontweight='bold', color='#991B1B')
    ax.text(7.55, 5.9, r"$\mathcal{T}_{\theta} \in \mathbb{R}^{2\times 3}$" + "\nAffine Grid", ha='center', fontsize=7.5, color='#B91C1C')

    stn2 = FancyBboxPatch((6.6, 3.3), 1.9, 1.5, boxstyle="round,pad=0.1",
                          facecolor='#FEE2E2', edgecolor='#EF4444', lw=1.5)
    ax.add_patch(stn2)
    ax.text(7.55, 4.3, "STN Temporal", ha='center', fontsize=9.0, fontweight='bold', color='#991B1B')
    ax.text(7.55, 3.7, r"$\mathcal{T}_{\theta} \in \mathbb{R}^{2\times 3}$" + "\nAffine Grid", ha='center', fontsize=7.5, color='#B91C1C')

    ax.annotate('', xy=(6.5, 6.3), xytext=(5.8, 6.3), arrowprops=dict(arrowstyle="-|>", color='#D97706', lw=1.8, mutation_scale=12))
    ax.annotate('', xy=(6.5, 4.1), xytext=(5.8, 4.1), arrowprops=dict(arrowstyle="-|>", color='#7C3AED', lw=1.8, mutation_scale=12))

    # Optical Reference guidance
    ax.annotate('', xy=(7.55, 7.1), xytext=(5.6, 8.0),
                arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.4, ls=':', mutation_scale=12, connectionstyle="arc3,rad=-0.2"))
    ax.text(6.3, 7.7, "Opt Guide", fontsize=7.5, color='#2563EB', fontweight='bold')

    # =========================================================================
    # 4. CONCATENATED AUXILIARY FEATURES
    # =========================================================================
    draw_isometric_cube(ax, 9.4, 3.2, 1.1, 2.8, 1.8, '#64748B', '#94A3B8', '#475569',
                        label_above="Concatenated\nAuxiliary Features", shape_text="160×128²")

    ax.annotate('', xy=(9.3, 5.6), xytext=(8.6, 6.2), arrowprops=dict(arrowstyle="-|>", color='#D97706', lw=1.6, mutation_scale=12))
    ax.annotate('', xy=(9.3, 4.4), xytext=(8.6, 4.0), arrowprops=dict(arrowstyle="-|>", color='#7C3AED', lw=1.6, mutation_scale=12))
    ax.annotate('', xy=(9.3, 3.4), xytext=(5.8, 2.0), arrowprops=dict(arrowstyle="-|>", color='#059669', lw=1.6, mutation_scale=12))

    # =========================================================================
    # 5. WINDOWED MULTI-HEAD CROSS-ATTENTION
    # =========================================================================
    attn_box = FancyBboxPatch((11.9, 3.8), 3.4, 4.7, boxstyle="round,pad=0.15",
                              facecolor='#FDF2F8', edgecolor='#EC4899', lw=2.0)
    ax.add_patch(attn_box)
    ax.text(13.6, 8.1, "Windowed Multi-Head\nCross-Attention", ha='center', fontsize=10.5, fontweight='bold', color='#9D174D')

    q_box = FancyBboxPatch((12.2, 6.9), 2.8, 0.7, boxstyle="round,pad=0.08", facecolor='#DBEAFE', edgecolor='#3B82F6', lw=1.2)
    ax.add_patch(q_box)
    ax.text(13.6, 7.25, r"Query $\mathbf{Q} = \mathbf{W}_Q \mathbf{F}_{\mathrm{opt}}$ (Optical)", ha='center', fontsize=7.5, color='#1E40AF', fontweight='bold')

    kv_box = FancyBboxPatch((12.2, 5.9), 2.8, 0.7, boxstyle="round,pad=0.08", facecolor='#F1F5F9', edgecolor='#64748B', lw=1.2)
    ax.add_patch(kv_box)
    ax.text(13.6, 6.25, r"Keys $\mathbf{K}$, Values $\mathbf{V} = \mathbf{W}_{K,V} \mathbf{F}_{\mathrm{aux}}$", ha='center', fontsize=7.5, color='#334155', fontweight='bold')

    math_box = FancyBboxPatch((12.2, 4.2), 2.8, 1.4, boxstyle="round,pad=0.08", facecolor='#FCE7F3', edgecolor='#F43F5E', lw=1.2)
    ax.add_patch(math_box)
    ax.text(13.6, 5.15, r"$\mathrm{Attn} = \mathrm{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{B}\right)\mathbf{V}$",
            ha='center', fontsize=8.0, color='#881337', fontweight='bold')
    ax.text(13.6, 4.55, "Local 8×8 Window Tokens", ha='center', fontsize=7.2, color='#9F1239')

    # Connections into Attention
    ax.annotate('', xy=(12.1, 7.25), xytext=(5.8, 8.5),
                arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.8, mutation_scale=14))
    ax.annotate('', xy=(12.1, 6.25), xytext=(11.0, 4.9),
                arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.8, mutation_scale=14))

    # =========================================================================
    # 6. MULTI-SCALE U-NET DECODER
    # =========================================================================
    draw_isometric_cube(ax, 16.0, 4.4, 1.2, 2.5, 1.2, '#06B6D4', '#22D3EE', '#0891B2',
                        label_above="Multi-Scale\nDecoder", shape_text="48×256²")

    ax.annotate('', xy=(15.9, 6.1), xytext=(15.4, 6.1),
                arrowprops=dict(arrowstyle="-|>", color='#EC4899', lw=2.0, mutation_scale=14))

    # High-Res Skip Connection smoothly routing above
    ax.annotate('', xy=(16.6, 7.5), xytext=(5.6, 9.4),
                arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.8, ls='--',
                                connectionstyle="arc3,rad=-0.12", mutation_scale=14))
    ax.text(11.5, 9.7, "High-Resolution Optical Skip Connection (Clear-Pixel Preservation)", fontsize=8.5, color='#2563EB', fontweight='bold', ha='center')

    # =========================================================================
    # 7. DUAL-HEAD HETEROSCEDASTIC OUTPUT
    # =========================================================================
    g_out = FancyBboxPatch((18.1, 2.2), 4.3, 7.0, boxstyle="round,pad=0.15",
                           facecolor='#F0FDF4', edgecolor='#22C55E', lw=1.8, ls='--')
    ax.add_patch(g_out)
    ax.text(20.25, 8.8, "Dual-Head Heteroscedastic Output", ha='center', fontsize=10.5, fontweight='bold', color='#15803D')

    # Head 1: Clear-Sky Mean
    draw_isometric_cube(ax, 18.6, 6.1, 1.4, 1.4, 0.6, '#10B981', '#34D399', '#059669',
                        label_above=r"Clear-Sky Mean $\hat{\mathbf{y}}$ (LISS-IV)", shape_text="3×256²")

    # Head 2: Uncertainty Log-Variance
    draw_isometric_cube(ax, 18.6, 3.0, 1.4, 1.4, 0.6, '#EF4444', '#F87171', '#DC2626',
                        label_above=r"Uncertainty $\log \sigma^2$" + "\n" + r"($\rho = +0.245$)", shape_text="3×256²")

    ax.annotate('', xy=(18.5, 7.0), xytext=(17.6, 6.4), arrowprops=dict(arrowstyle="-|>", color='#0891B2', lw=1.8, mutation_scale=13))
    ax.annotate('', xy=(18.5, 3.9), xytext=(17.6, 4.9), arrowprops=dict(arrowstyle="-|>", color='#0891B2', lw=1.8, mutation_scale=13))

    # =========================================================================
    # 8. LOSS FRAMEWORK BANNER (Bottom)
    # =========================================================================
    loss_box = FancyBboxPatch((0.5, 0.3), 21.9, 0.7, boxstyle="round,pad=0.08",
                              facecolor='#EEF2FF', edgecolor='#6366F1', lw=1.4)
    ax.add_patch(loss_box)
    loss_str = r"$\mathcal{L}_{\mathrm{Total}} = 1.0\,\mathcal{L}_{\mathrm{NLL}} + 10.0\,\mathcal{L}_{\mathrm{L1}} + 10.0\,\mathcal{L}_{\mathrm{FM}} + 0.1\,\mathcal{L}_{\mathrm{FFT}} + 0.5\,\mathcal{L}_{\mathrm{Sobel}} + 0.2\,\mathcal{L}_{\mathrm{Spec}} + 1.0\,\mathcal{L}_{\mathrm{Adv}}$"
    ax.text(11.45, 0.65, "Comprehensive 7-Term Loss Suite: Dual-Head NLL + Frequency FFT Spectrum + Multi-Scale Sobel + NDVI Constraints",
            ha='center', fontsize=8.5, fontweight='bold', color='#312E81')
    ax.text(11.45, 0.42, loss_str, ha='center', fontsize=9.0, color='#4338CA', fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    paper_fig_dir = r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\paper\figures"
    os.makedirs(paper_fig_dir, exist_ok=True)

    brain_fig_dir = r"C:\Users\HARSH AMBULE\.gemini\antigravity\brain\1020919c-a94d-481e-bd2f-f32559fc876f\outputs"
    os.makedirs(brain_fig_dir, exist_ok=True)

    fig = generate_pro_architecture()

    png_path = os.path.join(paper_fig_dir, "cloudfree_vision_v2_architecture.png")
    pdf_path = os.path.join(paper_fig_dir, "cloudfree_vision_v2_architecture.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

    brain_png = os.path.join(brain_fig_dir, "cloudfree_vision_v2_architecture_pro.png")
    fig.savefig(brain_png, dpi=300, bbox_inches='tight')

    plt.close(fig)
    print(f"Generated clean Pro 3D Isometric Architecture Diagram successfully:")
    print(f"  - PNG: {png_path}")
    print(f"  - PDF: {pdf_path}")


if __name__ == "__main__":
    main()
