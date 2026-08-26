"""
CloudFree-Vision dashboard v2.

WHAT'S NEW VS V1: the "Confidence" field is now a real number derived from the
generator's predicted log-variance head, not a placeholder. Also updated for
the corrected 3-band [Green,Red,NIR] LISS-IV input and displays a proper
false-color (NIR-Red-Green) composite instead of treating bands as literal RGB.

Run:
  streamlit run dashboard/app.py
"""
import os
import sys
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import numpy as np
import streamlit as st
import torch
from PIL import Image

from models.generator import CloudReconstructionGeneratorV2

st.set_page_config(page_title="CloudFree-Vision", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_generator(checkpoint_path, base_ch=48):
    gen = CloudReconstructionGeneratorV2(base_ch=base_ch).to(DEVICE)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        state_key = "gen_ema_state" if "gen_ema_state" in ckpt else "gen_state"
        gen.load_state_dict(ckpt[state_key])
        st.sidebar.success(f"Loaded '{state_key}' (epoch {ckpt.get('epoch', '?')})")
    else:
        st.sidebar.warning("No checkpoint found — using randomly initialized weights (demo only).")
    gen.eval()
    return gen


def false_color_display(arr_chw):
    """[Green,Red,NIR] tanh-range or [0,1] -> false-color (NIR,Red,Green) uint8 for display."""
    x = arr_chw.copy()
    if x.min() < 0:
        x = (x + 1) / 2
    x = np.clip(x, 0, 1)
    if x.shape[0] >= 3:
        disp = x[[2, 1, 0]]  # NIR, Red, Green
    else:
        disp = np.repeat(x[:1], 3, axis=0)
    return (disp.transpose(1, 2, 0) * 255).astype(np.uint8)


def load_npy_or_random(uploaded_file, shape, fallback_rng, name="input"):
    if uploaded_file is not None:
        arr = np.load(uploaded_file).astype(np.float32)
        if arr.shape != shape:
            st.warning(f"{name}: uploaded shape {arr.shape} differs from expected {shape}. "
                       f"Attempting to proceed — check your preprocessing.")
        return arr
    return fallback_rng.random(shape).astype(np.float32)


st.title("CloudFree-Vision — LISS-IV Cloud Removal Dashboard")
st.caption("Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery")
st.caption("Bands: Green / Red / NIR (LISS-IV has no Blue band) — displayed here as a "
           "false-color NIR-Red-Green composite, the standard remote-sensing convention.")

with st.sidebar:
    st.header("Select Input")
    checkpoint_path = st.text_input("Generator checkpoint path", "checkpoints/generator_best.pt")
    patch_size = st.number_input("Patch size", min_value=64, max_value=512, value=256, step=32)
    st.markdown("---")
    opt_file = st.file_uploader("Cloudy LISS-IV patch (.npy, 3×H×W, [0,1])", type=["npy"])
    sar_file = st.file_uploader("Sentinel-1 SAR patch (.npy, optional)", type=["npy"])
    temporal_file = st.file_uploader("Temporal reference patch (.npy, optional)", type=["npy"])
    dem_file = st.file_uploader("DEM patch (.npy, optional)", type=["npy"])
    clean_file = st.file_uploader("Ground truth clean patch (.npy, optional — enables PSNR/SSIM/SAM)", type=["npy"])
    run_btn = st.button("Run Reconstruction", type="primary")

rng = np.random.default_rng(0)

if run_btn:
    gen = load_generator(checkpoint_path)

    opt_cloudy = load_npy_or_random(opt_file, (3, patch_size, patch_size), rng, "Optical")
    sar = load_npy_or_random(sar_file, (2, patch_size, patch_size), rng, "SAR")
    temporal = load_npy_or_random(temporal_file, (3, patch_size, patch_size), rng, "Temporal")
    dem = load_npy_or_random(dem_file, (4, patch_size, patch_size), rng, "DEM")

    if opt_file is None:
        st.info("No cloudy patch uploaded — showing a random demo patch so the UI is explorable.")

    def prep(x, tanh_range):
        t = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
        return t * 2 - 1 if tanh_range else t

    opt_t = prep(opt_cloudy, tanh_range=True)
    sar_t = prep(sar, tanh_range=False)
    temp_t = prep(temporal, tanh_range=True)
    dem_t = prep(dem, tanh_range=False)

    with torch.no_grad():
        mean_out, logvar_out = gen(opt_t, sar_t, temp_t, dem_t)
        confidence_map = gen.predict_confidence(logvar_out)
    mean_np = mean_out.squeeze(0).cpu().numpy()
    confidence_avg = confidence_map.mean().item()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input: Cloudy LISS-IV (false color)")
        st.image(false_color_display(opt_cloudy), use_container_width=True)
    with col2:
        st.subheader("AI-Reconstructed Cloud-Free (false color)")
        st.image(false_color_display(mean_np), use_container_width=True)

    st.markdown("### Reconstruction Report")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scene Size", f"{patch_size}x{patch_size}")
    m1.metric("Method", "Cross-Attention Fusion GAN")
    m2.metric("Confidence (avg)", f"{confidence_avg:.0f}%")
    m2.caption("Derived from the predicted uncertainty head — not a fixed placeholder.")

    if clean_file is not None:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        from evaluate import masked_psnr, masked_ssim, masked_sam, to01

        clean = np.load(clean_file).astype(np.float32)
        clean_t = prep(clean, tanh_range=True)
        mask_approx = (opt_t - clean_t).abs().mean(dim=1, keepdim=True)
        mask_approx = (mask_approx > mask_approx.mean()).float()

        psnr_val = masked_psnr(to01(mean_out).clamp(0, 1), to01(clean_t), mask_approx)
        ssim_val = masked_ssim(to01(mean_out).clamp(0, 1), to01(clean_t), mask_approx)
        sam_val = masked_sam(to01(mean_out).clamp(0, 1), to01(clean_t), mask_approx)

        m3.metric("PSNR (cloud region)", f"{psnr_val:.2f} dB")
        m3.metric("SSIM (cloud region)", f"{ssim_val:.3f}")
        m4.metric("SAM (spectral)", f"{sam_val:.2f} deg")
    else:
        st.caption("Upload a ground-truth clean patch in the sidebar to compute PSNR / SSIM / SAM.")

    st.markdown("### Uncertainty Map")
    st.caption("Brighter = the model is less confident about the reconstructed value at that pixel "
               "(from the predicted log-variance head). Check `scripts/evaluate.py`'s calibration "
               "correlation before treating this as reliable — it's only meaningful once trained.")
    std_display = torch.exp(0.5 * logvar_out).mean(dim=1).squeeze(0).cpu().numpy()
    st.image(
        (255 * (std_display - std_display.min()) / (std_display.ptp() + 1e-8)).astype(np.uint8),
        use_container_width=True,
        clamp=True,
    )

    st.markdown("### Download")
    out_img = Image.fromarray(false_color_display(mean_np))
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    st.download_button("Download Reconstructed Image (PNG)", buf.getvalue(), file_name="reconstructed.png")
    st.download_button("Download Reconstructed Array (.npy)", mean_np.tobytes(), file_name="reconstructed.npy")
else:
    st.info("Upload patches (or leave blank for a random demo) in the sidebar, then click **Run Reconstruction**.")
