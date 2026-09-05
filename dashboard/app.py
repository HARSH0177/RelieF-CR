import os
import sys
import io
import glob
import numpy as np
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))

from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset
from scripts.evaluate import masked_psnr, masked_ssim, masked_sam, masked_ergas, masked_cc, to01, gradient_correlation

st.set_page_config(page_title="RelieF-CR Interactive Dashboard", layout="wide")
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
        st.sidebar.warning("No checkpoint found — using default initialization.")
    gen.eval()
    return gen


def false_color_display(arr_chw):
    """[Green,Red,NIR] tanh-range or [0,1] -> false-color (NIR,Red,Green) uint8 for display."""
    x = arr_chw.copy()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    # NIR (ch2), Red (ch1), Green (ch0)
    if x.shape[0] >= 3:
        disp = np.stack([x[2], x[1], x[0]], axis=-1)
    else:
        disp = np.repeat(x[0:1].transpose(1, 2, 0), 3, axis=-1)
    p2, p98 = np.percentile(disp, (2, 98), axis=(0, 1))
    disp_norm = np.clip((disp - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    return (disp_norm * 255).astype(np.uint8)


st.title("🛰️ RelieF-CR: Multi-Modal Cloud Removal & Uncertainty Dashboard")
st.markdown("**Quad-Modal LISS-IV Optical Reconstruction** (Optical + Sentinel-1 SAR + Sentinel-2 Temporal + CartoDEM)")

data_root = os.path.join(PROJECT_ROOT, "dataset_root")
checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "generator_best.pt")

with st.sidebar:
    st.header("⚙️ Settings & Dataset Browser")
    ckpt_input = st.text_input("Checkpoint path", checkpoint_path)
    
    split_choice = st.selectbox("Select Split", ["test", "val", "train"], index=0)
    split_dir = os.path.join(data_root, split_choice, "opt_cloudy")
    
    patch_list = []
    if os.path.isdir(split_dir):
        patch_list = sorted([os.path.basename(f).replace(".npy", "") for f in glob.glob(os.path.join(split_dir, "*.npy"))])
    
    st.markdown(f"**Available {split_choice} patches:** `{len(patch_list)}`")
    
    mode = st.radio("Input Source", ["Select from Dataset", "Upload Custom Patch (.npy)"])
    
    selected_pid = None
    if mode == "Select from Dataset" and patch_list:
        # Default to high cloud coverage patch
        default_idx = 0
        if "scene_00_p0000" in patch_list:
            default_idx = patch_list.index("scene_00_p0000")
        selected_pid = st.selectbox("Choose Patch ID", patch_list, index=default_idx)
    
    uploaded_opt = None
    if mode == "Upload Custom Patch (.npy)":
        uploaded_opt = st.file_uploader("Upload Cloudy LISS-IV (.npy)", type=["npy"])
        uploaded_sar = st.file_uploader("Upload SAR (.npy, optional)", type=["npy"])
        uploaded_temp = st.file_uploader("Upload Temporal (.npy, optional)", type=["npy"])
        uploaded_dem = st.file_uploader("Upload DEM (.npy, optional)", type=["npy"])
        uploaded_clean = st.file_uploader("Upload Clean Ground Truth (.npy, optional)", type=["npy"])
        uploaded_mask = st.file_uploader("Upload Mask (.npy, optional)", type=["npy"])

# Load generator model
gen = load_generator(ckpt_input)

# Prepare sample data
sample = None
if mode == "Select from Dataset" and selected_pid:
    f_opt = os.path.join(data_root, split_choice, "opt_cloudy", f"{selected_pid}.npy")
    f_clean = os.path.join(data_root, split_choice, "opt_clean", f"{selected_pid}.npy")
    f_sar = os.path.join(data_root, split_choice, "sar", f"{selected_pid}.npy")
    f_temp = os.path.join(data_root, split_choice, "temporal", f"{selected_pid}.npy")
    f_dem = os.path.join(data_root, split_choice, "dem", f"{selected_pid}.npy")
    f_mask = os.path.join(data_root, split_choice, "mask", f"{selected_pid}.npy")
    
    sample = {
        "opt_cloudy": np.load(f_opt),
        "opt_clean": np.load(f_clean) if os.path.exists(f_clean) else None,
        "sar": np.load(f_sar) if os.path.exists(f_sar) else np.zeros((2, 256, 256), dtype=np.float32),
        "temporal": np.load(f_temp) if os.path.exists(f_temp) else np.zeros((3, 256, 256), dtype=np.float32),
        "dem": np.load(f_dem) if os.path.exists(f_dem) else np.zeros((4, 256, 256), dtype=np.float32),
        "mask": np.load(f_mask) if os.path.exists(f_mask) else np.ones((1, 256, 256), dtype=np.float32)
    }

if sample is not None:
    # Run model forward pass
    opt_t = torch.from_numpy(sample["opt_cloudy"]).unsqueeze(0).float().to(DEVICE)
    sar_t = torch.from_numpy(sample["sar"]).unsqueeze(0).float().to(DEVICE)
    temp_t = torch.from_numpy(sample["temporal"]).unsqueeze(0).float().to(DEVICE)
    dem_t = torch.from_numpy(sample["dem"]).unsqueeze(0).float().to(DEVICE)
    
    with torch.no_grad():
        mean_fake, logvar_fake = gen(opt_t, sar_t, temp_t, dem_t)
    
    mean_np = mean_fake[0].cpu().numpy()
    std_np = torch.exp(0.5 * logvar_fake)[0].mean(dim=0).cpu().numpy()
    
    mask_np = sample["mask"]
    cloud_cov = float((mask_np > 0.05).mean() * 100)
    
    # Visual Columns
    st.subheader(f"Patch Inspection: `{selected_pid}` (Cloud Coverage: {cloud_cov:.1f}%)")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**1. Cloudy Optical Input**")
        st.image(false_color_display(sample["opt_cloudy"]), use_container_width=True)
    with c2:
        st.markdown("**2. Cloud / Shadow Mask**")
        st.image((sample["mask"][0] * 255).astype(np.uint8), use_container_width=True, clamp=True)
    with c3:
        st.markdown("**3. Model Reconstruction**")
        st.image(false_color_display(mean_np), use_container_width=True)
    with c4:
        if sample["opt_clean"] is not None:
            st.markdown("**4. Ground Truth Clean**")
            st.image(false_color_display(sample["opt_clean"]), use_container_width=True)
    
    # Auxiliary Modalities
    st.markdown("### Auxiliary Sensor Feeds & Uncertainty Quantification")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown("**Sentinel-1 SAR (VV Channel)**")
        sar_disp = sample["sar"][0]
        p2_s, p98_s = np.percentile(sar_disp, (2, 98))
        sar_norm = np.clip((sar_disp - p2_s) / (p98_s - p2_s + 1e-6), 0.0, 1.0)
        st.image((sar_norm * 255).astype(np.uint8), use_container_width=True)
    with a2:
        st.markdown("**Sentinel-2 Temporal Prior**")
        st.image(false_color_display(sample["temporal"]), use_container_width=True)
    with a3:
        st.markdown("**Topographic DEM (Elevation)**")
        dem_disp = sample["dem"][0]
        dem_norm = (dem_disp - dem_disp.min()) / (dem_disp.ptp() + 1e-6)
        st.image((dem_norm * 255).astype(np.uint8), use_container_width=True)
    with a4:
        st.markdown("**Empirical Residual Error |ŷ - y|**")
        if sample["opt_clean"] is not None:
            clean01 = ((sample["opt_clean"] + 1.0) / 2.0).clip(0, 1)
            fake01 = ((mean_np + 1.0) / 2.0).clip(0, 1)
            res_err = np.abs(fake01 - clean01).mean(axis=0)
            res_norm = np.clip(res_err / 0.20, 0.0, 1.0)
            st.image((res_norm * 255).astype(np.uint8), use_container_width=True)
    
    # Quantitative Metrics for this Patch
    if sample["opt_clean"] is not None:
        st.markdown("### 📊 Patch Evaluation Metrics")
        clean_t = torch.from_numpy(sample["opt_clean"]).unsqueeze(0).float().to(DEVICE)
        mask_t = torch.from_numpy(sample["mask"]).unsqueeze(0).float().to(DEVICE)
        
        c01 = to01(clean_t).clamp(0, 1)
        f01 = to01(mean_fake).clamp(0, 1)
        
        psnr_p = masked_psnr(f01, c01, mask_t)
        ssim_p = masked_ssim(f01, c01, mask_t)
        sam_p = masked_sam(f01, c01, mask_t)
        ergas_p = masked_ergas(f01, c01, mask_t)
        cc_p = masked_cc(f01, c01, mask_t)
        gc_mod = gradient_correlation(f01, sar_t, mask=mask_t)
        gc_gt = gradient_correlation(c01, sar_t, mask=mask_t)
        
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("PSNR (In-Cloud)", f"{psnr_p:.2f} dB")
        m2.metric("SSIM (In-Cloud)", f"{ssim_p:.4f}")
        m3.metric("SAM (Spectral)", f"{sam_p:.2f}°")
        m4.metric("ERGAS", f"{ergas_p:.2f}")
        m5.metric("Correlation (CC)", f"{cc_p:.3f}")
        m6.metric("SAR Edge Corr", f"{gc_mod:.3f}", delta=f"{gc_mod - gc_gt:+.4f}")
        
    st.markdown("---")
    st.markdown("### 💾 Export Reconstructed Output")
    rec_img = Image.fromarray(false_color_display(mean_np))
    buf = io.BytesIO()
    rec_img.save(buf, format="PNG")
    st.download_button("📥 Download Reconstructed PNG", buf.getvalue(), file_name=f"{selected_pid}_reconstructed.png")
