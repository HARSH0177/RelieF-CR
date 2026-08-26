# CloudFree Vision (RelieF-CR)
### Multi-Modal Generative AI for High-Resolution Satellite Cloud Removal & Topographic Reconstruction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PSNR](https://img.shields.io/badge/Test%20PSNR-29.10%20dB-brightgreen.svg)]()
[![SSIM](https://img.shields.io/badge/Test%20SSIM-0.963-success.svg)]()

> An end-to-end deep learning framework fusing **5.0m LISS-IV Optical**, **Sentinel-1 SAR (VV/VH)**, **4-Channel CartoDEM Topography**, and **Sentinel-2 Temporal Priors** via learned **Spatial Transformer Networks** and **Windowed Cross-Attention** to reconstruct cloud-occluded satellite imagery without GAN hallucinations.

---

## ⚡ Key Highlights & Benchmark Results

* 🎯 **29.10 dB PSNR & 0.963 SSIM:** State-of-the-art reconstruction fidelity evaluated across **1,611 held-out test patches** (5m GSD).
* 🌈 **1.49° Spectral Angle Mapper (SAM):** Near-perfect radiometric preservation across Green, Red, and Near-Infrared (NIR) bands for accurate downstream NDVI & vegetation analysis.
* 🛡️ **0.784 SAR-Optical Gradient Correlation:** Verifies that reconstructed edges correspond to physical Sentinel-1 radar backscatter rather than hallucinated GAN artifacts.
* 🔍 **Calibrated Heteroscedastic Uncertainty Head:** Generates per-pixel variance maps ($\log \sigma^2$) allowing operators to inspect model confidence under thick cloud gaps.
* 🗺️ **300 MPix Full-Swath Streaming Inference:** Memory-mapped sliding window inference reconstructing seamless $16,000 \times 18,000$ satellite swaths in **3.2 minutes**.

---

## 🏗️ System Architecture

```
                                 CLOUDFREE-VISION (RelieF-CR) PIPELINE
 
   [ 5m Cloudy Optical ] ───► [ Optical Stem ] ─────────────► [ Query Tokens ] ───────┐
                                                                                       │
   [ Sentinel-1 SAR (VV/VH) ] ──► [ SAR STN Warping ] ──┐                             ▼
                                                        ├───► [ Key/Val Projection ] ─► [ Windowed Cross-Attention ]
   [ Temporal Optical Prior ] ──► [ Temp STN Warping ] ─┘                                      │
                                                                                               ▼
   [ 4-Ch CartoDEM (Z, Slope, sin/cos Aspect) ] ────────────► [ DEM Stem ] ───────────► [ Feature Fusion Trunk ]
                                                                                               │
                                                                                               ▼
                                                                                  [ Deep U-Net Decoder Trunk ]
                                                                                               │
                                                                         ┌─────────────────────┴─────────────────────┐
                                                                         ▼                                           ▼
                                                              [ Reconstructed Optical ]                   [ Uncertainty Map ]
                                                                 (Green, Red, NIR)                       (Predicted Variance)
```

---

## 🛠️ Core Engineering Features

### 1. Spatial Transformer Network (STN) Resolution Alignment
Sentinel-1 SAR ($10\,\text{m}$) and temporal Sentinel-2 ($10\,\text{m}$) are resampled to match native LISS-IV ($5\,\text{m}$) grids. A differentiable localization network predicts dynamic affine matrices $\theta \in \mathbb{R}^{2\times 3}$ to correct inter-sensor spatial jitter before feature fusion.

### 2. Windowed Multi-Head Cross-Attention (Q-K-V Routing)
Rather than naive channel concatenation, optical feature maps act as **Queries**, while aligned SAR and temporal streams act as **Keys and Values**. Local $8\times 8$ window partitioning maintains linear $\mathcal{O}(HW)$ complexity while adaptively routing radar textures only into cloud-occluded pixels.

### 3. Physics-Guided 4-Channel DEM Surface Descriptors
Ingests elevation, Horn's terrain slope, and continuous cyclic aspect decomposition ($\sin \Phi, \cos \Phi$) from CartoDEM, allowing the generator to disambiguate terrain-induced radar layover/shadow from genuine optical cloud cover.

### 4. 7-Term Multi-Domain Composite Loss
* **Frequency-Domain FFT Loss ($\mathcal{L}_{\text{FFT}}$):** Enforces 2D Fourier magnitude consistency to prevent texture oversmoothing.
* **Domain-Agnostic Sobel Structural Loss ($\mathcal{L}_{\text{Sobel}}$):** Preserves field parcel boundaries and sharp linear infrastructure.
* **Multi-Scale Spectral-Normed PatchGAN with Feature Matching ($\mathcal{L}_{\text{FM}}$):** Ensures realistic ground texture synthesis across multiple receptive field scales.
* **Heteroscedastic Gaussian NLL ($\mathcal{L}_{\text{NLL}}$):** Attenuates loss penalty in ambiguous, opaque cloud cores to stabilize training.

---

## 📊 Quantitative Benchmarks

### 1. Model Comparison on 1,611 Held-Out Test Patches (5m LISS-IV)

| Method | Paradigm | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | SAM ($^\circ$) $\downarrow$ | ERGAS $\downarrow$ | CC $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Inpainting** | Spatial Interpolation | 6.21 | 0.386 | 10.35 | 166.90 | 0.327 |
| **Temporal Baseline** | Clear-Sky Dry Season Prior | 13.11 | 0.656 | 7.48 | 73.99 | 0.397 |
| **CloudFree Vision (Ours)** | **Quad-Modal STN Cross-Attention** | **29.10** | **0.963** | **1.49** | **7.89** | **0.863** |

---

### 2. 5-Variant Systematic Ablation Study (50 Epochs)

| Architecture Variant | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | SAM ($^\circ$) $\downarrow$ | ERGAS $\downarrow$ | Key Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Proposed Model** | **29.10** | **0.963** | **1.49** | **7.89** | **Complete Quad-Modal System** |
| w/o 4-Channel DEM Topography | 23.07 | 0.921 | 1.80 | 11.51 | $-6.03\,\text{dB}$ drop; slope/aspect resolves terrain radar layover |
| w/o STN Sub-Pixel Alignment | 22.72 | 0.919 | 1.84 | 11.86 | $-6.38\,\text{dB}$ drop; affine warping prevents cross-sensor blur |
| w/o Frequency FFT Loss ($\mathcal{L}_{\text{FFT}}$) | 21.99 | 0.920 | 1.88 | 12.92 | $-7.11\,\text{dB}$ drop; Fourier loss eliminates spectral oversmoothing |
| w/o Cross-Attention (Concat) | 24.59 | 0.930 | 1.71 | 9.70 | $-4.51\,\text{dB}$ drop; dynamic Q-K-V routing isolates cloud tokens |
| w/o Uncertainty Head ($\mathcal{L}_{\text{L1}}$ only) | 23.02 | 0.924 | 1.89 | 11.44 | $-6.08\,\text{dB}$ drop; heteroscedastic loss stabilizes GAN gradients |

---

## 💻 Quickstart & Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/HARSH0177/RelieF-CR.git
cd RelieF-CR

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Smoke Tests & Verification (No GPU required)

```bash
# Verify model architectures and forward tensor shapes
python models/generator.py
python models/discriminator.py
python models/losses.py

# Run a 1-epoch synthetic smoke test
python scripts/train_generator.py --synthetic --epochs 1 --batch_size 2 --device cpu
```

### 3. Launch the Interactive Streamlit Web Dashboard

```bash
streamlit run dashboard/app.py
```
*Features interactive cloud mask synthesis, false-color optical/SAR visualization, real-time confidence heatmap inspection, and GeoTIFF export.*

---

## 🏋️ Training & Evaluation

### Train on Custom Dataset

```bash
python scripts/train_generator.py \
    --data_root dataset_root \
    --epochs 50 \
    --batch_size 16 \
    --lr 2e-4 \
    --device cuda
```

### Evaluate on Test Set

```bash
python scripts/evaluate.py \
    --data_root dataset_root \
    --checkpoint checkpoints/generator_best.pt \
    --device cuda
```

### Full-Swath Scene Inference (300 MPix)

```bash
python scripts/infer_full_scene.py \
    --checkpoint checkpoints/generator_best.pt \
    --patch_size 256 \
    --stride 192 \
    --device cuda
```

---

## 📁 Repository Structure

```
CloudFree-Vision/
├── models/
│   ├── attention.py          # Windowed Self- & Cross-Attention with Q-K-V routing
│   ├── generator.py          # RelieF-CR Quad-Modal Generator with STN & dual heads
│   ├── discriminator.py      # Multi-Scale PatchGAN with Spectral Normalization
│   ├── losses.py             # 7-Term composite loss (FFT, Sobel, FM, NLL, Adv, NDVI)
│   └── segmentation.py       # Cloud and shadow segmentation baseline
├── data/
│   └── dataset.py            # Multi-modal dataset loader & organic cloud simulator
├── scripts/
│   ├── train_generator.py    # Main training engine (AMP, EMA, Cosine Annealing)
│   ├── evaluate.py           # Evaluation suite (PSNR, SSIM, SAM, ERGAS, CC)
│   ├── infer_full_scene.py   # Sliding-window full-swath memory-mapped inference
│   ├── make_comparison_grid.py # Qualitative comparison visualizer
│   ├── aggregate_ablation_results.py # Automated Table III reproduction script
│   ├── ablate1_without_dem.py # Ablation 1 training script
│   ├── ablate2_without_stn.py # Ablation 2 training script
│   ├── ablate3_without_fft.py # Ablation 3 training script
│   ├── ablate4_without_cross_attention.py # Ablation 4 training script
│   └── ablate5_without_uncertainty.py     # Ablation 5 training script
├── dashboard/
│   └── app.py                # Streamlit visualization application
├── checkpoints/              # Pretrained weights & ablation results JSONs
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
