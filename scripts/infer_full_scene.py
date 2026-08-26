"""
Full-Scene Sliding-Window Inference Engine with Seamless 2D Hann Window Blending.

Tiles an arbitrarily large multi-modal scene, runs generator per tile, and blends
overlapping tiles back into one seamless output using 2D Hann windowing.

Now computes & prints Regime B metrics (No-Ref Perceptual Score and SAR-Optical Grad Corr)
on the full seamless output.
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

import numpy as np
import torch
import torch.nn.functional as F

from models.generator import CloudReconstructionGeneratorV2
from evaluate import evaluate_no_reference

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def hann_window_2d(size):
    w1d = np.hanning(size)
    w1d = np.clip(w1d, 1e-3, None)
    return np.outer(w1d, w1d).astype(np.float32)


def sliding_window_infer(gen, opt_full, sar_full, temporal_full, dem_full, device,
                          patch_size=256, overlap=64, batch_size=4):
    """
    opt_full/sar_full/temporal_full/dem_full: numpy arrays (C,H,W), reflectance
    scaled to [0,1] or appropriately normalized.
    Returns: mean_recon (C,H,W) in [0,1], std_map (C,H,W).
    """
    C_opt, H, W = opt_full.shape
    stride = patch_size - overlap
    window = hann_window_2d(patch_size)

    pad_h = (patch_size - H % stride) % stride if H > patch_size else patch_size - H
    pad_w = (patch_size - W % stride) % stride if W > patch_size else patch_size - W
    pad_h, pad_w = max(pad_h, 0), max(pad_w, 0)

    def pad(x):
        return np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    opt_p, sar_p, temp_p, dem_p = pad(opt_full), pad(sar_full), pad(temporal_full), pad(dem_full)
    Hp, Wp = opt_p.shape[1], opt_p.shape[2]

    accum = np.zeros((C_opt, Hp, Wp), dtype=np.float32)
    accum_var = np.zeros((C_opt, Hp, Wp), dtype=np.float32)
    accum_w2 = np.zeros((1, Hp, Wp), dtype=np.float32)
    weight_sum = np.zeros((1, Hp, Wp), dtype=np.float32)

    ys = list(range(0, max(Hp - patch_size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(Wp - patch_size, 0) + 1, stride)) or [0]
    if ys[-1] != Hp - patch_size:
        ys.append(Hp - patch_size)
    if xs[-1] != Wp - patch_size:
        xs.append(Wp - patch_size)

    tiles = [(y, x) for y in ys for x in xs]
    print(f"  Total tiles to process: {len(tiles)} (grid {len(ys)}x{len(xs)}, stride={stride}px)")
    gen.eval()
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            opt_batch, sar_batch, temp_batch, dem_batch = [], [], [], []
            for (y, x) in batch_tiles:
                opt_batch.append(opt_p[:, y:y + patch_size, x:x + patch_size] * 2 - 1)
                sar_batch.append(sar_p[:, y:y + patch_size, x:x + patch_size])
                temp_batch.append(temp_p[:, y:y + patch_size, x:x + patch_size] * 2 - 1)
                dem_batch.append(dem_p[:, y:y + patch_size, x:x + patch_size])

            opt_t = torch.from_numpy(np.stack(opt_batch)).float().to(device)
            sar_t = torch.from_numpy(np.stack(sar_batch)).float().to(device)
            temp_t = torch.from_numpy(np.stack(temp_batch)).float().to(device)
            dem_t = torch.from_numpy(np.stack(dem_batch)).float().to(device)

            mean_out, logvar_out = gen(opt_t, sar_t, temp_t, dem_t)
            mean01 = ((mean_out + 1) / 2).clamp(0, 1).cpu().numpy()
            var01 = torch.exp(logvar_out).cpu().numpy()

            for j, (y, x) in enumerate(batch_tiles):
                w = window[None, :, :]
                accum[:, y:y + patch_size, x:x + patch_size] += mean01[j] * w
                accum_var[:, y:y + patch_size, x:x + patch_size] += var01[j] * w
                accum_w2[:, y:y + patch_size, x:x + patch_size] += w ** 2
                weight_sum[:, y:y + patch_size, x:x + patch_size] += w

            if (i // batch_size + 1) % max(1, len(tiles) // (5 * batch_size)) == 0:
                pct = 100.0 * min(i + batch_size, len(tiles)) / len(tiles)
                print(f"    Progress: {min(i + batch_size, len(tiles))}/{len(tiles)} tiles ({pct:.1f}%) blended with 2D Hann windowing...")

    ws = np.clip(weight_sum, 1e-6, None)
    mean_recon = accum / ws
    var_recon = (accum_var / ws) * (accum_w2 / (ws ** 2))
    std_recon = np.sqrt(np.clip(var_recon, 0, None))

    return mean_recon[:, :H, :W], std_recon[:, :H, :W]


def save_outputs(mean_recon, std_recon, out_prefix, reference_tif=None):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.save(f"{out_prefix}_mean.npy", mean_recon)
    np.save(f"{out_prefix}_std.npy", std_recon)
    print(f"Saved: {out_prefix}_mean.npy, {out_prefix}_std.npy")

    try:
        from PIL import Image
        display = mean_recon[[2, 1, 0]] if mean_recon.shape[0] >= 3 else mean_recon
        display = np.clip(display.transpose(1, 2, 0), 0, 1)
        Image.fromarray((display * 255).astype(np.uint8)).save(f"{out_prefix}_preview.png")
        print(f"Saved: {out_prefix}_preview.png (NIR-R-G false-color preview)")
    except Exception as e:
        print(f"(skipped PNG preview: {e})")

    if reference_tif is not None:
        if not HAS_RASTERIO:
            print("rasterio not installed - skipping GeoTIFF export.")
            return
        with rasterio.open(reference_tif) as ref:
            profile = ref.profile.copy()
            profile.update(count=mean_recon.shape[0], dtype="float32")
            with rasterio.open(f"{out_prefix}.tif", "w", **profile) as dst:
                dst.write(mean_recon.astype(np.float32))
        print(f"Saved georeferenced GeoTIFF: {out_prefix}.tif")


def main():
    p = argparse.ArgumentParser(description="Full-Scene Sliding Window Inference Engine")
    p.add_argument("--opt", required=True, help=".npy (C,H,W) cloudy optical scene, reflectance [0,1]")
    p.add_argument("--sar", required=True, help=".npy (C,H,W) co-registered SAR scene")
    p.add_argument("--temporal", required=True, help=".npy (C,H,W) co-registered temporal reference scene")
    p.add_argument("--dem", required=True, help=".npy (4,H,W) co-registered DEM [Elev,Slope,sin(Asp),cos(Asp)]")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="outputs/reconstructed_scene")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--reference_tif", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    gen = CloudReconstructionGeneratorV2(base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_key = "gen_ema_state" if "gen_ema_state" in ckpt else "gen_state"
    gen.load_state_dict(ckpt[state_key])
    print(f"Loaded '{state_key}' from checkpoint (epoch {ckpt.get('epoch', '?')})", flush=True)

    opt_full = np.load(args.opt).astype(np.float32)
    sar_full = np.load(args.sar).astype(np.float32)
    temporal_full = np.load(args.temporal).astype(np.float32)
    dem_full = np.load(args.dem).astype(np.float32)

    print(f"\nScene Dimensions: {opt_full.shape[1]}x{opt_full.shape[2]} px, tiling at {args.patch_size}px with {args.overlap}px overlap...", flush=True)
    mean_recon, std_recon = sliding_window_infer(
        gen, opt_full, sar_full, temporal_full, dem_full, device,
        patch_size=args.patch_size, overlap=args.overlap, batch_size=args.batch_size,
    )
    
    save_outputs(mean_recon, std_recon, args.out, reference_tif=args.reference_tif)

    # Compute Regime B metrics on full reconstructed scene
    print("\n" + "=" * 80)
    print("=== FULL-SCENE REGIME B EVALUATION METRICS ===")
    print("=" * 80)
    opt_t = torch.from_numpy(mean_recon).unsqueeze(0).to(device)
    sar_t = torch.from_numpy(sar_full).unsqueeze(0).to(device)
    regB = evaluate_no_reference(opt_t, sar_t)
    print(f"  Laplacian Sharpness Score     : {regB['laplacian_sharpness_score']:.2f}")
    print(f"  SAR-Optical Grad Correlation  : {regB['sar_gradient_correlation']:.3f}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
