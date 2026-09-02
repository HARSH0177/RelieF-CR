"""
Phase 6: Full-Scene Raster Assembly and Synthetic Cloud Injection.

Assembles full-resolution co-registered rasters (.npy arrays) for Scene 02 (and Scene 04)
covering a large continuous multi-modal extent (e.g. 2048x2048 px @ 5m resolution = 10.24km x 10.24km).
Applies seamless global synthetic cloud injection with directionally-offset shadows.
"""
import os
import sys
import glob
import math
import argparse
import numpy as np
import scipy.ndimage

try:
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window
    from rasterio.warp import Resampling
except ImportError:
    print("rasterio is required.")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_scenes import parse_liss4_metadata, sar_dn_to_scaled_db_patch


def horns_method_full(elevation_pad: np.ndarray, global_min: float, global_max: float, cell_size: float = 5.0):
    e_denom = (global_max - global_min) if (global_max - global_min) > 1e-6 else 1.0
    elev_clean = np.where(elevation_pad > -1000.0, elevation_pad, global_min)
    elev_norm = np.clip((elev_clean - global_min) / e_denom, 0.0, 1.0).astype(np.float32)
    
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / (8.0 * cell_size)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / (8.0 * cell_size)
    
    dz_dx = scipy.ndimage.convolve(elev_clean, kernel_x, mode='nearest')
    dz_dy = scipy.ndimage.convolve(elev_clean, kernel_y, mode='nearest')
    
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dy, dz_dx)
    
    slope_norm = np.clip(slope / (np.pi / 2.0), 0.0, 1.0).astype(np.float32)
    sin_aspect = np.sin(aspect).astype(np.float32)
    cos_aspect = np.cos(aspect).astype(np.float32)
    
    # Strip 1-pixel boundary
    return elev_norm[1:-1, 1:-1], slope_norm[1:-1, 1:-1], sin_aspect[1:-1, 1:-1], cos_aspect[1:-1, 1:-1]


def generate_full_scene_cloud_mask(h: int, w: int, severity: float = 0.65, sun_azimuth: float = 145.0, seed: int = 101):
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=np.float32)
    octaves = [8, 16, 32, 64, 128]
    for octv in octaves:
        coarse = rng.random((octv, octv)).astype(np.float32)
        smooth = scipy.ndimage.zoom(coarse, (h / octv, w / octv), order=1)[:h, :w]
        mask += smooth / len(octaves)
        
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    max_alpha = float(rng.uniform(0.75, 1.00))
    thresh = 0.65 - 0.35 * severity
    soft_mask = np.clip((mask - thresh) / 0.35, 0.0, 1.0).astype(np.float32)
    soft_mask = scipy.ndimage.gaussian_filter(soft_mask, sigma=4.0) * max_alpha
    soft_mask = np.clip(soft_mask, 0.0, 1.0).astype(np.float32)
    
    # Shadow generation
    rad = np.deg2rad(sun_azimuth)
    offset_px = float(rng.uniform(12.0, 30.0) * severity)
    dy = int(round(offset_px * np.cos(rad)))
    dx = int(round(offset_px * np.sin(rad)))
    shadow_shifted = scipy.ndimage.shift(soft_mask, shift=(dy, dx), order=1, mode='constant', cval=0.0)
    shadow_opacity = np.clip(shadow_shifted - soft_mask, 0.0, 1.0)
    shadow_opacity = scipy.ndimage.gaussian_filter(shadow_opacity, sigma=3.0)
    shadow_opacity = np.clip(shadow_opacity, 0.0, 1.0).astype(np.float32)
    
    base_brightness = float(rng.uniform(0.85, 0.95))
    raw_noise = rng.uniform(-0.05, 0.05, size=(3, h, w)).astype(np.float32)
    texture_noise = scipy.ndimage.gaussian_filter(raw_noise, sigma=(0, 4.0, 4.0))
    cloud_color = np.clip(base_brightness + texture_noise, 0.70, 0.98).astype(np.float32)
    
    mask_class = np.zeros((1, h, w), dtype=np.int64)
    mask_class[0, soft_mask > 0.40] = 1
    mask_class[0, (shadow_opacity > 0.20) & (soft_mask <= 0.40)] = 2
    
    return soft_mask, shadow_opacity, cloud_color, mask_class


def assemble_scene_raster(
    scene_dir: str,
    dem_file: str,
    sar_vv_file: str,
    sar_vh_file: str,
    s2_b03_file: str,
    out_dir: str,
    scene_name: str = "scene_02",
    roi_size: int = 2048,
    seed: int = 42
):
    os.makedirs(out_dir, exist_ok=True)
    b2_file = glob.glob(os.path.join(scene_dir, "**/*BAND2*.tif"), recursive=True)[0]
    b3_file = glob.glob(os.path.join(scene_dir, "**/*BAND3*.tif"), recursive=True)[0]
    b4_file = glob.glob(os.path.join(scene_dir, "**/*BAND4*.tif"), recursive=True)[0]

    scene_meta = parse_liss4_metadata(scene_dir)
    print(f"Assembling Full-Scene Raster for {scene_name} ({os.path.basename(scene_dir)})...")
    print(f"  Radiometric Calibration: Date={scene_meta['date']}, Elev={scene_meta['sun_elev']:.2f}°, Azim={scene_meta['sun_azim']:.2f}°")
    print(f"  Gains: B2={scene_meta['gain'][0]:.6f}, B3={scene_meta['gain'][1]:.6f}, B4={scene_meta['gain'][2]:.6f}")

    with rasterio.open(b2_file) as master_src:
        target_crs = master_src.crs
        target_transform = master_src.transform
        width, height = master_src.width, master_src.height

    # Choose central ROI where optical, SAR, temporal, and DEM are 100% valid
    x0 = max(0, (width - roi_size) // 2)
    y0 = max(0, (height - roi_size) // 2)
    win = Window(x0, y0, roi_size, roi_size)
    print(f"  Full Raster Size: {height}x{width} px | Extracted ROI: ({y0}:{y0+roi_size}, {x0}:{x0+roi_size}) @ 5m resolution ({roi_size*5/1000:.1f}km x {roi_size*5/1000:.1f}km)")

    # Read Optical Bands
    s_b2 = rasterio.open(b2_file)
    s_b3 = rasterio.open(b3_file)
    s_b4 = rasterio.open(b4_file)

    g_raw = s_b2.read(1, window=win).astype(np.float32)
    r_raw = s_b3.read(1, window=win).astype(np.float32)
    nir_raw = s_b4.read(1, window=win).astype(np.float32)

    opt_raw = np.stack([g_raw, r_raw, nir_raw], axis=0)
    opt_clean = np.clip(opt_raw * scene_meta["gain"][:, None, None] + scene_meta["bias"][:, None, None], 0.0, 1.0).astype(np.float32)

    # Read SAR VRT
    svv_src = rasterio.open(sar_vv_file)
    gcps_list, gcps_crs = svv_src.gcps if (svv_src.gcps and len(svv_src.gcps[0]) > 0) else (None, None)
    sar_vv_vrt = WarpedVRT(svv_src, crs=target_crs, transform=target_transform, width=width, height=height, src_crs=gcps_crs or svv_src.crs or 'EPSG:4326', src_gcps=gcps_list, resampling=Resampling.bilinear)
    sar_vv_dn = sar_vv_vrt.read(1, window=win).astype(np.float32)

    svh_src = rasterio.open(sar_vh_file)
    gcps_list_vh, gcps_crs_vh = svh_src.gcps if (svh_src.gcps and len(svh_src.gcps[0]) > 0) else (None, None)
    sar_vh_vrt = WarpedVRT(svh_src, crs=target_crs, transform=target_transform, width=width, height=height, src_crs=gcps_crs_vh or svh_src.crs or 'EPSG:4326', src_gcps=gcps_list_vh, resampling=Resampling.bilinear)
    sar_vh_dn = sar_vh_vrt.read(1, window=win).astype(np.float32)
    sar_2ch = sar_dn_to_scaled_db_patch(sar_vv_dn, sar_vh_dn)

    # Read Temporal VRT (Sentinel-2)
    s2_b04 = s2_b03_file.replace("B03", "B04")
    s2_b08 = s2_b03_file.replace("B03", "B08")
    s2_vrts = [WarpedVRT(rasterio.open(p), crs=target_crs, transform=target_transform, width=width, height=height, resampling=Resampling.bilinear) for p in [s2_b03_file, s2_b04, s2_b08]]
    s2_raw = np.stack([vrt.read(1, window=win).astype(np.float32) for vrt in s2_vrts], axis=0)
    temporal_full = np.clip((s2_raw - 1000.0) / 10000.0, 0.0, 1.0).astype(np.float32)

    # Read DEM with Horn's method
    dem_src = rasterio.open(dem_file)
    dem_vrt = WarpedVRT(dem_src, crs=target_crs, transform=target_transform, width=width, height=height, resampling=Resampling.bilinear)
    dem_sample = dem_vrt.read(1, out_shape=(100, 100)).astype(np.float32)
    valid_dem = dem_sample[dem_sample > -1000.0]
    global_min = float(valid_dem.min()) if len(valid_dem) > 0 else 0.0
    global_max = float(valid_dem.max()) if len(valid_dem) > 0 else 1000.0

    pad_win = Window(max(0, x0 - 1), max(0, y0 - 1), roi_size + 2, roi_size + 2)
    dem_pad = dem_vrt.read(1, window=pad_win).astype(np.float32)
    e_norm, s_norm, sin_a, cos_a = horns_method_full(dem_pad, global_min, global_max, cell_size=5.0)
    if e_norm.shape[0] > roi_size or e_norm.shape[1] > roi_size:
        e_norm = e_norm[:roi_size, :roi_size]
        s_norm = s_norm[:roi_size, :roi_size]
        sin_a = sin_a[:roi_size, :roi_size]
        cos_a = cos_a[:roi_size, :roi_size]
    dem_4ch = np.stack([e_norm, s_norm, sin_a, cos_a], axis=0).astype(np.float32)

    # Global Seamless Synthetic Cloud & Shadow Generation
    soft_mask, shadow_opacity, cloud_color, mask_class = generate_full_scene_cloud_mask(
        roi_size, roi_size, severity=0.65, sun_azimuth=scene_meta["sun_azim"], seed=seed
    )
    shadow_darkness = 0.55
    opt_cloudy = opt_clean * (1.0 - shadow_opacity[None, :, :] * shadow_darkness)
    opt_cloudy = opt_cloudy * (1.0 - soft_mask[None, :, :]) + cloud_color * soft_mask[None, :, :]
    opt_cloudy = np.clip(opt_cloudy, 0.0, 1.0).astype(np.float32)
    mask_cont = soft_mask[None, :, :].astype(np.float32)

    # Save to disk
    paths = {
        "opt_clean": os.path.join(out_dir, f"{scene_name}_opt_clean.npy"),
        "opt_cloudy": os.path.join(out_dir, f"{scene_name}_opt_cloudy.npy"),
        "sar": os.path.join(out_dir, f"{scene_name}_sar.npy"),
        "temporal": os.path.join(out_dir, f"{scene_name}_temporal.npy"),
        "dem": os.path.join(out_dir, f"{scene_name}_dem.npy"),
        "mask": os.path.join(out_dir, f"{scene_name}_mask.npy"),
        "mask_class": os.path.join(out_dir, f"{scene_name}_mask_class.npy")
    }
    np.save(paths["opt_clean"], opt_clean)
    np.save(paths["opt_cloudy"], opt_cloudy)
    np.save(paths["sar"], sar_2ch)
    np.save(paths["temporal"], temporal_full)
    np.save(paths["dem"], dem_4ch)
    np.save(paths["mask"], mask_cont)
    np.save(paths["mask_class"], mask_class)

    print(f"Successfully assembled and saved full-scene rasters to {out_dir}:")
    for k, p in paths.items():
        arr = np.load(p)
        print(f"  {k:12s}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

    s_b2.close()
    s_b3.close()
    s_b4.close()
    svv_src.close()
    svh_src.close()
    dem_src.close()
    sar_vv_vrt.close()
    sar_vh_vrt.close()
    dem_vrt.close()
    for vrt in s2_vrts:
        vrt.close()

    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--temp_dir", default=r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\dataset_root\_temp_extracted")
    p.add_argument("--out_dir", default=r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\outputs\full_scene_02")
    p.add_argument("--roi_size", type=int, default=2048)
    args = p.parse_args()

    scene_dirs = sorted(glob.glob(os.path.join(args.temp_dir, "LISS-IV", "*15FEB2023*")))
    if not scene_dirs:
        scene_dirs = sorted(glob.glob(os.path.join(args.temp_dir, "LISS-IV", "*")))
    scene_dir = scene_dirs[0]

    dem_files = glob.glob(os.path.join(args.temp_dir, "DEM", "**/*.tif"), recursive=True)
    dem_file = [f for f in dem_files if "DEM" in f][0]

    sar_vvs = glob.glob(os.path.join(args.temp_dir, "Sentinel-1", "**/*20230211*vv*.tiff"), recursive=True)
    if not sar_vvs:
        sar_vvs = glob.glob(os.path.join(args.temp_dir, "Sentinel-1", "**/*vv*.tiff"), recursive=True)
    sar_vv = sar_vvs[0]

    sar_vhs = glob.glob(os.path.join(args.temp_dir, "Sentinel-1", "**/*20230211*vh*.tiff"), recursive=True)
    if not sar_vhs:
        sar_vhs = glob.glob(os.path.join(args.temp_dir, "Sentinel-1", "**/*vh*.tiff"), recursive=True)
    sar_vh = sar_vhs[0]

    s2_b03s = glob.glob(os.path.join(args.temp_dir, "Sentinel-2", "**/*B03*10m*.jp2"), recursive=True)
    s2_b03 = s2_b03s[0]

    assemble_scene_raster(
        scene_dir=scene_dir,
        dem_file=dem_file,
        sar_vv_file=sar_vv,
        sar_vh_file=sar_vh,
        s2_b03_file=s2_b03,
        out_dir=args.out_dir,
        scene_name="scene_02",
        roi_size=args.roi_size
    )
