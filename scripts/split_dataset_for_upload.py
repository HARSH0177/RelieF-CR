#!/usr/bin/env python3
"""
Split CloudFree Vision v2 Dataset into Scene-by-Scene Zip Archives for Kaggle Upload

Walks dataset_root/{train,val,test}/{opt_cloudy,opt_clean,sar,temporal,dem,mask,mask_class}/
and packages all modalities per scene into a dedicated zip archive:
  - scene_00.zip
  - scene_01.zip
  - scene_02.zip
  - scene_03.zip
  - scene_04.zip
  - scene_05.zip
  - scene_06.zip

Each zip preserves the full relative directory structure:
  train/opt_cloudy/scene_00_p0000.npy
  train/opt_clean/scene_00_p0000.npy
  ...
so extracting all 7 zips reconstructs the canonical dataset_root layout seamlessly.
"""

import os
import sys
import glob
import time
import zipfile
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple


MODALITIES = ["opt_cloudy", "opt_clean", "sar", "temporal", "dem", "mask", "mask_class"]
SPLITS = ["train", "val", "test"]


def scan_dataset(dataset_root: str) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """
    Scans dataset_root and groups files by scene prefix.
    Returns:
      scene_files[scene_prefix][split] = [(abs_path, rel_path), ...]
    """
    scene_files = defaultdict(lambda: defaultdict(list))
    
    for split in SPLITS:
        for mod in MODALITIES:
            mod_dir = os.path.join(dataset_root, split, mod)
            if not os.path.isdir(mod_dir):
                print(f"WARNING: Missing modality directory: {mod_dir}")
                continue
            
            for file_path in glob.glob(os.path.join(mod_dir, "*.npy")):
                fname = os.path.basename(file_path)
                # Scene prefix matching logic: scene_00, scene_01, etc.
                parts = fname.split("_")
                if len(parts) >= 2 and parts[0] == "scene":
                    scene_prefix = f"{parts[0]}_{parts[1]}"
                else:
                    scene_prefix = "unknown"
                
                rel_path = os.path.relpath(file_path, dataset_root)
                scene_files[scene_prefix][split].append((file_path, rel_path))
                
    return scene_files


def create_scene_zip(
    scene_prefix: str,
    split_dict: Dict[str, List[Tuple[str, str]]],
    output_dir: str,
    compression: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 1
) -> Tuple[str, int, int, float, float]:
    """
    Creates a zip file for a single scene.
    Returns:
      (zip_filepath, patch_count, file_count, size_gb, elapsed_seconds)
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_filename = f"{scene_prefix}.zip"
    zip_filepath = os.path.join(output_dir, zip_filename)
    
    # Remove existing zip if present to avoid appending duplicates
    if os.path.exists(zip_filepath):
        try:
            os.remove(zip_filepath)
        except OSError as e:
            print(f"Error removing existing zip {zip_filepath}: {e}")

    all_items = []
    for split in SPLITS:
        all_items.extend(split_dict.get(split, []))
    
    total_files = len(all_items)
    patch_count = total_files // len(MODALITIES) if total_files % len(MODALITIES) == 0 else total_files / len(MODALITIES)
    
    print(f"\n--- Zipping {scene_prefix} ({total_files:,} files, ~{patch_count} patches across {len(MODALITIES)} modalities) ---", flush=True)
    t0 = time.time()
    
    with zipfile.ZipFile(zip_filepath, 'w', compression=compression, compresslevel=compresslevel) as zf:
        for idx, (abs_path, rel_path) in enumerate(all_items, 1):
            zf.write(abs_path, rel_path)
            if idx % 2000 == 0 or idx == total_files:
                pct = (idx / total_files) * 100
                print(f"  [{scene_prefix}] {idx:,}/{total_files:,} files zipped ({pct:5.1f}%)...", flush=True)
                
    elapsed = time.time() - t0
    size_bytes = os.path.getsize(zip_filepath)
    size_gb = size_bytes / (1024 ** 3)
    
    print(f"  --> Completed {zip_filename}: {size_gb:.2f} GB in {elapsed:.1f}s ({size_bytes:,} bytes)", flush=True)
    return zip_filepath, int(patch_count), total_files, size_gb, elapsed


def verify_zip_integrity(zip_filepath: str, expected_file_count: int) -> bool:
    """Verifies that the zip can be opened and has the exact expected file count."""
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            namelist = zf.namelist()
            if len(namelist) != expected_file_count:
                print(f"INTEGRITY ERROR: {zip_filepath} has {len(namelist)} files, expected {expected_file_count}")
                return False
            bad_file = zf.testzip()
            if bad_file is not None:
                print(f"INTEGRITY ERROR: Corrupted file detected in {zip_filepath}: {bad_file}")
                return False
        return True
    except Exception as e:
        print(f"INTEGRITY ERROR reading {zip_filepath}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Split Dataset into Per-Scene Zips for Kaggle Upload")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\dataset_root",
        help="Path to dataset_root containing train/val/test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\HARSH AMBULE\Downloads\Kaggle_Upload",
        help="Directory to save scene zip files"
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional specific scenes to zip (e.g. scene_00 scene_01)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use ZIP_STORED (no compression) instead of ZIP_DEFLATED level 1"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CLOUDFREE VISION V2 — DATASET SPLITTER FOR KAGGLE UPLOAD")
    print("=" * 80)
    print(f"Source Dataset Root : {args.dataset_root}")
    print(f"Target Output Dir   : {args.output_dir}")
    print(f"Modalities Monitored: {MODALITIES}")
    print(f"Splits Monitored    : {SPLITS}")
    
    if not os.path.isdir(args.dataset_root):
        print(f"ERROR: Dataset root directory '{args.dataset_root}' does not exist!")
        sys.exit(1)

    print("\nScanning dataset_root for scene partitions...")
    scene_files = scan_dataset(args.dataset_root)
    discovered_scenes = sorted(scene_files.keys())
    print(f"Discovered {len(discovered_scenes)} distinct scene partitions: {discovered_scenes}")

    if args.scenes:
        target_scenes = [s for s in args.scenes if s in scene_files]
    else:
        target_scenes = [s for s in discovered_scenes if s.startswith("scene_")]

    print(f"Processing target scenes: {target_scenes}\n")

    compression = zipfile.ZIP_STORED if args.fast else zipfile.ZIP_DEFLATED
    compresslevel = 1 if not args.fast else None

    results = []
    total_patches_all = 0
    total_files_all = 0
    total_size_gb_all = 0.0

    for scene in target_scenes:
        split_dict = scene_files[scene]
        zip_path, patch_cnt, file_cnt, size_gb, elapsed = create_scene_zip(
            scene, split_dict, args.output_dir, compression=compression, compresslevel=compresslevel
        )
        
        is_valid = verify_zip_integrity(zip_path, file_cnt)
        if not is_valid:
            print(f"FATAL: Integrity check failed for {zip_path}! Aborting.")
            sys.exit(1)
            
        results.append({
            "scene": scene,
            "patches": patch_cnt,
            "files": file_cnt,
            "size_gb": size_gb,
            "path": zip_path,
            "time_s": elapsed,
            "status": "VALID (PASS)" if is_valid else "INVALID (FAIL)"
        })
        total_patches_all += patch_cnt
        total_files_all += file_cnt
        total_size_gb_all += size_gb

    print("\n" + "=" * 90)
    print("SCENE-BY-SCENE ZIP ARCHIVE GENERATION SUMMARY")
    print("=" * 90)
    print(f"{'Scene':<10} | {'Patches':>8} | {'Files (7x)':>10} | {'Size (GB)':>10} | {'Time (s)':>8} | {'Integrity':<14} | {'Path'}")
    print("-" * 90)
    for r in results:
        print(f"{r['scene']:<10} | {r['patches']:>8,d} | {r['files']:>10,d} | {r['size_gb']:>9.2f} GB | {r['time_s']:>7.1f}s | {r['status']:<14} | {r['path']}")
    print("-" * 90)
    print(f"{'TOTAL':<10} | {total_patches_all:>8,d} | {total_files_all:>10,d} | {total_size_gb_all:>9.2f} GB | {'':>8} | {'ALL PASS':<14} |")
    print("=" * 90)


if __name__ == "__main__":
    main()
