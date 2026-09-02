#!/usr/bin/env python3
import os
import sys
import glob
import zipfile
import subprocess
import argparse
import time

def resolve_dataset_root(custom_root=None):
    """
    Intelligently finds or prepares the dataset root.
    Handles:
      1. Pre-extracted directory in /kaggle/input (e.g. /kaggle/input/.../train)
      2. Zip archives in /kaggle/input (train.zip, val.zip, test.zip)
      3. Custom specified --data_root
    """
    if custom_root and os.path.isdir(os.path.join(custom_root, "train")):
        print(f"Using explicitly specified data_root: {custom_root}")
        return custom_root

    search_roots = ["/kaggle/input", "/kaggle/working", ".", "./dataset_archives"]
    
    # Check 1: Look for dataset root containing 'train'
    for s_root in search_roots:
        if os.path.isdir(s_root):
            for root, dirs, _ in os.walk(s_root):
                if "train" in dirs:
                    train_p = os.path.join(root, "train")
                    # Check direct or nested modality directory
                    if os.path.isdir(train_p) and (
                        any(os.path.isdir(os.path.join(train_p, m)) for m in ["opt_clean", "opt_cloudy", "sar"]) or
                        os.path.isdir(os.path.join(train_p, "train"))
                    ):
                        print(f"Found ready-to-use extracted dataset at: {root}")
                        return root

    # Check 2: Are there ZIP archives to extract?
    zips = {"train": None, "val": None, "test": None}
    for s_root in search_roots:
        if os.path.isdir(s_root):
            for zf in glob.glob(os.path.join(s_root, "**", "*.zip"), recursive=True):
                fname = os.path.basename(zf).lower()
                for split in ["train", "val", "test"]:
                    if split in fname and zips[split] is None:
                        zips[split] = zf
                        print(f"Found {split} archive: {zf} ({os.path.getsize(zf)/(1024**3):.2f} GB)")

    target_temp = "/kaggle/temp/dataset_root" if os.path.exists("/kaggle") else "dataset_root"
    os.makedirs(target_temp, exist_ok=True)

    for split, zip_path in zips.items():
        if zip_path and os.path.exists(zip_path):
            dest_dir = os.path.join(target_temp, split)
            if not os.path.exists(dest_dir) or len(os.listdir(dest_dir)) == 0:
                print(f"Extracting {split} from {zip_path} to {target_temp}...")
                t0 = time.time()
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_temp)
                print(f"  -> Extracted '{split}' in {time.time()-t0:.1f}s")
            else:
                print(f"Split '{split}' already extracted in {dest_dir}.")

    if os.path.isdir(os.path.join(target_temp, "train")):
        return target_temp

    # If still not found, print detailed diagnostic of /kaggle/input
    print("\n" + "!" * 80)
    print("DEBUG: Could not locate 'train' split. Available directories and files:")
    for s_root in ["/kaggle/input", "."]:
        if os.path.isdir(s_root):
            for root, dirs, files in os.walk(s_root):
                depth = root.count(os.sep)
                if depth < 4:
                    print(f"  Directory: {root}")
                    if files:
                        print(f"    Files: {files[:5]} (total {len(files)})")
    print("!" * 80 + "\n")
    return target_temp


def main():
    parser = argparse.ArgumentParser(description="Kaggle RelieF-CR Master Runner")
    parser.add_argument("--data_root", type=str, default=None, help="Path to extracted dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size (Total batch = batch_size * num_gpus)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working", help="Output directory for checkpoints and metrics")
    args = parser.parse_args()

    import torch
    num_gpus = torch.cuda.device_count()
    print("=" * 80)
    print("=== RelieF-CR (CloudFree Vision v2) — KAGGLE TRAINING PIPELINE ===")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    print(f"Detected GPU Count: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU [{i}]: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
    print("=" * 80)

    target_data_root = resolve_dataset_root(args.data_root)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print("\n[2/4] Launching Training...")
    if num_gpus >= 2:
        print(f"Launching Multi-GPU DDP with torchrun on {num_gpus} GPUs (Effective Batch Size: {args.batch_size * num_gpus})...")
        train_cmd = [
            sys.executable, "-m", "torchrun",
            f"--nproc_per_node={num_gpus}",
            "scripts/train_generator_ddp.py",
            "--data_root", target_data_root,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr_g", str(args.lr),
            "--lr_d", str(args.lr),
            "--num_workers", str(args.num_workers),
            "--checkpoint_dir", ckpt_dir
        ]
    else:
        print(f"Launching Single-GPU / CPU training with batch size {args.batch_size}...")
        train_cmd = [
            sys.executable, "scripts/train_generator.py",
            "--data_root", target_data_root,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr_g", str(args.lr),
            "--lr_d", str(args.lr),
            "--num_workers", str(args.num_workers),
            "--checkpoint_dir", ckpt_dir
        ]

    print("Running command:", " ".join(train_cmd))
    res = subprocess.run(train_cmd)
    if res.returncode != 0:
        print(f"ERROR: Training failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    best_ckpt = os.path.join(ckpt_dir, "generator_best.pt")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(ckpt_dir, "generator_latest.pt")

    print("\n[3/4] Running Full Test Set Benchmark Evaluation...")
    eval_cmd = [
        sys.executable, "scripts/evaluate.py",
        "--data_root", target_data_root,
        "--split", "test",
        "--checkpoint", best_ckpt,
        "--batch_size", "16",
        "--out_metrics", os.path.join(args.output_dir, "test_benchmark_metrics.json")
    ]
    subprocess.run(eval_cmd)

    print("\n[4/4] Generating Publication Comparison Grid...")
    grid_cmd = [
        sys.executable, "scripts/make_comparison_grid.py",
        "--data_root", target_data_root,
        "--checkpoint", best_ckpt,
        "--n_samples", "4",
        "--out", os.path.join(args.output_dir, "comparison_grid_trained.png")
    ]
    subprocess.run(grid_cmd)

    print("\n" + "=" * 80)
    print("=== KAGGLE PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")
    print(f"Trained Checkpoints : {ckpt_dir}")
    print(f"Evaluation Metrics  : {os.path.join(args.output_dir, 'test_benchmark_metrics.json')}")
    print(f"Comparison Grid     : {os.path.join(args.output_dir, 'comparison_grid_trained.png')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
