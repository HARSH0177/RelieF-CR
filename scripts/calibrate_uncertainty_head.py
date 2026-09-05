import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))

from models.generator import CloudReconstructionGeneratorV2
from data.dataset import CloudReconstructionDataset


def beta_nll_loss(pred_mean, target, logvar, beta=0.5):
    """
    Beta-NLL Loss (Seitzer et al., ICLR 2022: 'On the Pitfalls of Heteroscedastic Uncertainty Estimation')
    Prevents loss saturation and variance collapse to the clamp floor.
    """
    var = torch.exp(logvar)
    sq_err = (pred_mean - target) ** 2
    
    # Beta-weighting factor detaches variance power to prevent gradient starvation
    if beta > 0.0:
        beta_weight = (var.detach()) ** beta
    else:
        beta_weight = 1.0
        
    nll = 0.5 * (beta_weight * (sq_err / (var + 1e-8) + logvar))
    return nll.mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"STEP 2: FAST 'HEAD-ONLY' UNCERTAINTY CALIBRATION (Device: {device})")
    print("=" * 80)
    
    data_root = os.path.join(PROJECT_ROOT, "dataset_root")
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "generator_best.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
        
    ckpt = torch.load(ckpt_path, map_location=device)
    gen = CloudReconstructionGeneratorV2(base_ch=48).to(device)
    state_key = "gen_ema_state" if "gen_ema_state" in ckpt else "gen_state"
    gen.load_state_dict(ckpt[state_key])
    
    # Freeze ALL parameters EXCEPT logvar_head
    frozen_params = 0
    trainable_params = 0
    for name, param in gen.named_parameters():
        if "logvar_head" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
            
    print(f"Frozen backbone parameters: {frozen_params:,}")
    print(f"Trainable logvar_head parameters: {trainable_params:,}")
    
    optimizer = torch.optim.Adam(gen.logvar_head.parameters(), lr=1e-3, weight_decay=1e-5)
    
    ds_train = CloudReconstructionDataset(data_root, split="train")
    ds_val = CloudReconstructionDataset(data_root, split="val")
    
    if device.type == "cpu":
        ds_train.patch_ids = ds_train.patch_ids[:1000]
        ds_val.patch_ids = ds_val.patch_ids[:200]
    
    # Batch size 32 for fast execution
    loader_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"\nTraining set: {len(ds_train)} patches | Val set: {len(ds_val)} patches", flush=True)
    print("Starting 3-epoch beta-NLL calibration pass (beta=0.5)...", flush=True)
    
    t0 = time.time()
    epochs = 3
    
    for epoch in range(1, epochs + 1):
        gen.eval()  # Keep BN/IN stats fixed
        gen.logvar_head.train()
        
        train_loss = 0.0
        n_batches = 0
        
        for batch in loader_train:
            opt_cloudy = batch["opt_cloudy"].to(device)
            opt_clean = batch["opt_clean"].to(device)
            sar = batch["sar"].to(device)
            temporal = batch["temporal"].to(device)
            dem = batch["dem"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            
            optimizer.zero_grad()
            mean_fake, logvar_fake = gen(opt_cloudy, sar, temporal, dem)
            
            # Compute beta-NLL loss over masked region
            loss = beta_nll_loss((mean_fake + 1.0) / 2.0, (opt_clean + 1.0) / 2.0, logvar_fake, beta=0.5)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
            
        avg_train_loss = train_loss / max(n_batches, 1)
        
        # Validation pass
        gen.eval()
        val_stds = []
        with torch.no_grad():
            for batch in loader_val:
                opt_cloudy = batch["opt_cloudy"].to(device)
                sar = batch["sar"].to(device)
                temporal = batch["temporal"].to(device)
                dem = batch["dem"].to(device)
                
                _, logvar_fake = gen(opt_cloudy, sar, temporal, dem)
                std_fake = torch.exp(0.5 * logvar_fake).mean(dim=1)
                val_stds.append(std_fake.cpu().numpy().flatten())
                
        all_val_stds = np.concatenate(val_stds)
        p5, p50, p95 = np.percentile(all_val_stds, [5, 50, 95])
        pegged_pct = np.mean(np.isclose(all_val_stds, np.exp(-3.0), atol=1e-4)) * 100.0
        
        print(f"Epoch [{epoch}/{epochs}] | Loss: {avg_train_loss:.6f} | Val std percentiles: p5={p5:.4f}, p50={p50:.4f}, p95={p95:.4f} | Pegged={pegged_pct:.1f}%", flush=True)

    elapsed = time.time() - t0
    print(f"\nCalibration complete in {elapsed:.1f}s!", flush=True)
    
    # Save updated checkpoint
    ckpt[state_key] = gen.state_dict()
    calib_ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "generator_best.pt")
    backup_ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "generator_uncertainty_calibrated.pt")
    
    torch.save(ckpt, calib_ckpt_path)
    torch.save(ckpt, backup_ckpt_path)
    print(f"Saved calibrated model state to:\n  - {calib_ckpt_path}\n  - {backup_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
