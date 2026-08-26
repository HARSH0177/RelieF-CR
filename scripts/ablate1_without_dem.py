"""
Ablation Study Variant 1: Training WITHOUT CartoDEM Topography (Elevation, Slope, Aspect).
Evaluates the contribution of the 4-channel DEM descriptor to multi-modal cloud removal.

Output checkpoint: checkpoints/ablate1_no_dem_best.pt
"""
import os
import sys
import copy
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from models.generator import CloudReconstructionGeneratorV2
from models.discriminator import MultiScaleDiscriminator
from models.losses import CombinedGeneratorLoss, AdversarialLoss
from data.dataset import CloudReconstructionDataset, SyntheticCloudDataset
from scripts.evaluate import run_evaluation


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.mul_(self.decay).add_(m_p, alpha=1 - self.decay)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)


class GeneratorNoDEM(CloudReconstructionGeneratorV2):
    """Generator variant where DEM features are zeroed out before fusion."""
    def forward(self, opt, sar, temp, dem):
        opt_full = self.opt_stem(opt)
        sar_full = self.sar_stem(sar)
        temp_full = self.temp_stem(temp)
        
        opt_h = self.opt_down(opt_full)
        sar_h = self.sar_down(sar_full)
        temp_h = self.temp_down(temp_full)

        sar_h = self.sar_stn(sar_h, opt_h)
        temp_h = self.temp_stn(temp_h, opt_h)

        aux_h = self.aux_proj(torch.cat([sar_h, temp_h], dim=1))
        fused_h, reliance_on_aux = self.cross_attn(opt_h, aux_h)
        self._last_reliance_on_aux = reliance_on_aux

        # DEM branch zeroed out (Ablation 1)
        dem_channels = self.fuse[0].in_channels - fused_h.size(1)
        dem_h_zero = torch.zeros(fused_h.size(0), dem_channels, 
                                 fused_h.size(2), fused_h.size(3), device=fused_h.device)
        f0 = self.fuse(torch.cat([fused_h, dem_h_zero], dim=1))

        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f2 = self.self_attn(f2)
        f2 = self.bottleneck_res(f2)

        d1 = self.up1(f2, f1)
        d0 = self.up2(d1, f0)
        d_full = self.final_up(d0, opt_full)

        mean_out = self.mean_head(d_full)
        logvar_out = self.logvar_head(d_full).clamp(-6.0, 6.0)
        return mean_out, logvar_out


def main():
    parser = argparse.ArgumentParser(description="Ablation 1: Train without 4-channel DEM")
    parser.add_argument("--data_root", default="dataset_root", help="Path to preprocessed dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (50 matches baseline)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic dataset for testing")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    print(f"=== Running Ablation 1 (Without DEM) on {device} for {args.epochs} epochs ===")

    # Dataloaders
    if args.synthetic:
        train_ds = SyntheticCloudDataset(n_samples=32, patch_size=256)
        val_ds = SyntheticCloudDataset(n_samples=16, patch_size=256)
        test_ds = val_ds
    else:
        train_ds = CloudReconstructionDataset(args.data_root, split="train")
        val_ds = CloudReconstructionDataset(args.data_root, split="val")
        test_ds = CloudReconstructionDataset(args.data_root, split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    gen = GeneratorNoDEM(c_opt=3, c_sar=2, c_temp=3, c_dem=4, base_ch=48).to(device)
    disc = MultiScaleDiscriminator(in_channels=3).to(device)
    ema = EMA(gen)

    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs * len(train_loader), eta_min=1e-6)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs * len(train_loader), eta_min=1e-6)

    gen_loss_fn = CombinedGeneratorLoss().to(device)
    adv_loss_fn = AdversarialLoss().to(device)
    scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_psnr = -1.0
    best_ckpt = "checkpoints/ablate1_no_dem_best.pt"

    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()
        total_g_loss = 0.0

        for batch in train_loader:
            opt_cloudy = batch["opt_cloudy"].to(device)
            opt_clean = batch["opt_clean"].to(device)
            sar = batch["sar"].to(device)
            temp = batch["temporal"].to(device)
            dem = batch["dem"].to(device)
            mask = batch["mask"].to(device)

            # Train Discriminator
            opt_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                mean_fake, logvar_fake = gen(opt_cloudy, sar, temp, dem)
                real_outputs = disc(opt_clean)
                fake_outputs_detached = disc(mean_fake.detach())
                real_logits = [out[0] for out in real_outputs]
                fake_logits_det = [out[0] for out in fake_outputs_detached]
                d_loss = adv_loss_fn.discriminator_loss(real_logits, fake_logits_det)

            if use_amp:
                scaler_d.scale(d_loss).backward()
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake_outputs = disc(mean_fake)
                real_outputs_det = [(out[0].detach(), [f.detach() for f in out[1]]) for out in real_outputs]
                g_total, breakdown = gen_loss_fn(mean_fake, logvar_fake, opt_clean, mask, real_outputs_det, fake_outputs)

            if use_amp:
                scaler_g.scale(g_total).backward()
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                g_total.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
                opt_g.step()

            ema.update(gen)
            sched_g.step()
            sched_d.step()
            total_g_loss += breakdown["total"]

        # Validation PSNR check using EMA model
        ema.shadow.eval()
        val_psnr_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                opt_cloudy = batch["opt_cloudy"].to(device)
                opt_clean = batch["opt_clean"].to(device)
                sar = batch["sar"].to(device)
                temp = batch["temporal"].to(device)
                dem = batch["dem"].to(device)
                pred, _ = ema.shadow(opt_cloudy, sar, temp, dem)
                
                mse = torch.mean((pred - opt_clean) ** 2, dim=[1, 2, 3])
                psnr = 10.0 * torch.log10(4.0 / torch.clamp(mse, min=1e-8))
                val_psnr_sum += psnr.sum().item()
                val_count += pred.size(0)

        epoch_psnr = val_psnr_sum / max(1, val_count)
        print(f"[Epoch {epoch:02d}/{args.epochs:02d}] G_Loss: {total_g_loss/len(train_loader):.4f} | Val PSNR: {epoch_psnr:.2f} dB")

        if epoch_psnr > best_val_psnr:
            best_val_psnr = epoch_psnr
            torch.save(ema.shadow.state_dict(), best_ckpt)

    print(f"\nTraining Complete! Best Val PSNR: {best_val_psnr:.2f} dB. Checkpoint: {best_ckpt}")
    print("Evaluating on 1,611 Held-Out Test Patches...")
    
    # Load best checkpoint and evaluate on Test Set
    eval_model = GeneratorNoDEM(c_opt=3, c_sar=2, c_temp=3, c_dem=4, base_ch=48).to(device)
    eval_model.load_state_dict(torch.load(best_ckpt, map_location=device))
    eval_model.eval()
    test_metrics = run_evaluation(eval_model, test_loader, device)
    
    results = {
        "ablation": "w/o 4-Channel DEM (Topography)",
        "epochs": args.epochs,
        "best_val_psnr": round(best_val_psnr, 2),
        "test_metrics": test_metrics
    }
    with open("checkpoints/ablate1_no_dem_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("=== ABLATION 1 TEST RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
