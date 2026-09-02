"""
Ablation Study Variant 5: Training WITHOUT Heteroscedastic Uncertainty Head (Standard L1 Loss Only).
Evaluates the contribution of dynamic observation-noise attenuation to training stability.

Output checkpoint: checkpoints/ablate5_no_unc_best.pt
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


def main():
    parser = argparse.ArgumentParser(description="Ablation 5: Train without Heteroscedastic Uncertainty Head (L1 only)")
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
    print(f"=== Running Ablation 5 (Without Uncertainty Head, w_nll=0.0) on {device} for {args.epochs} epochs ===")

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

    gen = CloudReconstructionGeneratorV2(c_opt=3, c_sar=2, c_temp=3, c_dem=4, base_ch=48).to(device)
    disc = MultiScaleDiscriminator(in_channels=3).to(device)
    ema = EMA(gen)

    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs * len(train_loader), eta_min=1e-6)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs * len(train_loader), eta_min=1e-6)

    # Ablation 5: w_nll set to 0.0 (Standard L1 supervision only)
    gen_loss_fn = CombinedGeneratorLoss(w_nll=0.0).to(device)
    adv_loss_fn = AdversarialLoss().to(device)
    scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_psnr = -1.0
    best_ckpt = "checkpoints/ablate5_no_unc_best.pt"

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
    
    eval_model = CloudReconstructionGeneratorV2(c_opt=3, c_sar=2, c_temp=3, c_dem=4, base_ch=48).to(device)
    eval_model.load_state_dict(torch.load(best_ckpt, map_location=device))
    eval_model.eval()
    summary_full_ref, summary_no_ref, calibration_corr = run_evaluation(eval_model, test_loader, device)
    test_metrics = summary_full_ref["model"]
    
    results = {
        "ablation": "w/o Uncertainty Head (L1 Loss Only)",
        "epochs": args.epochs,
        "best_val_psnr": round(best_val_psnr, 2),
        "test_metrics": test_metrics
    }
    with open("checkpoints/ablate5_no_unc_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("=== ABLATION 5 TEST RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
