"""
Multi-GPU DistributedDataParallel (DDP) Training Script for CloudFree Vision v2.

Runs 2 dedicated processes via torchrun on Dual T4 GPUs (cuda:0 and cuda:1).
Uses NCCL backend for ring-allreduce gradient synchronization.
"""

import os
import sys
import copy
import argparse
import time
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from models.generator import CloudReconstructionGeneratorV2
from models.discriminator import MultiScaleDiscriminator
from models.losses import CombinedGeneratorLoss, AdversarialLoss
from data.dataset import CloudReconstructionDataset, SyntheticCloudDataset
from evaluate import run_evaluation


class EMA:
    """Exponential moving average of a model's parameters."""
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


def warmup_cosine_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def get_dataloaders_ddp(args, rank, world_size, severity=0.6):
    if args.synthetic:
        n_train = args.max_train_patches if args.max_train_patches else args.synthetic_n
        n_val = args.max_val_patches if args.max_val_patches else max(4, n_train // 4)
        train_ds = SyntheticCloudDataset(n_samples=n_train, patch_size=args.patch_size, severity=severity)
        val_ds = SyntheticCloudDataset(n_samples=n_val, patch_size=args.patch_size, severity=0.8, seed=1)
    else:
        train_ds = CloudReconstructionDataset(args.data_root, split="train")
        val_ds = CloudReconstructionDataset(args.data_root, split="val")
        if args.max_train_patches:
            train_ds.patch_ids = train_ds.patch_ids[:args.max_train_patches]
        if args.max_val_patches:
            val_ds.patch_ids = val_ds.patch_ids[:args.max_val_patches]

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if rank == 0:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_ds, train_sampler, train_loader, val_loader


def train_one_epoch_ddp(gen, disc, ema, loader, opt_g, opt_d, sched_g, sched_d,
                        gen_loss_fn, adv_loss_fn, device, scaler_g, scaler_d, rank):
    gen.train()
    disc.train()
    running = {"g_total": 0.0, "d_loss": 0.0, "l1": 0.0, "nll": 0.0, "structural": 0.0, "spectral": 0.0, "fm": 0.0}
    n = 0

    for batch in loader:
        opt_cloudy = batch["opt_cloudy"].to(device, non_blocking=True)
        opt_clean = batch["opt_clean"].to(device, non_blocking=True)
        sar = batch["sar"].to(device, non_blocking=True)
        temporal = batch["temporal"].to(device, non_blocking=True)
        dem = batch["dem"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        opt_d.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=True):
            mean_fake, logvar_fake = gen(opt_cloudy, sar, temporal, dem)
            real_outputs = disc(opt_clean)
            fake_outputs_detached = disc(mean_fake.detach())
            real_logits = [out[0] for out in real_outputs]
            fake_logits_det = [out[0] for out in fake_outputs_detached]
            d_loss = adv_loss_fn.discriminator_loss(real_logits, fake_logits_det)

        scaler_d.scale(d_loss).backward()
        scaler_d.unscale_(opt_d)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        scaler_d.step(opt_d)
        scaler_d.update()

        opt_g.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=True):
            fake_outputs = disc(mean_fake)
            real_outputs_det = [(out[0].detach(), [f.detach() for f in out[1]]) for out in real_outputs]
            g_total, breakdown = gen_loss_fn(mean_fake, logvar_fake, opt_clean, mask, real_outputs_det, fake_outputs)

        scaler_g.scale(g_total).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        scaler_g.step(opt_g)
        scaler_g.update()

        sched_g.step()
        sched_d.step()

        if rank == 0 and ema is not None:
            ema.update(gen.module)

        running["g_total"] += g_total.item()
        running["l1"] += breakdown.get("l1", 0.0)
        running["nll"] += breakdown.get("nll", 0.0)
        running["structural"] += breakdown.get("structural", 0.0)
        running["spectral"] += breakdown.get("spectral", 0.0)
        running["fm"] += breakdown.get("fm", 0.0)
        running["d_loss"] += d_loss.item()
        n += 1

    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def validate_dual(gen_ema, loader, device):
    gen_ema.eval()
    summary_full_ref, summary_no_ref, calib_corr = run_evaluation(gen_ema, loader, device, eval_mode="both")
    return summary_full_ref["model"], summary_no_ref["model"], calib_corr


def main():
    p = argparse.ArgumentParser(description="Train Stage B v2 on Dual T4 GPUs using DistributedDataParallel")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_n", type=int, default=32)
    p.add_argument("--max_train_patches", type=int, default=None)
    p.add_argument("--max_val_patches", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size (Total batch = batch_size * 2)")
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--curriculum", action="store_true", default=True)
    p.add_argument("--no_curriculum", dest="curriculum", action="store_false")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("==========================================================================================", flush=True)
        print("CLOUDFREE VISION V2 — MULTI-GPU DISTRIBUTED TRAINING (DDP)", flush=True)
        print(f"World Size: {world_size} GPUs | Per-GPU Batch: {args.batch_size} | Effective Total Batch: {args.batch_size * world_size}", flush=True)
        print("==========================================================================================", flush=True)

    start_severity = 0.2 if (args.synthetic and args.curriculum) else 0.6
    train_ds, train_sampler, train_loader, val_loader = get_dataloaders_ddp(
        args, rank=local_rank, world_size=world_size, severity=start_severity
    )

    if local_rank == 0:
        print(f"Dataset Loaded: {len(train_ds)} train patches, {len(train_loader)} batches per GPU.", flush=True)

    raw_gen = CloudReconstructionGeneratorV2(base_ch=args.base_ch).to(device)
    raw_disc = MultiScaleDiscriminator(in_channels=3, n_scales=2).to(device)

    gen = DDP(raw_gen, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    disc = DDP(raw_disc, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    ema = EMA(raw_gen, decay=args.ema_decay) if local_rank == 0 else None

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: warmup_cosine_lambda(s, warmup_steps, total_steps))
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda s: warmup_cosine_lambda(s, warmup_steps, total_steps))

    gen_loss_fn = CombinedGeneratorLoss(device=device).to(device)
    adv_loss_fn = AdversarialLoss().to(device)
    scaler_g = torch.amp.GradScaler("cuda", enabled=True)
    scaler_d = torch.amp.GradScaler("cuda", enabled=True)

    if local_rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_psnr = -float("inf")

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        stats = train_one_epoch_ddp(
            gen, disc, ema, train_loader, opt_g, opt_d, sched_g, sched_d,
            gen_loss_fn, adv_loss_fn, device, scaler_g, scaler_d, local_rank
        )
        dt = time.time() - t0

        if local_rank == 0:
            regA, regB, calib_corr = validate_dual(ema.shadow, val_loader, device)
            print(f"\n==========================================================================================", flush=True)
            print(f"[DDP Epoch {epoch}/{args.epochs}] Completed in {dt:.1f}s across {world_size} GPUs", flush=True)
            print(f"  Train Losses: Total={stats['g_total']:.3f} | L1={stats['l1']:.3f} | NLL={stats['nll']:.3f} | Struct={stats['structural']:.3f} | Spec={stats['spectral']:.3f} | FM={stats['fm']:.3f} | D_Loss={stats['d_loss']:.3f}", flush=True)
            print(f"  Regime A (Full-Ref): PSNR={regA['psnr']:.2f}dB | SSIM={regA['ssim']:.3f} | SAM={regA['sam']:.2f}° | ERGAS={regA['ergas']:.2f} | CC={regA['cc']:.3f}", flush=True)
            print(f"  Regime B (No-Ref):   Laplacian Sharpness={regB['laplacian_sharpness_score']:.2f} | SAR-Optical Grad Corr={regB['sar_gradient_correlation']:.3f}", flush=True)
            print(f"==========================================================================================", flush=True)

            if regA['psnr'] > best_val_psnr:
                best_val_psnr = regA['psnr']
                ckpt_path = os.path.join(args.checkpoint_dir, "generator_best.pt")
                torch.save(
                    {
                        "gen_state": gen.module.state_dict(),
                        "gen_ema_state": ema.shadow.state_dict(),
                        "disc_state": disc.module.state_dict(),
                        "epoch": epoch,
                        "val_psnr": regA['psnr'],
                    },
                    ckpt_path,
                )
                print(f"  -> Saved best DDP checkpoint to {ckpt_path}", flush=True)

    dist.destroy_process_group()
    if local_rank == 0:
        print("\nMulti-GPU DDP Training Complete! All checkpoints saved successfully.", flush=True)


if __name__ == "__main__":
    main()
