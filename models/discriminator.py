"""
v2: Multi-scale PatchGAN discriminator (pix2pixHD-style).

WHAT'S NEW VS V1: v1 used a single PatchGAN at one resolution. A single scale
tends to either focus on fine texture (small receptive field) or global
structure (large receptive field), not both. Running two discriminators - one
on the full-res image, one on a 2x-downsampled version - gives the generator
gradient signal for both local texture realism and larger-scale structural
coherence (field boundaries, road continuity), which matters more for
satellite imagery than for typical photo-to-photo GAN tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, n_layers=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        self.blocks.append(nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        
        ch = base_ch
        for i in range(1, n_layers):
            out_ch = min(ch * 2, base_ch * 8)
            self.blocks.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(ch, out_ch, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch = out_ch
            
        out_ch = min(ch * 2, base_ch * 8)
        self.blocks.append(nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ch, out_ch, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        
        self.final_layer = nn.utils.spectral_norm(nn.Conv2d(out_ch, 1, 4, stride=1, padding=1))

    def forward(self, x):
        features = []
        out = x
        for block in self.blocks:
            out = block(out)
            features.append(out)
        logits = self.final_layer(out)
        return logits, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, n_layers=3, n_scales=2):
        super().__init__()
        self.n_scales = n_scales
        self.discs = nn.ModuleList(
            [PatchGANDiscriminator(in_channels, base_ch, n_layers) for _ in range(n_scales)]
        )

    def forward(self, x):
        outputs = []
        cur = x
        for i, disc in enumerate(self.discs):
            outputs.append(disc(cur))
            if i < self.n_scales - 1:
                cur = F.avg_pool2d(cur, kernel_size=3, stride=2, padding=1)
        return outputs  # list of (logits, features) tuples, one per scale


if __name__ == "__main__":
    d = MultiScaleDiscriminator(in_channels=3, n_scales=2)
    x = torch.randn(2, 3, 256, 256)
    outs = d(x)
    for i, (logits, feats) in enumerate(outs):
        print(f"scale {i}: logits {logits.shape}, num feats {len(feats)}")
    assert len(outs) == 2
    print("OK")
