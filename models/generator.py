"""
Stage B v2: Cross-Attention Fusion Generator with heteroscedastic uncertainty.

CORRECTED FROM v1: LISS-IV is a 3-band sensor - Green (B2), Red (B3), NIR (B4).
There is no Blue band. v1 incorrectly assumed 4 channels [R,G,B,NIR]. Default
here is c_opt=3, band order [Green, Red, NIR]. If your teammate's pipeline
uses a different order, update NDVI_RED_IDX/NDVI_NIR_IDX in losses.py to match.

WHAT'S NEW VS V1:
  1. Real cross-modal attention fusion (WindowedCrossAttention): optical
     features are the query, SAR+temporal are key/value. The network learns
     to lean on SAR/temporal when optical is unreliable (cloudy), instead of
     v1's fixed channel-concatenation. This is what the pitch deck's
     "Attention-based Fusion... adaptive weighting based on reliability" bullet
     actually refers to - v1 never implemented it, only asserted it.
  2. Dual-head output: predicts both the reconstructed mean AND a per-pixel
     log-variance (heteroscedastic uncertainty). This is what actually
     produces the "Confidence" value in the dashboard/wireframe - v1 had no
     mechanism for this and just displayed "N/A".
  3. Full-resolution skip connection from the optical stem straight to the
     final upsampling block, so fine spatial detail isn't lost by fusing at
     half-resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import WindowedSelfAttention, WindowedCrossAttention
except ImportError:
    from attention import WindowedSelfAttention, WindowedCrossAttention


def conv_bn_act(in_ch, out_ch, k=3, s=1, p=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p), nn.InstanceNorm2d(out_ch)]
    if act:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(conv_bn_act(ch, ch), conv_bn_act(ch, ch, act=False))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(conv_bn_act(in_ch, out_ch, s=2), ResBlock(out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = nn.Sequential(conv_bn_act(out_ch + skip_ch, out_ch), ResBlock(out_ch))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SpatialAlignmentSTN(nn.Module):
    def __init__(self, c_aux, c_opt):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(c_aux + c_opt, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 6)
        )
        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, aux, opt):
        x = torch.cat([aux, opt], dim=1)
        x = self.loc(x)
        x = x.view(x.size(0), -1)
        theta = self.fc(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, aux.size(), align_corners=False)
        aux_aligned = F.grid_sample(aux, grid, align_corners=False)
        return aux_aligned


class CloudReconstructionGeneratorV2(nn.Module):
    def __init__(
        self,
        c_opt=3,       # Green, Red, NIR (LISS-IV) -- corrected from v1's 4
        c_sar=2,       # VV, VH
        c_temp=3,      # matches c_opt band order
        c_dem=4,       # Elevation_norm, Slope_norm, sin(Aspect), cos(Aspect)
        base_ch=48,
        attn_heads=4,
        window=8,
    ):
        super().__init__()
        self.c_opt = c_opt

        # --- full-resolution per-modality stems (kept for a long skip connection) ---
        self.opt_stem = conv_bn_act(c_opt, base_ch)
        self.sar_stem = conv_bn_act(c_sar, base_ch // 2)
        self.temp_stem = conv_bn_act(c_temp, base_ch // 2)
        self.dem_stem = conv_bn_act(c_dem, base_ch // 4)

        # --- downsample each stream to a working resolution for cross-attention ---
        self.opt_down = conv_bn_act(base_ch, base_ch, s=2)
        self.sar_down = conv_bn_act(base_ch // 2, base_ch // 2, s=2)
        self.temp_down = conv_bn_act(base_ch // 2, base_ch // 2, s=2)
        self.dem_down = conv_bn_act(base_ch // 4, base_ch // 4, s=2)

        self.sar_stn = SpatialAlignmentSTN(base_ch // 2, base_ch)
        self.temp_stn = SpatialAlignmentSTN(base_ch // 2, base_ch)

        self.aux_proj = conv_bn_act(base_ch // 2 + base_ch // 2, base_ch, k=1, p=0)
        self.cross_attn = WindowedCrossAttention(base_ch, num_heads=attn_heads, window=window)
        self.fuse = conv_bn_act(base_ch + base_ch // 4, base_ch)  # + projected DEM

        # --- U-Net trunk on the fused feature map ---
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.self_attn = WindowedSelfAttention(base_ch * 4, num_heads=attn_heads, window=window)
        self.bottleneck_res = ResBlock(base_ch * 4)

        self.up1 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch, base_ch)
        self.final_up = Up(base_ch, base_ch, base_ch)  # skip = full-res opt_stem

        self.mean_head = nn.Sequential(nn.Conv2d(base_ch, c_opt, 3, padding=1), nn.Tanh())
        self.logvar_head = nn.Conv2d(base_ch, c_opt, 3, padding=1)

        self._last_reliance_on_aux = None  # diagnostic, set on forward()

    def forward(self, opt, sar, temp, dem):
        opt_full = self.opt_stem(opt)
        sar_full = self.sar_stem(sar)
        temp_full = self.temp_stem(temp)
        dem_full = self.dem_stem(dem)

        opt_h = self.opt_down(opt_full)
        sar_h = self.sar_down(sar_full)
        temp_h = self.temp_down(temp_full)
        dem_h = self.dem_down(dem_full)

        sar_h = self.sar_stn(sar_h, opt_h)
        temp_h = self.temp_stn(temp_h, opt_h)

        aux_h = self.aux_proj(torch.cat([sar_h, temp_h], dim=1))
        fused_h, reliance_on_aux = self.cross_attn(opt_h, aux_h)
        self._last_reliance_on_aux = reliance_on_aux

        f0 = self.fuse(torch.cat([fused_h, dem_h], dim=1))  # H/2, base_ch

        f1 = self.down1(f0)   # H/4, base_ch*2
        f2 = self.down2(f1)   # H/8, base_ch*4
        f2 = self.self_attn(f2)
        f2 = self.bottleneck_res(f2)

        d1 = self.up1(f2, f1)          # H/4
        d0 = self.up2(d1, f0)          # H/2
        d_full = self.final_up(d0, opt_full)  # H  (full-res skip)

        mean_out = self.mean_head(d_full)
        logvar_out = self.logvar_head(d_full).clamp(-6.0, 6.0)  # numeric stability
        return mean_out, logvar_out

    def predict_confidence(self, logvar_out):
        """
        Maps predicted log-variance to a 0-100 'confidence' score for the
        dashboard. std -> confidence via a smooth decay; purely a display
        heuristic, the underlying quantity that matters is std itself.
        """
        std = torch.exp(0.5 * logvar_out)
        std_avg = std.mean(dim=1, keepdim=True)  # average across bands
        confidence = 100.0 * torch.exp(-std_avg / 0.15)
        return confidence.clamp(0, 100)


if __name__ == "__main__":
    g = CloudReconstructionGeneratorV2(c_opt=3, c_sar=2, c_temp=3, c_dem=4, base_ch=48)
    opt = torch.randn(2, 3, 256, 256)
    sar = torch.randn(2, 2, 256, 256)
    temp = torch.randn(2, 3, 256, 256)
    dem = torch.randn(2, 4, 256, 256)
    mean_out, logvar_out = g(opt, sar, temp, dem)
    print("mean_out:", mean_out.shape, "logvar_out:", logvar_out.shape)
    assert mean_out.shape == (2, 3, 256, 256)
    assert logvar_out.shape == (2, 3, 256, 256)
    conf = g.predict_confidence(logvar_out)
    print("confidence map:", conf.shape, "range:", conf.min().item(), conf.max().item())
    print("reliance_on_aux (mean cross-attn weight on SAR+temporal):", g._last_reliance_on_aux)
    n_params = sum(p.numel() for p in g.parameters())
    print(f"Generator params: {n_params/1e6:.2f}M")
    print("OK")
