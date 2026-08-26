"""
Stage A v2: Cloud & Shadow Segmentation Model.
CORRECTED FROM V1: default in_channels=3 (LISS-IV is Green/Red/NIR, no Blue).
Output classes unchanged: {0: clear, 1: cloud, 2: shadow}.
Architecture unchanged from v1 - it was already solid; only the band-count
default needed fixing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        if psi.shape[-2:] != x.shape[-2:]:
            psi = F.interpolate(psi, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x * psi

class UNetSegmenter(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_ch=32, c_sar=2):
        super().__init__()
        c = base_ch
        
        self.c_sar = c_sar
        if self.c_sar > 0:
            self.sar_stem = conv_block(c_sar, c)
            self.sar_fuse = nn.Conv2d(c * 2, c, 1)

        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.enc4 = conv_block(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.ag4 = AttentionGate(gate_ch=c * 8, skip_ch=c * 8, inter_ch=c * 4)
        self.dec4 = conv_block(c * 16, c * 8)
        
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.ag3 = AttentionGate(gate_ch=c * 4, skip_ch=c * 4, inter_ch=c * 2)
        self.dec3 = conv_block(c * 8, c * 4)
        
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.ag2 = AttentionGate(gate_ch=c * 2, skip_ch=c * 2, inter_ch=c)
        self.dec2 = conv_block(c * 4, c * 2)
        
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.ag1 = AttentionGate(gate_ch=c, skip_ch=c, inter_ch=c // 2)
        self.dec1 = conv_block(c * 2, c)

        self.out_conv = nn.Conv2d(c, num_classes, 1)

    def forward(self, x, sar=None):
        e1 = self.enc1(x)
        if sar is not None and self.c_sar > 0:
            s = self.sar_stem(sar)
            if s.shape[-2:] != e1.shape[-2:]:
                s = F.interpolate(s, size=e1.shape[-2:], mode='bilinear', align_corners=False)
            e1 = torch.cat([e1, s], dim=1)
            e1 = self.sar_fuse(e1)

        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, size=e4.shape[-2:], mode='bilinear', align_corners=False)
        e4_att = self.ag4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        
        d3 = self.up3(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        e3_att = self.ag3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        e2_att = self.ag2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        e1_att = self.ag1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.out_conv(d1)


if __name__ == "__main__":
    m = UNetSegmenter(in_channels=3, num_classes=3)
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    print("UNetSegmenter output (optical only):", y.shape)
    assert y.shape == (2, 3, 256, 256)
    
    sar = torch.randn(2, 2, 256, 256)
    y2 = m(x, sar=sar)
    print("UNetSegmenter output (optical + SAR):", y2.shape)
    assert y2.shape == (2, 3, 256, 256)
    
    # Test mismatch dims
    m3 = UNetSegmenter(in_channels=3, num_classes=3)
    x3 = torch.randn(2, 3, 257, 257)
    y3 = m3(x3)
    print("UNetSegmenter output (mismatched dims):", y3.shape)
    assert y3.shape == (2, 3, 257, 257)
    
    print("OK")
