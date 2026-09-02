"""
Shared windowed-attention primitives.

Two attention types are used in the v2 generator:
  - WindowedCrossAttention: shallow fusion — optical features (query) attend to
    SAR + temporal features (key/value). This is what actually implements
    "adaptive weighting of features based on reliability & context" from the
    architecture slide: when the optical patch is heavily clouded, the network
    can learn to lean on SAR/temporal instead, rather than fixed concatenation.
  - WindowedSelfAttention: deep bottleneck — global spatial context (same role
    as in v1, kept because it's still useful for long-range structure).

Both operate on local windows (not full quadratic attention) to keep compute
bounded on hackathon-scale GPUs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window):
    """(B,C,H,W) -> (B*nH*nW, window*window, C), with padding info returned."""
    B, C, H, W = x.shape
    pad_h = (window - H % window) % window
    pad_w = (window - W % window) % window
    x = F.pad(x, (0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.permute(0, 2, 3, 1)  # B,H,W,C
    x = x.view(B, Hp // window, window, Wp // window, window, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, C)
    return x, (B, Hp, Wp, C)


def window_reverse(x, window, shape, orig_hw):
    B, Hp, Wp, C = shape
    H, W = orig_hw
    x = x.view(B, Hp // window, Wp // window, window, window, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C).permute(0, 3, 1, 2)
    return x[:, :, :H, :W]


class WindowedSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window=8):
        super().__init__()
        self.window = window
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_pre = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = x.shape[-2:]
        tokens, shape = window_partition(x, self.window)
        tokens_n = self.norm_pre(tokens)
        attn_out, _ = self.mha(tokens_n, tokens_n, tokens_n)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ff(tokens))
        return window_reverse(tokens, self.window, shape, (H, W))


class WindowedCrossAttention(nn.Module):
    """Query from one modality (optical), Key/Value from another (SAR+temporal)."""

    def __init__(self, dim, num_heads=4, window=8):
        super().__init__()
        self.window = window
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm_ff = nn.LayerNorm(dim)

    def forward(self, query_feat, kv_feat):
        H, W = query_feat.shape[-2:]
        q_tokens, shape = window_partition(query_feat, self.window)
        kv_tokens, _ = window_partition(kv_feat, self.window)

        q_tokens_n = self.norm_q(q_tokens)
        kv_tokens_n = self.norm_kv(kv_tokens)
        attn_out, attn_weights = self.mha(q_tokens_n, kv_tokens_n, kv_tokens_n)
        fused = q_tokens + attn_out  # residual: optical + attended aux context
        fused = self.norm_ff(fused + self.ff(fused))

        out = window_reverse(fused, self.window, shape, (H, W))
        if attn_weights is not None:
            # Shannon entropy across auxiliary spatial keys within window (measuring spatial dispersion)
            entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
            aux_spatial_entropy = entropy.mean().item()
        else:
            aux_spatial_entropy = None
        return out, aux_spatial_entropy


if __name__ == "__main__":
    x = torch.randn(2, 32, 64, 64)
    y = torch.randn(2, 32, 64, 64)
    sa = WindowedSelfAttention(32, num_heads=4, window=8)
    out_sa = sa(x)
    assert out_sa.shape == x.shape
    print("Self-attention OK:", out_sa.shape)

    ca = WindowedCrossAttention(32, num_heads=4, window=8)
    out_ca, aux_entropy = ca(x, y)
    assert out_ca.shape == x.shape
    print("Cross-attention OK:", out_ca.shape, f"aux_spatial_entropy={aux_entropy:.4f}")
    print("OK")
