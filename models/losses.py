"""
Loss functions v2.

WHAT CHANGED VS V1 AND WHY:

1. Band order corrected: LISS-IV is [Green, Red, NIR] (3 bands, no Blue).
   NDVI_RED_IDX=1, NDVI_NIR_IDX=2.

2. v1's "perceptual loss" ran ImageNet-pretrained VGG16 on the first 3 channels
   treated as if they were RGB. That's a domain mismatch: VGG was trained on
   natural RGB photos, and even in v1's (wrong) 4-channel assumption, [R,G,B]
   satellite reflectance doesn't look like anything in ImageNet's training
   distribution. With the v2 3-band [Green,Red,NIR] data there isn't even a
   literal RGB triple to hand it. Default primary "perceptual-like" term is
   now a domain-agnostic MultiScaleStructuralLoss (Sobel gradient + windowed
   SSIM at 2 scales) - standard in image-restoration literature and doesn't
   assume any particular channel semantics. VGG is kept as an OPTIONAL,
   off-by-default auxiliary term applied to a false-color NIR-Red-Green
   composite (a standard remote-sensing display convention) for anyone who
   wants the extra texture prior and has internet access to fetch weights.

3. New: HeteroscedasticNLLLoss - the generator now predicts a per-pixel
   log-variance alongside the mean reconstruction. This loss calibrates that
   variance head (Kendall & Gal 2017 aleatoric-uncertainty formulation),
   which is what actually produces a meaningful confidence value in the
   dashboard instead of v1's "Confidence: N/A" placeholder.

4. Adversarial loss now averages across however many discriminator scales are
   passed in (works with MultiScaleDiscriminator's list output).
"""
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F

NDVI_RED_IDX = 1   # band order [Green, Red, NIR]
NDVI_NIR_IDX = 2


class MaskedL1Loss(nn.Module):
    def __init__(self, cloud_weight=5.0, clear_weight=1.0):
        super().__init__()
        self.cloud_weight = cloud_weight
        self.clear_weight = clear_weight

    def forward(self, pred, target, mask):
        weight = mask * self.cloud_weight + (1 - mask) * self.clear_weight
        return ((pred - target).abs() * weight).mean()


class HeteroscedasticNLLLoss(nn.Module):
    """
    Gaussian negative log-likelihood on (mean, log-variance) predictions.
    Weighted higher inside the cloud mask, same rationale as MaskedL1Loss.
    NOTE: this trains the variance head to be well-CALIBRATED (matches actual
    error magnitude), not just small - a network that is confidently wrong
    is penalized harder than one that is honestly uncertain.
    """

    def __init__(self, cloud_weight=5.0, clear_weight=1.0):
        super().__init__()
        self.cloud_weight = cloud_weight
        self.clear_weight = clear_weight

    def forward(self, mean_pred, logvar_pred, target, mask):
        weight = mask * self.cloud_weight + (1 - mask) * self.clear_weight
        precision = torch.exp(-logvar_pred)
        nll = 0.5 * precision * (mean_pred - target) ** 2 + 0.5 * logvar_pred
        return (nll * weight).mean()


class FeatureMatchingLoss(nn.Module):
    """Computes L1 distance between discriminator features of real vs fake."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, real_features_list, fake_features_list):
        if not isinstance(real_features_list, (list, tuple)):
            real_features_list = [real_features_list]
            fake_features_list = [fake_features_list]
        loss = 0.0
        n_features = 0
        for real_feats, fake_feats in zip(real_features_list, fake_features_list):
            for rf, ff in zip(real_feats, fake_feats):
                loss += self.l1(rf, ff)
                n_features += 1
        return loss / max(1, n_features)



class AdversarialLoss(nn.Module):
    """LSGAN loss, averaged across however many discriminator scales are given."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def generator_loss(self, disc_fake_logits_list):
        if not isinstance(disc_fake_logits_list, (list, tuple)):
            disc_fake_logits_list = [disc_fake_logits_list]
        losses = [self.mse(l, torch.ones_like(l)) for l in disc_fake_logits_list]
        return sum(losses) / len(losses)

    def discriminator_loss(self, disc_real_logits_list, disc_fake_logits_list):
        if not isinstance(disc_real_logits_list, (list, tuple)):
            disc_real_logits_list = [disc_real_logits_list]
            disc_fake_logits_list = [disc_fake_logits_list]
        losses = []
        for real_l, fake_l in zip(disc_real_logits_list, disc_fake_logits_list):
            loss_real = self.mse(real_l, torch.ones_like(real_l))
            loss_fake = self.mse(fake_l, torch.zeros_like(fake_l))
            losses.append(0.5 * (loss_real + loss_fake))
        return sum(losses) / len(losses)


_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)


class MultiScaleStructuralLoss(nn.Module):
    """
    Domain-agnostic structural loss: multi-scale Sobel-gradient L1 + windowed
    SSIM, computed per-channel (works for any number/semantics of bands, no
    ImageNet dependency). This is the primary "perceptual-like" term in v2.
    """

    def __init__(self, n_scales=2, ssim_window=7):
        super().__init__()
        self.n_scales = n_scales
        self.ssim_window = ssim_window
        self.register_buffer("sobel_x", _SOBEL_X.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", _SOBEL_Y.view(1, 1, 3, 3))

    def _gradient_l1(self, pred, target):
        B, C, H, W = pred.shape
        p = pred.reshape(B * C, 1, H, W)
        t = target.reshape(B * C, 1, H, W)
        sobel_x = self.sobel_x.to(device=p.device, dtype=p.dtype)
        sobel_y = self.sobel_y.to(device=p.device, dtype=p.dtype)
        gx_p = F.conv2d(p, sobel_x, padding=1)
        gy_p = F.conv2d(p, sobel_y, padding=1)
        gx_t = F.conv2d(t, sobel_x, padding=1)
        gy_t = F.conv2d(t, sobel_y, padding=1)
        return (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()

    def _ssim(self, pred, target, C1=0.01 ** 2, C2=0.03 ** 2):
        w = self.ssim_window
        pad = w // 2
        mu_p = F.avg_pool2d(pred, w, stride=1, padding=pad)
        mu_t = F.avg_pool2d(target, w, stride=1, padding=pad)
        sigma_p = F.avg_pool2d(pred ** 2, w, stride=1, padding=pad) - mu_p ** 2
        sigma_t = F.avg_pool2d(target ** 2, w, stride=1, padding=pad) - mu_t ** 2
        sigma_pt = F.avg_pool2d(pred * target, w, stride=1, padding=pad) - mu_p * mu_t
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
            (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2)
        )
        return 1 - ssim_map.mean()  # loss form: 0 = identical

    def forward(self, pred01, target01):
        loss = 0.0
        p, t = pred01, target01
        for scale in range(self.n_scales):
            loss = loss + self._gradient_l1(p, t) + self._ssim(p, t)
            if scale < self.n_scales - 1:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
        return loss / self.n_scales


class SpectralConsistencyLoss(nn.Module):
    """NDVI consistency between reconstructed and ground-truth pixels."""

    def __init__(self, red_idx=NDVI_RED_IDX, nir_idx=NDVI_NIR_IDX, eps=1e-6):
        super().__init__()
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.eps = eps

    def _ndvi(self, x01):
        red = x01[:, self.red_idx : self.red_idx + 1]
        nir = x01[:, self.nir_idx : self.nir_idx + 1]
        return (nir - red) / (nir + red + self.eps)

    def forward(self, pred01, target01):
        return F.l1_loss(self._ndvi(pred01), self._ndvi(target01))


class FFTLoss(nn.Module):
    """Frequency-domain L1 loss on 2D FFT magnitude spectrum.

    Penalizes discrepancies in high-frequency content (edges, texture) that
    spatial-domain losses like L1/SSIM are relatively insensitive to.
    Standard in recent image restoration literature (e.g. Focal Frequency Loss,
    ECCV 2022).
    """

    def __init__(self, log_scale=True):
        super().__init__()
        self.log_scale = log_scale

    def forward(self, pred01, target01):
        # 2D FFT per channel, shift DC to center
        fft_pred = torch.fft.fft2(pred01, norm="ortho")
        fft_target = torch.fft.fft2(target01, norm="ortho")
        mag_pred = torch.abs(fft_pred)
        mag_target = torch.abs(fft_target)
        if self.log_scale:
            mag_pred = torch.log1p(mag_pred)
            mag_target = torch.log1p(mag_target)
        return F.l1_loss(mag_pred, mag_target)


def _has_internet():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1.5)
        return True
    except OSError:
        return False


class OptionalVGGAuxLoss(nn.Module):
    """
    Off by default. If enabled, maps [Green,Red,NIR] -> false-color composite
    [NIR,Red,Green] (a standard CIR display convention) and runs VGG16 on
    that. Treat this strictly as an auxiliary texture prior, not a
    scientifically literal "perceptual" match, since it's still out-of-domain
    for ImageNet-pretrained features.
    """

    def __init__(self, device="cpu"):
        super().__init__()
        import torchvision.models as tv_models

        weights = tv_models.VGG16_Weights.IMAGENET1K_V1 if _has_internet() else None
        vgg = tv_models.vgg16(weights=weights).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.slice = nn.Sequential(*list(vgg.children())[:9])  # up to relu2_2
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _false_color(self, x01):
        # x01 channel order [Green, Red, NIR] -> display as [NIR, Red, Green]
        nir, red, green = x01[:, 2:3], x01[:, 1:2], x01[:, 0:1]
        rgb = torch.cat([nir, red, green], dim=1)
        return (rgb - self.mean.to(dtype=rgb.dtype)) / self.std.to(dtype=rgb.dtype)

    def forward(self, pred01, target01):
        p = self.slice(self._false_color(pred01))
        t = self.slice(self._false_color(target01))
        return F.l1_loss(p, t)


class CombinedGeneratorLoss(nn.Module):
    """
    Weighted sum: masked L1 (mean) + heteroscedastic NLL (variance calibration)
    + adversarial + multi-scale structural + spectral consistency
    [+ optional VGG aux, weight 0 by default] + feature matching.
    """

    def __init__(
        self,
        device="cpu",
        w_l1=10.0,
        w_nll=1.0,
        w_adv=1.0,
        w_structural=2.0,
        w_spectral=5.0,
        w_vgg_aux=0.0,
        w_fm=10.0,
        w_fft=1.0,
    ):
        super().__init__()
        self.masked_l1 = MaskedL1Loss()
        self.nll = HeteroscedasticNLLLoss()
        self.adversarial = AdversarialLoss()
        self.structural = MultiScaleStructuralLoss()
        self.spectral = SpectralConsistencyLoss()
        self.feature_matching = FeatureMatchingLoss()
        self.fft_loss = FFTLoss()
        self.w_l1 = w_l1
        self.w_nll = w_nll
        self.w_adv = w_adv
        self.w_structural = w_structural
        self.w_spectral = w_spectral
        self.w_vgg_aux = w_vgg_aux
        self.w_fm = w_fm
        self.w_fft = w_fft
        self.vgg_aux = OptionalVGGAuxLoss(device=device) if w_vgg_aux > 0 else None

    @staticmethod
    def _to01(x_tanh):
        return (x_tanh + 1) / 2

    def forward(self, mean_pred, logvar_pred, target, mask, disc_real_outputs, disc_fake_outputs):
        l1 = self.masked_l1(mean_pred, target, mask)
        nll = self.nll(mean_pred, logvar_pred, target, mask)
        
        fake_logits = [out[0] for out in disc_fake_outputs]
        adv = self.adversarial.generator_loss(fake_logits)
        
        real_features = [out[1] for out in disc_real_outputs]
        fake_features = [out[1] for out in disc_fake_outputs]
        fm = self.feature_matching(real_features, fake_features)

        pred01 = self._to01(mean_pred).clamp(0, 1)
        target01 = self._to01(target).clamp(0, 1)
        structural = self.structural(pred01, target01)
        spectral = self.spectral(pred01, target01)
        fft = self.fft_loss(pred01, target01)

        total = (
            self.w_l1 * l1
            + self.w_nll * nll
            + self.w_adv * adv
            + self.w_structural * structural
            + self.w_spectral * spectral
            + self.w_fm * fm
            + self.w_fft * fft
        )
        breakdown = {
            "l1": l1.item(),
            "nll": nll.item(),
            "adv": adv.item(),
            "structural": structural.item(),
            "spectral": spectral.item(),
            "fm": fm.item(),
            "fft": fft.item(),
        }
        if self.vgg_aux is not None:
            vgg_loss = self.vgg_aux(pred01, target01)
            total = total + self.w_vgg_aux * vgg_loss
            breakdown["vgg_aux"] = vgg_loss.item()
        breakdown["total"] = total.item()
        return total, breakdown


if __name__ == "__main__":
    pred = torch.rand(2, 3, 64, 64) * 2 - 1
    target = torch.rand(2, 3, 64, 64) * 2 - 1
    logvar = torch.randn(2, 3, 64, 64) * 0.1
    mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    real_outputs = [
        (torch.randn(2, 1, 6, 6), [torch.randn(2, 64, 32, 32), torch.randn(2, 128, 16, 16)]),
        (torch.randn(2, 1, 3, 3), [torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8)])
    ]
    fake_outputs = [
        (torch.randn(2, 1, 6, 6), [torch.randn(2, 64, 32, 32), torch.randn(2, 128, 16, 16)]),
        (torch.randn(2, 1, 3, 3), [torch.randn(2, 64, 16, 16), torch.randn(2, 128, 8, 8)])
    ]

    loss_fn = CombinedGeneratorLoss(device="cpu")
    total, breakdown = loss_fn(pred, logvar, target, mask, real_outputs, fake_outputs)
    print("Combined loss breakdown:", breakdown)
    print("OK")
