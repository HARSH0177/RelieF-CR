"""
PyTorch Dataset v2.

UPDATED DATA CONTRACT:
  - opt_cloudy / opt_clean / temporal are 3 channels [Green, Red, NIR] float32 in [0, 1] (mapped to tanh [-1, 1] on load).
  - mask is continuous soft alpha opacity in [0.0, 1.0] (1, H, W) float32 for generator loss weighting.
  - mask_class is discrete semantic mask in {0, 1, 2} (H, W) int64 (0=clear, 1=cloud, 2=shadow) for Stage A segmentation.
  - sar is 2 channels [VV, VH] dB-scaled float32 in [-1, 1].
  - dem is 4 channels [Elev_norm, Slope_norm, sin(Aspect), cos(Aspect)] float32 in [-1, 1].

dataset_root/
  train/
    opt_cloudy/    scene_XX_p0001.npy   (3, H, W) float32, TOA reflectance in [0,1]
    opt_clean/     scene_XX_p0001.npy   (3, H, W) ground truth TOA reflectance
    sar/           scene_XX_p0001.npy   (2, H, W) float32, dB-scaled Sentinel-1
    temporal/      scene_XX_p0001.npy   (3, H, W) recent clear Sentinel-2 reference
    dem/           scene_XX_p0001.npy   (4, H, W) float32, Horn's terrain [Elev_norm, Slope_norm, sin(Asp), cos(Asp)]
    mask/          scene_XX_p0001.npy   (1, H, W) float32, continuous soft alpha in [0.0, 1.0]
    mask_class/    scene_XX_p0001.npy   (1, H, W) int64 in {0, 1, 2} (0=clear, 1=cloud, 2=shadow)
  val/    ... same structure ...
  test/   ... same structure ...
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class CloudReconstructionDataset(Dataset):
    MODALITIES = ["opt_cloudy", "opt_clean", "sar", "temporal", "dem", "mask"]

    def __init__(self, root, split="train"):
        self.split_dir = os.path.join(root, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Expected split directory {self.split_dir} to exist.")

        # Intersection check: only include patch IDs present across ALL 6 core modalities
        modality_sets = []
        for mod in self.MODALITIES:
            mod_dir = os.path.join(self.split_dir, mod)
            if os.path.isdir(mod_dir):
                pids = set(
                    os.path.splitext(os.path.basename(f))[0]
                    for f in glob.glob(os.path.join(mod_dir, "*.npy"))
                )
                modality_sets.append(pids)
            else:
                modality_sets.append(set())

        if modality_sets:
            valid_ids = set.intersection(*modality_sets)
            self.patch_ids = sorted(list(valid_ids))
        else:
            self.patch_ids = []

        if len(self.patch_ids) == 0:
            raise RuntimeError(f"No complete 6-modality .npy patch sets found in {self.split_dir}")

    def __len__(self):
        return len(self.patch_ids)

    def _load(self, modality, patch_id):
        path = os.path.join(self.split_dir, modality, f"{patch_id}.npy")
        return torch.from_numpy(np.load(path).astype(np.float32))

    @staticmethod
    def _to_tanh_range(x):
        return x * 2.0 - 1.0

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        opt_cloudy = self._to_tanh_range(self._load("opt_cloudy", pid))
        opt_clean = self._to_tanh_range(self._load("opt_clean", pid))
        sar = self._load("sar", pid)
        temporal = self._to_tanh_range(self._load("temporal", pid))
        dem = self._load("dem", pid)
        mask_continuous = self._load("mask", pid)  # shape (1, H, W) float32 in [0, 1]

        # Load discrete mask_class {0, 1, 2} if available, otherwise synthesize
        mask_class_path = os.path.join(self.split_dir, "mask_class", f"{pid}.npy")
        if os.path.exists(mask_class_path):
            mc_raw = np.load(mask_class_path).astype(np.int64)
            mask_class = torch.from_numpy(mc_raw).squeeze(0) if mc_raw.ndim == 3 else torch.from_numpy(mc_raw)
        else:
            mask_class = (mask_continuous.squeeze(0) > 0.40).long()

        return {
            "opt_cloudy": opt_cloudy,
            "opt_clean": opt_clean,
            "sar": sar,
            "temporal": temporal,
            "dem": dem,
            "mask": mask_continuous,
            "mask_class": mask_class,
            "patch_id": pid,
        }


class SyntheticCloudDataset(Dataset):
    """
    Generates shape-correct synthetic (cloudy, clean, mask) triples on the fly.
    Used for pretraining and smoke-testing.
    """

    def __init__(self, n_samples=64, patch_size=256, c_opt=3, c_sar=2, c_temp=3, c_dem=4,
                 severity=0.6, seed=0):
        self.n_samples = n_samples
        self.patch_size = patch_size
        self.c_opt = c_opt
        self.c_sar = c_sar
        self.c_temp = c_temp
        self.c_dem = c_dem
        self.severity = severity
        self.base_seed = seed

    def __len__(self):
        return self.n_samples

    def set_severity(self, severity):
        """Ramps up curriculum difficulty."""
        self.severity = float(np.clip(severity, 0.0, 1.0))

    def _value_noise(self, rng, h, w, octave):
        grid = rng.random((octave, octave)).astype(np.float32)
        grid_t = torch.from_numpy(grid)[None, None]
        up = torch.nn.functional.interpolate(
            grid_t, size=(h, w), mode="bicubic", align_corners=False
        )
        return up[0, 0].numpy()

    def _fractal_cloud_mask(self, rng, h, w):
        mask = np.zeros((h, w), dtype=np.float32)
        octaves = [3, 6, 12, 24]
        for octave in octaves:
            mask += self._value_noise(rng, h, w, octave) / len(octaves)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        thresh = 0.75 - 0.35 * self.severity
        soft = 0.15 - 0.10 * self.severity
        cloud = np.clip((mask - thresh) / max(soft, 1e-3), 0, 1)
        return cloud, mask

    def _shadow_from_cloud(self, cloud_opacity, sun_azimuth_deg, offset_px):
        h, w = cloud_opacity.shape
        rad = np.deg2rad(sun_azimuth_deg)
        dy, dx = int(round(offset_px * np.cos(rad))), int(round(offset_px * np.sin(rad)))
        shadow = np.roll(cloud_opacity, shift=(dy, dx), axis=(0, 1))
        if dy > 0:
            shadow[:dy, :] = 0
        elif dy < 0:
            shadow[dy:, :] = 0
        if dx > 0:
            shadow[:, :dx] = 0
        elif dx < 0:
            shadow[:, dx:] = 0
        return shadow

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.base_seed * 100000 + idx)
        h = w = self.patch_size

        clean = np.stack(
            [self._value_noise(rng, h, w, 8) * 0.4 + 0.25 for _ in range(self.c_opt)], axis=0
        ).astype(np.float32)

        cloud_opacity, _ = self._fractal_cloud_mask(rng, h, w)
        sun_azimuth = rng.uniform(0, 360)
        offset_px = rng.uniform(4, 14) * (0.4 + 0.6 * self.severity)
        shadow_opacity = self._shadow_from_cloud(cloud_opacity, sun_azimuth, offset_px)
        shadow_opacity = np.clip(shadow_opacity - cloud_opacity, 0, 1)

        cloud_brightness = rng.uniform(0.7, 0.95)
        shadow_darkness = rng.uniform(0.3, 0.6)

        cloudy = clean.copy()
        cloudy = cloudy * (1 - cloud_opacity) + cloud_brightness * cloud_opacity
        cloudy = cloudy * (1 - shadow_opacity * (1 - shadow_darkness))
        cloudy = np.clip(cloudy, 0, 1).astype(np.float32)

        mask_class = np.zeros((h, w), dtype=np.int64)
        mask_class[cloud_opacity > 0.40] = 1
        mask_class[(shadow_opacity > 0.20) & (mask_class == 0)] = 2
        mask_continuous = cloud_opacity.astype(np.float32)[None, :, :]

        sar = np.stack([self._value_noise(rng, h, w, 16) for _ in range(self.c_sar)], axis=0).astype(np.float32)
        temporal = np.clip(clean + rng.normal(0, 0.03, clean.shape), 0, 1).astype(np.float32)
        dem = np.stack([self._value_noise(rng, h, w, 6) for _ in range(self.c_dem)], axis=0).astype(np.float32)
        dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)

        def to_tanh(x):
            return torch.from_numpy(x) * 2.0 - 1.0

        return {
            "opt_cloudy": to_tanh(cloudy),
            "opt_clean": to_tanh(clean),
            "sar": torch.from_numpy(sar),
            "temporal": to_tanh(temporal),
            "dem": torch.from_numpy(dem),
            "mask": torch.from_numpy(mask_continuous),
            "mask_class": torch.from_numpy(mask_class),
            "patch_id": f"synthetic_{idx:04d}",
        }


if __name__ == "__main__":
    ds = SyntheticCloudDataset(n_samples=4, patch_size=128, severity=0.7)
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape), v.dtype, f"min={v.min():.3f} max={v.max():.3f}")
        else:
            print(k, v)
    print("OK")
