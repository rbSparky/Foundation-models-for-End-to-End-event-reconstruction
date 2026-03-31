import os
from typing import Dict, List, Optional, Tuple, OrderedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .dataloader import read_file


FEATURE_NAMES = ["pT", "eta", "phi", "energy"]


def _stable_reco_target(masked_targets: np.ndarray) -> np.ndarray:
    """Convert [pT, eta, phi, E] -> [log1p(pT), eta, sin(phi), cos(phi), log1p(E)]."""
    p_t = np.log1p(np.clip(masked_targets[:, 0], a_min=0.0, a_max=None))
    eta = masked_targets[:, 1]
    phi = masked_targets[:, 2]
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    energy = np.log1p(np.clip(masked_targets[:, 3], a_min=0.0, a_max=None))

    return np.stack([p_t, eta, sin_phi, cos_phi, energy], axis=1).astype(np.float32)


def _mask_particle(particles: np.ndarray, mode: str = "random") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_idx = np.where(np.any(particles != 0, axis=1))[0]
    if valid_idx.size == 0:
        # degenerate padded event, keep behavior deterministic
        mask_idx = np.array([0], dtype=np.int64)
    elif mode == "random":
        mask_idx = np.array([np.random.choice(valid_idx)], dtype=np.int64)
    elif mode == "biased":
        # Keep baseline behavior: lower indices are more likely.
        total = np.sum(1 / (np.arange(0, particles.shape[0]) + 1))
        idx = particles.shape[0] - 1
        u, w = 0.0, 1.0
        while (u < w) or (idx not in valid_idx):
            u = np.random.uniform(0, 1)
            idx = np.random.randint(0, particles.shape[0])
            w = (1 / (idx + 1)) / total
        mask_idx = np.array([idx], dtype=np.int64)
    elif mode == "pt_high":
        # Curriculum-friendly option: prefer higher-pT particles among valid entries.
        weights = np.clip(particles[valid_idx, 0], a_min=1e-6, a_max=None)
        weights = weights / weights.sum()
        mask_idx = np.array([np.random.choice(valid_idx, p=weights)], dtype=np.int64)
    elif mode == "first":
        mask_idx = valid_idx[:1].astype(np.int64)
    else:
        mask_idx = np.array([np.random.choice(valid_idx)], dtype=np.int64)

    masked_particles = particles.copy()
    masked_targets = masked_particles[mask_idx, :].copy()
    masked_particles[mask_idx, :] = 0.0

    return masked_particles, masked_targets, mask_idx


class JetClassDataset(Dataset):
    """In-memory JetClass dataset used for both classification and MAE pretraining."""

    def __init__(
        self,
        X_particles: np.ndarray,
        y: np.ndarray,
        normalize: List[bool] = [True, False, False, True],
        norm_dict: Dict[str, Tuple[float, float]] = None,
        mask_mode: Optional[str] = None,
        target_mode: str = "raw",
    ):
        super().__init__()
        self.X_particles = X_particles
        self.y = y
        self.normalize = normalize
        self.norm_dict = norm_dict
        self.mask_mode = mask_mode
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.X_particles)

    def _apply_norm_inplace(self, arr: np.ndarray) -> None:
        if self.norm_dict is None:
            return

        for i, feature in enumerate(FEATURE_NAMES):
            if not self.normalize[i]:
                continue
            mean, std = self.norm_dict[feature]
            if i in (0, 3):
                arr[:, i] = arr[:, i] / mean
            else:
                arr[:, i] = (arr[:, i] - mean) / std

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        particles = self.X_particles[idx].T.copy()

        if self.mask_mode is not None:
            masked_particles, masked_targets, mask_idx = _mask_particle(particles, self.mask_mode)
            self._apply_norm_inplace(masked_particles)

            if self.target_mode == "stable":
                masked_targets_out = _stable_reco_target(masked_targets)
            else:
                self._apply_norm_inplace(masked_targets)
                masked_targets_out = masked_targets.astype(np.float32)

            return (
                torch.tensor(masked_particles, dtype=torch.float32),
                torch.tensor(masked_targets_out.squeeze(0), dtype=torch.float32),
                torch.tensor(mask_idx, dtype=torch.int64),
            )

        self._apply_norm_inplace(particles)
        return torch.from_numpy(particles).float(), torch.from_numpy(self.y[idx]).float()


class JetClassSubsetDataset(Dataset):
    """Memmapped 100k subset dataset driven by a split NPZ file."""

    def __init__(
        self,
        split_npz: str,
        split: str,
        normalize: List[bool] = [True, False, False, True],
        norm_dict: Dict[str, Tuple[float, float]] = None,
        mask_mode: Optional[str] = None,
        target_mode: str = "raw",
    ):
        super().__init__()
        self.split_npz = split_npz
        self.split = split
        self.normalize = normalize
        self.norm_dict = norm_dict
        self.mask_mode = mask_mode
        self.target_mode = target_mode

        split_obj = np.load(split_npz, allow_pickle=True)
        key = f"{split}_indices"
        if key not in split_obj:
            raise KeyError(f"Missing key '{key}' in {split_npz}")

        self.indices = split_obj[key].astype(np.int64)
        split_dir = os.path.dirname(os.path.abspath(split_npz))

        particles_file = split_obj.get("particles_file", np.array(["jetclass_100k_particles.npy"]))
        labels_file = split_obj.get("labels_file", np.array(["jetclass_100k_labels.npy"]))

        particles_file = str(np.array(particles_file).reshape(-1)[0])
        labels_file = str(np.array(labels_file).reshape(-1)[0])

        self.particles = np.load(os.path.join(split_dir, particles_file), mmap_mode="r")
        self.labels = np.load(os.path.join(split_dir, labels_file), mmap_mode="r")

        if self.particles.shape[0] < self.indices.max() + 1:
            raise ValueError("Split indices exceed particle cache size")

    def __len__(self) -> int:
        return len(self.indices)

    def _apply_norm_inplace(self, arr: np.ndarray) -> None:
        if self.norm_dict is None:
            return

        for i, feature in enumerate(FEATURE_NAMES):
            if not self.normalize[i]:
                continue
            mean, std = self.norm_dict[feature]
            if i in (0, 3):
                arr[:, i] = arr[:, i] / mean
            else:
                arr[:, i] = (arr[:, i] - mean) / std

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        row_idx = int(self.indices[idx])
        particles = self.particles[row_idx].T.astype(np.float32, copy=True)

        if self.mask_mode is not None:
            masked_particles, masked_targets, mask_idx = _mask_particle(particles, self.mask_mode)
            self._apply_norm_inplace(masked_particles)

            if self.target_mode == "stable":
                masked_targets_out = _stable_reco_target(masked_targets)
            else:
                self._apply_norm_inplace(masked_targets)
                masked_targets_out = masked_targets.astype(np.float32)

            return (
                torch.tensor(masked_particles, dtype=torch.float32),
                torch.tensor(masked_targets_out.squeeze(0), dtype=torch.float32),
                torch.tensor(mask_idx, dtype=torch.int64),
            )

        self._apply_norm_inplace(particles)
        label = self.labels[row_idx].astype(np.float32, copy=True)
        return torch.from_numpy(particles).float(), torch.from_numpy(label).float()


class LazyJetClassDataset(JetClassDataset):
    """File-backed fallback dataset for full JetClass directory training."""

    def __init__(
        self,
        data_dir: str,
        normalize: List[bool] = [True, False, False, True],
        norm_dict: Dict[str, Tuple[float, float]] = None,
        mask_mode: Optional[str] = None,
        cache_size: int = 10,
        target_mode: str = "raw",
    ):
        super().__init__(None, None, normalize, norm_dict, mask_mode, target_mode)
        self.files = sorted(
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".root")
        )
        files_per_class = max(len(self.files) // 10, 1)
        self.files_by_class = [
            list(range(i * files_per_class, min((i + 1) * files_per_class, len(self.files))))
            for i in range(10)
        ]
        self.events_per_file = 100_000

        self._cache_size = int(cache_size)
        self._cache = OrderedDict()

    def __len__(self) -> int:
        return len(self.files) * self.events_per_file

    def _get_file(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx in self._cache:
            particles, labels = self._cache.pop(idx)
            self._cache[idx] = (particles, labels)
            return particles, labels

        particles, _, labels = read_file(self.files[idx])
        self._cache[idx] = (particles, labels)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return particles, labels

    def __getitem__(self, key):
        if isinstance(key, tuple):
            file_idx, event_idx = int(key[0]), int(key[1])
        else:
            key = int(key)
            file_idx, event_idx = key // self.events_per_file, key % self.events_per_file

        particles, labels = self._get_file(file_idx)
        part = particles[event_idx].T.copy()

        if self.mask_mode is not None:
            masked_particles, masked_targets, mask_idx = _mask_particle(part, self.mask_mode)
            self._apply_norm_inplace(masked_particles)
            if self.target_mode == "stable":
                masked_targets_out = _stable_reco_target(masked_targets)
            else:
                self._apply_norm_inplace(masked_targets)
                masked_targets_out = masked_targets.astype(np.float32)

            return (
                torch.tensor(masked_particles, dtype=torch.float32),
                torch.tensor(masked_targets_out.squeeze(0), dtype=torch.float32),
                torch.tensor(mask_idx, dtype=torch.int64),
            )

        self._apply_norm_inplace(part)
        return torch.from_numpy(part).float(), torch.from_numpy(labels[event_idx]).float()
