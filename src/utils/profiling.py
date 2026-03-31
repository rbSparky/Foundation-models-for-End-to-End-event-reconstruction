import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class EpochProfile:
    epoch: int
    train_events: int
    train_steps: int
    wall_time_sec: float
    events_per_sec: float
    peak_gpu_memory_bytes: int


@dataclass
class RunProfiler:
    output_path: str
    metadata: Dict[str, object] = field(default_factory=dict)
    epochs: List[EpochProfile] = field(default_factory=list)
    run_start: float = field(default_factory=time.perf_counter)

    def begin_epoch(self, device: torch.device) -> float:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def end_epoch(
        self,
        epoch: int,
        start_time: float,
        train_events: int,
        train_steps: int,
        device: torch.device,
    ) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
        else:
            peak = 0

        elapsed = max(time.perf_counter() - start_time, 1e-8)
        eps = float(train_events) / elapsed
        self.epochs.append(
            EpochProfile(
                epoch=epoch,
                train_events=int(train_events),
                train_steps=int(train_steps),
                wall_time_sec=float(elapsed),
                events_per_sec=float(eps),
                peak_gpu_memory_bytes=peak,
            )
        )

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        total_time = float(time.perf_counter() - self.run_start)
        payload = {
            "metadata": self.metadata,
            "total_time_sec": total_time,
            "epochs": [
                {
                    "epoch": ep.epoch,
                    "train_events": ep.train_events,
                    "train_steps": ep.train_steps,
                    "wall_time_sec": ep.wall_time_sec,
                    "events_per_sec": ep.events_per_sec,
                    "peak_gpu_memory_bytes": ep.peak_gpu_memory_bytes,
                }
                for ep in self.epochs
            ],
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def maybe_save(self, rank: int) -> None:
        if rank == 0:
            self.save()
