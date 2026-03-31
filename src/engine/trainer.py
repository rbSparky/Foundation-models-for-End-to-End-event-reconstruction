import csv
import os
import random
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributed import get_rank, get_world_size, is_available, is_initialized
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from ..configs import TrainConfig
from ..loss import LOSS_REGISTRY
from ..optim import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from ..utils import (
    CALLBACK_REGISTRY,
    RunProfiler,
    get_callbacks_from_config,
    get_loss_from_config,
    get_optim_from_config,
    get_optim_wrapper_from_config,
    get_scheduler_from_config,
)
from ..utils.data import JetClassDistributedSampler


class Trainer:
    """Base trainer with dataloading, checkpointing and profiling utilities."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        device: Optional[Union[torch.device, int]] = None,
        metric: Optional[Callable] = None,
        config: Optional[TrainConfig] = None,
        batch_size: Optional[int] = None,
        criterion: Optional[Dict] = None,
        optimizer: Optional[Dict] = None,
        optimizer_wrapper: Optional[Dict] = None,
        scheduler: Optional[Dict] = None,
        callbacks: Optional[List[Dict]] = None,
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = None,
        logging_dir: Optional[str] = None,
        logging_steps: Optional[int] = None,
        progress_bar: Optional[bool] = None,
        save_best: Optional[bool] = None,
        save_ckpt: Optional[bool] = None,
        save_fig: Optional[bool] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        seed: Optional[int] = None,
    ):
        self.rank = 0
        self.world_size = 1
        if is_available() and is_initialized():
            self.rank = get_rank()
            self.world_size = get_world_size()

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank}")
        else:
            self.device = torch.device("cpu")

        self._is_distributed = self.world_size > 1
        self.model = model.to(self.device)
        if self._is_distributed:
            self.model = DDP(
                module=self.model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )

        if config is not None:
            self.batch_size = batch_size if batch_size is not None else config.batch_size
            self.criterion = get_loss_from_config(criterion if criterion is not None else config.criterion, LOSS_REGISTRY)
            self.optimizer = get_optim_from_config(
                optimizer if optimizer is not None else config.optimizer,
                OPTIM_REGISTRY,
                self.model,
            )
            if optimizer_wrapper is not None or config.optimizer_wrapper is not None:
                self.optimizer = get_optim_wrapper_from_config(
                    optimizer_wrapper if optimizer_wrapper is not None else config.optimizer_wrapper,
                    OPTIM_REGISTRY,
                    self.optimizer,
                )
            self.scheduler = get_scheduler_from_config(
                scheduler if scheduler is not None else config.scheduler,
                SCHEDULER_REGISTRY,
                self.optimizer,
            )
            self.callbacks = get_callbacks_from_config(
                callbacks if callbacks is not None else (config.callbacks or []),
                CALLBACK_REGISTRY,
            )
            self.num_epochs = num_epochs if num_epochs is not None else config.num_epochs
            self.start_epoch = start_epoch if start_epoch is not None else config.start_epoch
            self.logging_dir = logging_dir if logging_dir is not None else config.logging_dir
            self.logging_steps = logging_steps if logging_steps is not None else config.logging_steps
            self.progress_bar = progress_bar if progress_bar is not None else config.progress_bar
            self.save_best = save_best if save_best is not None else config.save_best
            self.save_ckpt = save_ckpt if save_ckpt is not None else config.save_ckpt
            self.save_fig = save_fig if save_fig is not None else config.save_fig
            self.num_workers = num_workers if num_workers is not None else config.num_workers
            self.pin_memory = pin_memory if pin_memory is not None else config.pin_memory
            self.seed = int(seed if seed is not None else config.seed)
            self._requested_run_name = config.run_name
            self._requested_profiling_json = config.profiling_json
        else:
            if criterion is None or optimizer is None:
                raise ValueError("criterion and optimizer are required when TrainConfig is not provided")
            self.batch_size = batch_size if batch_size is not None else 64
            self.criterion = get_loss_from_config(criterion, LOSS_REGISTRY)
            self.optimizer = get_optim_from_config(optimizer, OPTIM_REGISTRY, self.model)
            if optimizer_wrapper is not None:
                self.optimizer = get_optim_wrapper_from_config(optimizer_wrapper, OPTIM_REGISTRY, self.optimizer)
            self.scheduler = (
                get_scheduler_from_config(scheduler, SCHEDULER_REGISTRY, self.optimizer)
                if scheduler
                else None
            )
            self.callbacks = get_callbacks_from_config(callbacks or [], CALLBACK_REGISTRY)
            self.num_epochs = num_epochs if num_epochs is not None else 20
            self.start_epoch = start_epoch if start_epoch is not None else 0
            self.logging_dir = logging_dir if logging_dir is not None else "logs"
            self.logging_steps = logging_steps if logging_steps is not None else 500
            self.progress_bar = progress_bar if progress_bar is not None else True
            self.save_best = save_best if save_best is not None else True
            self.save_ckpt = save_ckpt if save_ckpt is not None else True
            self.save_fig = save_fig if save_fig is not None else False
            self.num_workers = num_workers if num_workers is not None else 0
            self.pin_memory = pin_memory if pin_memory is not None else False
            self.seed = int(seed if seed is not None else 42)
            self._requested_run_name = ""
            self._requested_profiling_json = ""

        self._loader_generator = torch.Generator()
        self._loader_generator.manual_seed(self.seed + self.rank)

        def _seed_worker(worker_id: int):
            worker_seed = self.seed + worker_id + 1000 * self.rank
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        train_sampler = (
            JetClassDistributedSampler(
                files_by_class=train_dataset.files_by_class,
                events_per_file=train_dataset.events_per_file,
                batch_size=self.batch_size,
                rank=self.rank,
                world_size=self.world_size,
                shuffle_files=True,
                seed=self.seed,
            )
            if self._is_distributed
            else None
        )
        val_sampler = (
            JetClassDistributedSampler(
                files_by_class=val_dataset.files_by_class,
                events_per_file=val_dataset.events_per_file,
                batch_size=self.batch_size,
                rank=self.rank,
                world_size=self.world_size,
                shuffle_files=False,
                seed=self.seed,
            )
            if self._is_distributed
            else None
        )
        test_sampler = (
            JetClassDistributedSampler(
                files_by_class=test_dataset.files_by_class,
                events_per_file=test_dataset.events_per_file,
                batch_size=self.batch_size,
                rank=self.rank,
                world_size=self.world_size,
                shuffle_files=False,
                seed=self.seed,
            )
            if (test_dataset is not None and self._is_distributed)
            else None
        )

        if self._is_distributed:
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_sampler=train_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=_seed_worker,
                generator=self._loader_generator,
            )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=_seed_worker,
                generator=self._loader_generator,
            )
            self.test_loader = (
                DataLoader(
                    dataset=test_dataset,
                    batch_sampler=test_sampler,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    worker_init_fn=_seed_worker,
                    generator=self._loader_generator,
                )
                if test_dataset is not None
                else None
            )
        else:
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=_seed_worker,
                generator=self._loader_generator,
            )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=_seed_worker,
                generator=self._loader_generator,
            )
            self.test_loader = (
                DataLoader(
                    dataset=test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    worker_init_fn=_seed_worker,
                    generator=self._loader_generator,
                )
                if test_dataset is not None
                else None
            )

        self.metric = metric
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_metric": [],
            "val_loss": [],
            "val_metric": [],
        }
        self.best_monitor = "val_loss"
        self.best_mode = "min"
        if config is not None:
            self.best_monitor = getattr(config, "best_monitor", "val_loss")
            self.best_mode = getattr(config, "best_mode", "min")
        if self.best_mode not in {"min", "max"}:
            raise ValueError(f"Unsupported best_mode='{self.best_mode}', expected 'min' or 'max'")
        self.best_score = float("inf") if self.best_mode == "min" else -float("inf")

        self.model_name = self.model.module.__class__.__name__ if isinstance(self.model, DDP) else self.model.__class__.__name__
        os.makedirs(self.logging_dir, exist_ok=True)
        self.log_dir = os.path.join(self.logging_dir, self.model_name)
        self.best_models_dir = os.path.join(self.log_dir, "best")
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        self.loggings_dir = os.path.join(self.log_dir, "logging")
        self.outputs_dir = os.path.join(self.log_dir, "output")
        for d in [self.log_dir, self.best_models_dir, self.checkpoints_dir, self.loggings_dir, self.outputs_dir]:
            os.makedirs(d, exist_ok=True)

        self.run_name = self._requested_run_name or self._get_next_run_index()
        self._log_header_written = False
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

        profiling_path = self._requested_profiling_json or os.path.join(self.outputs_dir, f"{self.run_name}_profiling.json")
        self.profiler = RunProfiler(
            output_path=profiling_path,
            metadata={
                "run_name": self.run_name,
                "model_name": self.model_name,
                "seed": self.seed,
                "batch_size": self.batch_size,
                "world_size": self.world_size,
                "num_epochs": self.num_epochs,
            },
        )

    def _get_next_run_index(self) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"pid{os.getpid()}_{ts}"

    def _set_logging_paths(self, run_name: str):
        self.run_name = run_name
        self._log_header_written = True
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

    def save_checkpoint(self, epoch: int):
        if not self.checkpoint_path or self.rank != 0:
            return

        model_state = self.model.module.state_dict() if self._is_distributed else self.model.state_dict()
        checkpoint = {
            "run_name": self.run_name,
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "history": self.history,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._set_logging_paths(checkpoint["run_name"])
        target = self.model.module if self._is_distributed else self.model
        target.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.history = checkpoint["history"]

    def load_best_model(self, best_model_path: str):
        run_name = os.path.splitext(os.path.basename(best_model_path))[0]
        self._set_logging_paths(run_name)
        target = self.model.module if self._is_distributed else self.model
        target.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def log_csv(self, log_dict: Dict[str, float]):
        if self.rank != 0:
            return

        write_header = not self._log_header_written
        with open(self.logging_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
            if write_header:
                writer.writeheader()
                self._log_header_written = True
            writer.writerow(log_dict)

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        raise NotImplementedError

    def _is_better(self, current: float) -> bool:
        if self.best_mode == "min":
            return current < self.best_score
        return current > self.best_score
