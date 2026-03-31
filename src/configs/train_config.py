from dataclasses import dataclass, field, fields


@dataclass
class TrainConfig:
    batch_size: int = 64
    criterion: dict = field(default_factory=lambda: {"name": "cross_entropy_loss", "kwargs": {}})
    optimizer: dict = field(default_factory=lambda: {"name": "adam", "kwargs": {"lr": 1e-4}})
    optimizer_wrapper: dict = None
    scheduler: dict = None
    callbacks: list = None
    num_epochs: int = 20
    start_epoch: int = 0
    logging_dir: str = "logs"
    logging_steps: int = 500
    progress_bar: bool = True
    save_best: bool = True
    best_monitor: str = "val_loss"
    best_mode: str = "min"
    save_ckpt: bool = True
    save_fig: bool = False
    device: str = None
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42
    run_name: str = ""
    profiling_json: str = ""

    @classmethod
    def from_dict(cls, d: dict):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
