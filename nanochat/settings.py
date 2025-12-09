"""
Settings classes for nanochat training and configuration.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any
from pathlib import Path
import tomllib

from nanochat.common import autodetect_device_type


class TrainSettings(BaseSettings):
    # wandb run name default ("dummy" is special - we won't log to wandb)
    run: str = "dummy"
    # Runtime
    # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
    device_type: str = ""
    # Model architecture
    # the depth of the Transformer model to train, rest of the kwargs are derived
    depth: int = 20
    # max context length
    max_seq_len: int = 2048
    # Type of linear approximation: "svd" or "abba"
    approx_type: str = "svd"
    # Whether to use low-rank MLP weights approximation
    approx_mlp_proj: bool = True
    # Rank for low-rank MLP weights approximation
    mlp_proj_rank: int = 16
    # Whether to build model layer-by-layer incrementally
    build_by_layer: bool = True
    # Whether new blocks copy weights from previous layer
    copy_block_weights: bool = True
    # Whether to freeze previous blocks during training
    freeze_previous_weights: bool = False
    # Whether to use Linformer attention
    use_linformer: bool = False
    # Projection dimension k for Linformer
    linformer_proj_dim: int = 128
    # Parameter sharing strategy for Linformer projections
    linformer_sharing: str = "layerwise"
    # Training horizon. Only one of these 3 will be used, in this order of precedence.
    # explicit number of steps of the optimization (-1 = disable)
    num_iterations: int = -1
    # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
    target_flops: float = -1.0
    # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
    target_param_data_ratio: int = 20
    # Optimization
    # per-device batch size (set to not OOM)
    device_batch_size: int = 32
    # total desired batch size, in #tokens
    total_batch_size: int = 524288
    # learning rate for the embedding parameters (Adam)
    embedding_lr: float = 0.2
    # learning rate for the unembedding parameters (Adam)
    unembedding_lr: float = 0.004
    # weight decay for the embedding/unembedding parameters (Adam)
    weight_decay: float = 0.0
    # learning rate for the matrix parameters (Muon)
    matrix_lr: float = 0.02
    # gradient clipping value (0.0 = disabled)
    grad_clip: float = 1.0
    # ratio of iterations for LR warmup
    warmup_ratio: float = 0.0
    # ratio of iterations for LR warmdown
    warmdown_ratio: float = 0.2
    # final LR is this fraction of the initial LR
    final_lr_frac: float = 0.0
    # resume training from this step of the optimization (-1 = disable)
    resume_from_step: int = -1
    # Evaluation
    # every how many steps to evaluate the model for val bpb
    eval_every: int = 250
    # number of tokens to evaluate val loss on
    eval_tokens: int = 20 * 524288
    # every how many steps to evaluate the core metric (-1 = disable)
    core_metric_every: int = 2000
    # examples per task in estimating the core metric
    core_metric_max_per_task: int = 500
    # every how many steps to sample from the model
    sample_every: int = 2000
    # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
    save_every: int = -1
    # Output
    # optionally override the model tag for the output checkpoint directory name
    model_tag: str = ""

    model_config = SettingsConfigDict(env_prefix="TRAIN_")

    @classmethod
    def load_or_create(cls, config_path: str = "train_config.toml"):
        if Path(config_path).exists():
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
            return cls(**config_data)
        else:
            # Generate default config file
            with open(config_path, "w") as f:
                f.write("# Training Configuration\n")
                settings = cls()

                for name, field in cls.model_fields.items():
                    value = getattr(settings, name)
                    if isinstance(value, str):
                        f.write(f'{name} = "{value}"\n')
                    else:
                        f.write(f"{name} = {value}\n")
            print(f"Created default configuration file at {config_path}")
            return settings

    def model_post_init(self, __context: Any) -> None:
        if self.device_type == "":
            self.device_type = autodetect_device_type()
