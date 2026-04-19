import os
import random
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EnvConfig:
    name: str = "Craftax-Symbolic-v1"
    num_envs: int = 1024
    use_optimistic_resets: bool = True
    reset_ratio: int = 16

    def __post_init__(self):
        if self.use_optimistic_resets:
            assert self.num_envs % self.reset_ratio == 0


@dataclass
class TrainConfig:
    total_timesteps: int = 1_000_000_000
    seed: int = field(default_factory=lambda: random.randint(1, 10000))
    lr: float = 3e-4
    lr_linear_decay: bool = True
    max_grad_norm: float = 1.0
    gamma: float = 0.999
    num_steps: int = 128
    num_minibatches: int = 8
    update_epochs: int = 4
    log_wandb: bool = True
    wandb_project: str = "craftax"
    wandb_entity: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    use_remat: bool = True
    gradient_accumulation_steps: int = 1
    use_action_masking: bool = True


@dataclass
class ValidationConfig:
    enabled: bool = False
    eval_interval_pct: float = 5.0
    num_eval_envs: int = 16
    num_eval_steps: int = 20000
    log_to_wandb: bool = True


@dataclass
class PPOConfig:
    gae_lambda: float = 0.8
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    normalize_adv: bool = True
    add_last_action: bool = True


@dataclass
class ModelConfig:
    hidden_dim: int = 512
    num_layers: int = 2

    embed_dim: int = 256
    num_heads: int = 8
    head_dim: int = 64  # qkv_features = num_heads * head_dim
    mlp_ratio: int = 1
    dropout: float = 0.0
    gru_bias: float = 2.0
    use_remat: bool = True

    memory_len: int = 256
    window_grad: int = 128  # > 1


@dataclass
class CheckpointConfig:
    save_checkpoints: bool = False
    save_interval: int = 1000000
    max_to_keep: int = 1
    save_final: bool = True


@dataclass
class RewardShapingConfig:
    enabled: bool = True
    w_kill: float = 1.0
    w_iron: float = 0.5
    w_diamond: float = 1.0
    w_coal: float = 0.3
    w_wood: float = 0.1
    w_stone: float = 0.01
    w_armor: float = 3.0
    w_levelup: float = 1.5  #(STR/DEX/INT)
    w_sapphire: float = 1.0
    w_ruby: float = 1.0
    max_shaping_per_step: float = 8.0
    w_floor_clear: float = 3.0

    #depth PBRS: phi(s) = w_depth * player_level, F = depth_gamma * phi(s') - phi(s)
    w_depth: float = 2.0
    depth_gamma: float = 0.999

    #floor-scaled kills: effective = w_kill * (1 + kill_floor_scale * level)
    kill_floor_scale: float = 0.25


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)


    @property
    def checkpoint_dir(self):
        return self.checkpoint.checkpoint_dir if hasattr(self.checkpoint, 'checkpoint_dir') else self.train.checkpoint_dir


    @property
    def num_updates(self) -> int:
        return self.train.total_timesteps // (self.train.num_steps * self.env.num_envs)


    @property
    def minibatch_size(self) -> int:
        return (self.env.num_envs * self.train.num_steps) // self.train.num_minibatches


    @property
    def effective_minibatch_size(self) -> int:
        return self.minibatch_size // self.train.gradient_accumulation_steps


    @property
    def update_steps_per_epoch(self) -> int:
        return self.train.num_minibatches * self.train.gradient_accumulation_steps


    def get_run_name(self) -> str:
        parts = ["ppo", "gtrxl"]
        parts.append(str(self.train.seed))

        return "_".join(parts)


    def get_wandb_config(self) -> Dict[str, Any]:
        config_dict = {
            "algorithm": "ppo",
            "backend": "transformer",
            "env_name": self.env.name,
            "num_envs": self.env.num_envs,
            "use_optimistic_resets": self.env.use_optimistic_resets,
            "reset_ratio": self.env.reset_ratio,

            #training
            "total_timesteps": self.train.total_timesteps,
            "num_updates": self.num_updates,
            "seed": self.train.seed,
            "lr": self.train.lr,
            "lr_linear_decay": self.train.lr_linear_decay,
            "max_grad_norm": self.train.max_grad_norm,
            "gamma": self.train.gamma,
            "num_steps": self.train.num_steps,
            "num_minibatches": self.train.num_minibatches,
            "update_epochs": self.train.update_epochs,
            "use_remat": self.train.use_remat,
            "gradient_accumulation_steps": self.train.gradient_accumulation_steps,
            "use_action_masking": self.train.use_action_masking,

            #validation
            "validation_enabled": self.validation.enabled,
            "validation_interval_pct": self.validation.eval_interval_pct,
            "validation_num_eval_envs": self.validation.num_eval_envs,
            "validation_num_eval_steps": self.validation.num_eval_steps,

            #model(transformer)
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "embed_dim": self.model.embed_dim,
            "num_heads": self.model.num_heads,
            "head_dim": self.model.head_dim,
            "mlp_ratio": self.model.mlp_ratio,
            "dropout": self.model.dropout,
            "gru_bias": self.model.gru_bias,
            "memory_len": self.model.memory_len,
            "window_grad": self.model.window_grad,

            #ppo
            "gae_lambda": self.ppo.gae_lambda,
            "clip_eps": self.ppo.clip_eps,
            "vf_coef": self.ppo.vf_coef,
            "ent_coef": self.ppo.ent_coef,
            "normalize_adv": self.ppo.normalize_adv,
            "add_last_action": self.ppo.add_last_action,

            #reward shaping
            "reward_shaping_enabled": self.reward_shaping.enabled,
            "reward_shaping_w_kill": self.reward_shaping.w_kill,
            "reward_shaping_w_iron": self.reward_shaping.w_iron,
            "reward_shaping_w_diamond": self.reward_shaping.w_diamond,
            "reward_shaping_w_coal": self.reward_shaping.w_coal,
            "reward_shaping_w_wood": self.reward_shaping.w_wood,
            "reward_shaping_w_stone": self.reward_shaping.w_stone,
            "reward_shaping_w_armor": self.reward_shaping.w_armor,
            "reward_shaping_w_levelup": self.reward_shaping.w_levelup,
            "reward_shaping_max_per_step": self.reward_shaping.max_shaping_per_step,
            "reward_shaping_w_depth": self.reward_shaping.w_depth,
            "reward_shaping_depth_gamma": self.reward_shaping.depth_gamma,
            "reward_shaping_kill_floor_scale": self.reward_shaping.kill_floor_scale,
            "reward_shaping_w_sapphire": self.reward_shaping.w_sapphire,
            "reward_shaping_w_ruby": self.reward_shaping.w_ruby,
            "reward_shaping_w_floor_clear": self.reward_shaping.w_floor_clear}

        return config_dict


    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "algorithm": "ppo",
            "backend": "transformer",
            "env_name": self.env.name,
            "seed": self.train.seed,

            #architecture
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "embed_dim": self.model.embed_dim,
            "num_heads": self.model.num_heads,
            "head_dim": self.model.head_dim,
            "mlp_ratio": self.model.mlp_ratio,
            "dropout": self.model.dropout,
            "gru_bias": self.model.gru_bias,
            "memory_len": self.model.memory_len,
            "window_grad": self.model.window_grad,
            "use_remat": self.train.use_remat,
            "add_last_action": self.ppo.add_last_action,
            "use_action_masking": self.train.use_action_masking}


    @staticmethod
    def from_checkpoint_metadata(metadata: Dict[str, Any]) -> 'Config':
        config = Config()
        config.env.name = metadata.get("env_name", "Craftax-Symbolic-v1")
        config.train.seed = metadata.get("seed", 0)

        #model architecture
        config.model.hidden_dim = metadata.get("hidden_dim", 512)
        config.model.num_layers = metadata.get("num_layers", 2)
        config.model.embed_dim = metadata.get("embed_dim", 256)
        config.model.num_heads = metadata.get("num_heads", 8)
        config.model.head_dim = metadata.get("head_dim", 64)
        config.model.mlp_ratio = metadata.get("mlp_ratio", 1)
        config.model.dropout = metadata.get("dropout", 0.0)
        config.model.gru_bias = metadata.get("gru_bias", 2.0)
        config.model.memory_len = metadata.get("memory_len", 256)
        config.model.window_grad = metadata.get("window_grad", 128)
        config.train.use_remat = metadata.get("use_remat", True)
        config.ppo.add_last_action = metadata.get("add_last_action", True)
        config.train.use_action_masking = metadata.get("use_action_masking", True)

        return config

    def save_metadata(self, path: str):
        metadata = self.get_checkpoint_metadata()
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_metadata(path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return json.load(f)
