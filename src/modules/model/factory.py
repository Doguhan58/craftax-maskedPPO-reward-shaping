from typing import Any
from src.arguments import Config
from src.modules.model.networks import ActorCriticTransformer


def create_network(config: Config, action_dim: int) -> Any:
    mlp_dim = config.model.embed_dim * config.model.mlp_ratio

    return ActorCriticTransformer( action_dim=action_dim, embed_dim=config.model.embed_dim,
                                   hidden_dim=config.model.hidden_dim, num_heads=config.model.num_heads,
                                   head_dim=config.model.head_dim, num_layers=config.model.num_layers,
                                   mlp_dim=mlp_dim, dropout_rate=config.model.dropout,
                                   gru_bias=config.model.gru_bias, use_remat=config.train.use_remat,
                                   add_last_action=config.ppo.add_last_action)