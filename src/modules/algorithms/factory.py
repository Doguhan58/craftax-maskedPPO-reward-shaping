from typing import Any

from src.modules.algorithms.ppo import PPOAlgorithm
from src.modules.algorithms.base import BaseAlgorithm


def create_algorithm(config, env: Any, network: Any) -> BaseAlgorithm:
    return PPOAlgorithm(config, env, network)
