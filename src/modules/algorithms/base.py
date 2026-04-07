from abc import ABC, abstractmethod
from typing import Any


class BaseAlgorithm(ABC):
    name: str = "base"

    def __init__(self, config: Any, env: Any, network: Any):
        self.config = config
        self.env = env
        self.network = network

    @abstractmethod
    def make_train(self):
        pass