import jax
import jax.numpy as jnp
from typing import Tuple, Any
from craftax.craftax_env import make_craftax_env_from_name
from src.arguments import Config
from src.modules.environments.wrappers import LogWrapper, OptimisticResetVecEnvWrapper, BatchEnvWrapper
from src.modules.environments.action_mask_wrapper import ActionMaskWrapper
from src.modules.environments.reward_shaping_wrapper import RewardShapingWrapper

class CraftaxEnvironment:
    def __init__(self, config: Config):
        self.config = config
        self.num_envs = config.env.num_envs
        self.env, self.env_params = self._build_env()

        self.observation_shape = self.env.observation_space(self.env_params).shape
        self.action_dim = self.env.action_space(self.env_params).n


    def _build_env(self):
        basic_env = make_craftax_env_from_name(self.config.env.name, not self.config.env.use_optimistic_resets)
        env_params = basic_env.default_params

        #action masking is before LogWrapper so it operates on raw EnvState, gets vmapped by BatchEnvWrapper
        if self.config.train.use_action_masking:
            basic_env = ActionMaskWrapper(basic_env)

        #log
        env = LogWrapper(basic_env)

        #batching
        if self.config.env.use_optimistic_resets:
            env = OptimisticResetVecEnvWrapper(env, num_envs=self.config.env.num_envs, reset_ratio=min(self.config.env.reset_ratio, self.config.env.num_envs))
        else:
            env = BatchEnvWrapper(env, num_envs=self.config.env.num_envs)

        #reward shaping
        if self.config.reward_shaping.enabled:
            env = RewardShapingWrapper(env, num_envs=self.config.env.num_envs, config=self.config.reward_shaping)

        return env, env_params


    def reset(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Any]:
        return self.env.reset(key, self.env_params)


    def step(self, key: jax.random.PRNGKey, state: Any, action: jnp.ndarray) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, dict]:
        return self.env.step(key, state, action, self.env_params)
