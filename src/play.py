import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import distrax
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from craftax.craftax.renderer import render_craftax_pixels
from flax.training import checkpoints
from src.arguments import Config
from src.modules.environments.craftax_env import CraftaxEnvironment
from src.modules.model.factory import create_network
from src.modules.constants import action_id_to_name
import glob as glob_mod

@dataclass
class PlayConfig:
    checkpoint_path: str = "checkpoints/ppo_gtrxl_"
    deterministic: bool = False
    render_fps: int = 10
    screen_width: int = 768
    screen_height: int = 768
    font_size: int = 24
    show_info: bool = True
    record_video: bool = True
    video_dir: str = "videos"
    max_episodes: int = 100


class CraftaxPlayer:
    def __init__(self, config: PlayConfig):
        self.config = config
        self.render_fps = config.render_fps

        self.train_config = self._load_train_config(config.checkpoint_path)
        self.add_last_action = self.train_config.ppo.add_last_action

        self.train_config.env.num_envs = 1
        self.train_config.env.use_optimistic_resets = False
        self.use_action_masking = self.train_config.train.use_action_masking
        self.action_mask = None
        self.env_wrapper = CraftaxEnvironment(self.train_config)

        self.network = create_network(self.train_config, self.env_wrapper.action_dim)
        self.params = self._load_checkpoint(config.checkpoint_path)

        pygame.init()
        self.screen_size = (config.screen_width, config.screen_height)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Craftax")
        self.clock = pygame.time.Clock()
        base_font_size = int(config.font_size * 0.8)
        self.font = pygame.font.Font(None, base_font_size)
        self.action_font = pygame.font.Font(None, max(10, int(base_font_size * 0.8)))

        self.total_episodes = 0
        self.episode_returns = []

        self.key = jax.random.PRNGKey(0)
        self.last_actions = deque(maxlen=10)

        self.video_writer = None
        self.video_path = None

        self.reset()
        self._init_video_writer()


    def _load_train_config(self, checkpoint_path: str) -> Config:
        checkpoint_path = os.path.abspath(checkpoint_path)
        metadata_path = os.path.join(checkpoint_path, "metadata.json")

        if os.path.exists(metadata_path):
            print(f"Loading config from metadata: {metadata_path}")
            metadata = Config.load_metadata(metadata_path)
            config = Config.from_checkpoint_metadata(metadata)
            print(f"Loaded model: PPO + GTrXL")
            return config
        else:
            print(f"No metadata found at {metadata_path}, using default config")
            return Config()


    def _init_memory(self) -> jnp.ndarray:
        return jnp.zeros((1, self.train_config.model.memory_len, self.train_config.model.num_layers, self.train_config.model.embed_dim), dtype=jnp.bfloat16)


    def _init_mask(self) -> jnp.ndarray:
        window_mem = self.train_config.model.memory_len
        num_heads = self.train_config.model.num_heads

        return jnp.zeros((1, num_heads, 1, window_mem + 1), dtype=jnp.bool_)


    def _init_video_writer(self):
        self.frame_size = (self.screen_size[1], self.screen_size[0])

        if not self.config.record_video:
            self.video_writer = None
            self.video_path = None
            return


        os.makedirs(self.config.video_dir, exist_ok=True)
        filename = time.strftime("craftax_%Y%m%d_%H%M%S.mp4")
        self.video_path = os.path.join(self.config.video_dir, filename)

        self.video_writer = imageio.get_writer(self.video_path, fps=self.render_fps, macro_block_size=None)
        print(f"Recording gameplay to: {self.video_path}")


    def _write_frame(self):
        if self.video_writer is None:
            return

        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))

        h, w, _ = frame.shape
        target_h, target_w = self.frame_size

        if (h, w) != (target_h, target_w):
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surface = pygame.transform.scale(surface, self.screen_size)
            frame = pygame.surfarray.array3d(surface)
            frame = np.transpose(frame, (1, 0, 2))

        self.video_writer.append_data(frame)


    def _close_video_writer(self):
        if self.video_writer is not None:
            print(f"Closing video file: {self.video_path}")
            self.video_writer.close()
            self.video_writer = None


    def _load_checkpoint(self, checkpoint_path: str) -> Any:
        checkpoint_path = os.path.abspath(checkpoint_path)

        dummy_obs = jnp.zeros((1, *self.env_wrapper.observation_shape))
        init_key = jax.random.PRNGKey(0)

        dummy_mem = jnp.zeros((1, self.train_config.model.memory_len, self.train_config.model.num_layers, self.train_config.model.embed_dim), dtype=jnp.bfloat16)
        dummy_mask = jnp.zeros((1, self.train_config.model.num_heads, 1, self.train_config.model.memory_len + 1), dtype=jnp.bool_)
        dummy_last_action = jnp.zeros((1,), dtype=jnp.int32)

        variables = self.network.init(init_key, dummy_mem, dummy_obs, dummy_mask, dummy_last_action)
        dummy_params = variables['params']
        target = {"params": dummy_params}


        for prefix in ("checkpoint_final_", "checkpoint_"):
            try:
                checkpoint = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=target, step=None, prefix=prefix)

                if checkpoint is not None and "params" in checkpoint:
                    pattern = os.path.join(checkpoint_path, f"{prefix}*")

                    if glob_mod.glob(pattern):
                        print(f"Loaded checkpoint from {checkpoint_path} (prefix={prefix!r})")

                        return checkpoint["params"]
            except Exception:
                continue

        raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}. " f"No files matching checkpoint_final_* or checkpoint_* found.")


    def _get_variables(self) -> dict:
        return {'params': self.params}


    def reset(self):
        self.key, reset_key = jax.random.split(self.key)
        self.obs, self.env_state = self.env_wrapper.reset(reset_key)
        self.episode_return = 0.0
        self.episode_length = 0
        self.last_actions.clear()
        self.memory = self._init_memory()
        self.mask = self._init_mask()
        self.memories_mask_idx = jnp.full((1,), self.train_config.model.memory_len + 1, dtype=jnp.int32)
        self.last_action = jnp.zeros((1,), dtype=jnp.int32)
        self.action_mask = None


    def select_action(self):
        obs_for_net = self.obs
        if obs_for_net.ndim == 1:
            obs_for_net = obs_for_net[None, :]
        obs_for_net = obs_for_net.astype(jnp.bfloat16)
        variables = self._get_variables()


        #update attention mask
        window_mem = self.train_config.model.memory_len
        num_heads = self.train_config.model.num_heads
        self.memories_mask_idx = jnp.clip(self.memories_mask_idx - 1, 0, window_mem)
        indices = jnp.arange(window_mem + 1)[None, None, None, :]
        mask_threshold = self.memories_mask_idx[:, None, None, None]
        self.mask = jnp.broadcast_to((indices >= mask_threshold).astype(jnp.bool_), (1, num_heads, 1, window_mem + 1))

        logits, value, new_mem = self.network.apply(variables, self.memory, obs_for_net, self.mask, self.last_action, method=self.network.model_forward_eval)
        self.memory = self._update_transformer_memory(self.memory, new_mem)

        if self.use_action_masking and self.action_mask is not None:
            logits = jnp.where(self.action_mask, logits, -1e9)


        if self.config.deterministic:
            action = jnp.argmax(logits, axis=-1)
        else:
            self.key, action_key = jax.random.split(self.key)
            dist = distrax.Categorical(logits=logits)
            action = dist.sample(seed=action_key)

        return action


    def _update_transformer_memory(self, old_mem: jnp.ndarray, new_mem: jnp.ndarray) -> jnp.ndarray:
        mem_len = self.train_config.model.memory_len
        new_mem = new_mem[:, None, :, :].astype(old_mem.dtype)
        combined = jnp.concatenate([old_mem, new_mem], axis=1)
        return combined[:, -mem_len:, :, :]


    def step(self):
        action = self.select_action()

        action = jnp.atleast_1d(jnp.squeeze(action))
        action_int = int(action[0].item())
        self.last_actions.append(action_int)
        self.last_action = action

        self.key, step_key = jax.random.split(self.key)
        self.obs, self.env_state, reward, done, info = self.env_wrapper.step(step_key, self.env_state, action)

        if self.use_action_masking:
            self.action_mask = info.get("action_mask", None)

        self.episode_return += float(reward[0])
        self.episode_length += 1

        if done[0]:
            self.total_episodes += 1
            self.episode_returns.append(float(self.episode_return))
            print(f"Episode {self.total_episodes}: " f"Return={self.episode_return:.2f}," f" Length={self.episode_length}")

            if self.total_episodes >= self.config.max_episodes:
                return False

            self.reset()

        return True


    def render(self):
        inner_state = self.env_state
        while hasattr(inner_state, 'env_state'):
            inner_state = inner_state.env_state

        state_to_render = jax.tree.map(lambda x: x[0], inner_state)
        pixel_obs = render_craftax_pixels(state_to_render, block_pixel_size=64)
        obs_image = np.array(pixel_obs)

        if obs_image.shape[-1] == 1:
            obs_image = np.repeat(obs_image, 3, axis=-1)

        if obs_image.max() <= 1.0:
            obs_image = (obs_image * 255).astype(np.uint8)
        else:
            obs_image = obs_image.astype(np.uint8)

        surface = pygame.surfarray.make_surface(obs_image.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, self.screen_size)

        self.screen.blit(surface, (0, 0))

        if self.config.show_info:
            self._draw_info()

        pygame.display.flip()
        self._write_frame()


    def _draw_info(self):
        info_texts = [f"Episode: {self.total_episodes}",
                      f"Return: {self.episode_return:.2f}",
                      f"Length: {self.episode_length}",
                      f"FPS: {self.clock.get_fps():.1f}", f"Algorithm: PPO + GTrXL"]


        if self.episode_returns:
            avg_return = np.mean(self.episode_returns[-min(100, len(self.episode_returns)):])
            info_texts.append(f"Avg Return : {avg_return:.2f}")

        y_offset = 10

        for text in info_texts:
            text_surface = self.font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect()
            text_rect.topleft = (10, y_offset)

            bg_rect = text_rect.copy()
            bg_rect.inflate_ip(10, 5)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)

            self.screen.blit(text_surface, text_rect)
            y_offset += self.font.get_linesize() + 4

        y_offset += 8

        title_surface = self.font.render("Last actions (10):", True, (255, 255, 255))
        title_rect = title_surface.get_rect()
        title_rect.topleft = (10, y_offset)

        title_bg_rect = title_rect.copy()
        title_bg_rect.inflate_ip(10, 5)
        pygame.draw.rect(self.screen, (0, 0, 0), title_bg_rect)

        self.screen.blit(title_surface, title_rect)
        y_offset += self.font.get_linesize() + 4


        if self.last_actions:
            for a in self.last_actions:
                action_text = f"{a} ({action_id_to_name(a)})"
                action_surface = self.action_font.render(action_text, True, (255, 255, 255))
                action_rect = action_surface.get_rect()
                action_rect.topleft = (20, y_offset)

                action_bg_rect = action_rect.copy()
                action_bg_rect.inflate_ip(10, 5)
                pygame.draw.rect(self.screen, (0, 0, 0), action_bg_rect)

                self.screen.blit(action_surface, action_rect)
                y_offset += self.action_font.get_linesize() + 2
        else:
            action_surface = self.action_font.render("N/A", True, (255, 255, 255))
            action_rect = action_surface.get_rect()
            action_rect.topleft = (20, y_offset)

            action_bg_rect = action_rect.copy()
            action_bg_rect.inflate_ip(10, 5)
            pygame.draw.rect(self.screen, (0, 0, 0), action_bg_rect)

            self.screen.blit(action_surface, action_rect)


    def run(self):
        running = True

        print("Controls:")
        print("ESC -> Quit")
        print("R -> Reset episode")
        print("D -> Toggle deterministic actions")
        print("I -> Toggle info display")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                        print("Environment reset")
                    elif event.key == pygame.K_d:
                        self.config.deterministic = not self.config.deterministic
                        print(f"Deterministic: {self.config.deterministic}")
                    elif event.key == pygame.K_i:
                        self.config.show_info = not self.config.show_info

            if not self.step():
                running = False
            else:
                self.render()
                self.clock.tick(self.render_fps)

        self._close_video_writer()
        pygame.quit()

        if self.episode_returns:
            print("\nFinal Statistics:")
            print(f"Total Episodes: {self.total_episodes}")
            print(f"Mean Return: {np.mean(self.episode_returns):.2f}")
            print(f"Std Return: {np.std(self.episode_returns):.2f}")
            print(f"Max Return: {np.max(self.episode_returns):.2f}")


def main():
    config = PlayConfig()
    player = CraftaxPlayer(config)
    player.run()


if __name__ == "__main__":
    main()
