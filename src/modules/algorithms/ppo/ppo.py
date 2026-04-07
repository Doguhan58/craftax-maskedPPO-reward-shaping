"""
Reference: https://github.com/Reytuag/transformerXL_PPO_JAX
Reference: https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py
"""
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
import wandb

from src.arguments import Config
from src.modules.algorithms.base import BaseAlgorithm
from src.modules.algorithms.shared.types import Transition, CustomTrainState
from src.modules.algorithms.shared.memory import MemoryManager
from src.modules.algorithms.shared.rollout import sample_action
from src.modules.algorithms.shared.training_utils import (
    shuffle_batch, split_into_minibatches, build_segment_mask,
    segment_rollout_for_window_grad, accumulate_grads, average_grads,
    split_minibatches_for_accumulation, update_avg_return_ema)

from src.modules.algorithms.shared.validation import run_eval_rollout, create_eval_env
from src.modules.algorithms.ppo.gae import compute_gae, normalize_advantages
from src.modules.algorithms.ppo.losses import compute_ppo_loss
from src.modules.algorithms.ppo.training_utils import (
    prepare_batch_transformer, compute_metrics,
)
from craftax.craftax_env import make_craftax_env_from_name

class PPOAlgorithm(BaseAlgorithm):
    name = "ppo"

    def __init__(self, config: Config, env: Any, network: Any):
        super().__init__(config, env, network)
        self.cfg = config.ppo
        self._setup_config()


    def _setup_config(self):
        self.num_envs = self.config.env.num_envs
        self.num_steps = self.config.train.num_steps
        self.num_minibatches = self.config.train.num_minibatches
        self.update_epochs = self.config.train.update_epochs
        self.num_updates = self.config.num_updates
        self.gradient_accumulation_steps = self.config.train.gradient_accumulation_steps
        self.window_grad = self.config.model.window_grad


    def _create_optimizer(self) -> optax.GradientTransformation:
        if self.config.train.lr_linear_decay:
            num_updates = self.num_updates
            num_minibatches = self.num_minibatches
            update_epochs = self.update_epochs
            base_lr = self.config.train.lr
            gradient_accumulation_steps = self.gradient_accumulation_steps

            def lr_schedule(count):
                effective_minibatches = num_minibatches // gradient_accumulation_steps
                frac = 1.0 - (count // (effective_minibatches * update_epochs)) / num_updates
                return base_lr * frac

            lr = lr_schedule
        else:
            lr = self.config.train.lr

        return optax.chain(optax.clip_by_global_norm(self.config.train.max_grad_norm), optax.adam(learning_rate=lr, eps=1e-5))


    def _init_network(self, rng: jnp.ndarray):
        env_params = self.env.default_params
        window_mem = self.config.model.memory_len
        num_layers = self.config.model.num_layers
        encoder_size = self.config.model.embed_dim
        num_heads = self.config.model.num_heads

        init_obs = jnp.zeros((2, *self.env.observation_space(env_params).shape))
        init_memory = jnp.zeros((2, window_mem, num_layers, encoder_size))
        init_mask = jnp.zeros((2, num_heads, 1, window_mem + 1), dtype=jnp.bool_)
        init_action = jnp.zeros((2,), dtype=jnp.int32)

        return self.network.init(rng, init_memory, init_obs, init_mask, init_action)


    def _create_train_state(self, rng: jnp.ndarray) -> CustomTrainState:
        rng, init_rng = jax.random.split(rng)
        network_variables = self._init_network(init_rng)
        tx = self._create_optimizer()

        return CustomTrainState.create(apply_fn=self.network.apply, params=network_variables.get("params", network_variables), tx=tx)


    def _setup_eval_env(self):
        if not self.config.validation.enabled:
            return None, None

        basic_env = make_craftax_env_from_name(self.config.env.name, auto_reset=True)
        eval_env = create_eval_env(self.config, basic_env)
        return eval_env, basic_env.default_params


    def make_train(self) -> Callable:
        config = self.config
        env = self.env
        network = self.network
        env_params = env.default_params

        eval_env, eval_env_params = self._setup_eval_env()

        memory_manager = MemoryManager(config, network)

        def train(rng: jnp.ndarray):
            train_state = self._create_train_state(rng)

            rng, env_rng = jax.random.split(rng)
            obsv, env_state = env.reset(env_rng, env_params)

            return self._train(train_state, env_state, obsv, rng, eval_env, eval_env_params, memory_manager)

        return train

    def _train(self, train_state: CustomTrainState, env_state: Any, obsv: jnp.ndarray, rng: jnp.ndarray, eval_env: Any, eval_env_params: Any, memory_manager: MemoryManager):
        config = self.config
        cfg = self.cfg
        env = self.env
        network = self.network
        env_params = env.default_params

        #config parameters
        num_envs = self.num_envs
        num_steps = self.num_steps
        num_minibatches = self.num_minibatches
        update_epochs = self.update_epochs
        num_updates = self.num_updates
        gamma = config.train.gamma
        gae_lambda = cfg.gae_lambda
        clip_eps = cfg.clip_eps
        vf_coef = cfg.vf_coef
        ent_coef = cfg.ent_coef
        normalize_adv = cfg.normalize_adv
        log_wandb = config.train.log_wandb
        total_timesteps = config.train.total_timesteps
        gradient_accumulation_steps = self.gradient_accumulation_steps

        #transformer config
        window_mem = config.model.memory_len
        window_grad = config.model.window_grad
        num_heads = config.model.num_heads
        num_segments = num_steps // window_grad
        assert num_steps % window_grad == 0

        #validation
        validation_enabled = config.validation.enabled
        validation_interval = (int(num_updates * config.validation.eval_interval_pct / 100.0) if config.validation.eval_interval_pct > 0 else 0)
        num_eval_envs = config.validation.num_eval_envs
        num_eval_steps = config.validation.num_eval_steps

        #action masking
        use_action_masking = config.train.use_action_masking
        num_actions = env.action_space(env_params).n


        def _env_step(runner_state, unused):
            (train_state_s, env_state_s, memory, last_obs, done, last_action, last_action_mask, step_env, rng_s, avg_ret_ema) = runner_state

            rng_s, action_rng, step_rng = jax.random.split(rng_s, 3)

            memory = memory_manager.reset_on_done(memory, done)

            #pass
            new_mem, logits, value = memory_manager.forward_eval(train_state_s.params, memory, last_obs, last_action)

            #apply action mask and sample
            current_mask = last_action_mask
            action, log_prob = sample_action(logits, action_rng, current_mask)

            #step
            obsv, env_state_new, reward, env_done, info = env.step(step_rng, env_state_s, action, env_params)

            next_last_action = jnp.where(env_done,
                                         jnp.zeros_like(action, dtype=jnp.int32), action.astype(jnp.int32))

            #extract new mask
            next_action_mask = info.get("action_mask", None) if use_action_masking else None

            transition = Transition(obs=jax.tree.map(lambda x: x.astype(jnp.bfloat16), last_obs),
                                    action=action, reward=reward, done=env_done, last_done=env_done,
                                    last_action=last_action, info=info, value=value, log_prob=log_prob,
                                    memories_mask=memory.mask, memories_indices=memory.mask_idx, action_mask=current_mask)

            new_runner_state = (train_state_s, env_state_new, new_mem, obsv, env_done, next_last_action, next_action_mask, step_env + num_envs, rng_s, avg_ret_ema)

            return new_runner_state, transition


        def _compute_segment_loss(params, segment_data, segment_mem, segment_mem_mask, ts):
            obs, action, done, log_prob, value, last_action, advantages, targets, seg_action_mask = segment_data

            obs_bt = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), obs)
            last_action_bt = jnp.swapaxes(last_action, 0, 1)

            train_mask = build_segment_mask(done, segment_mem_mask, window_mem, window_grad, num_heads)
            cached_mem = jax.lax.stop_gradient(segment_mem)

            logits, value_pred, transformer_out = network.apply({"params": params}, cached_mem, obs_bt, train_mask, last_action_bt,  method=network.model_forward_train_with_memory)

            logits = jnp.swapaxes(logits, 0, 1)
            value_pred = jnp.swapaxes(value_pred, 0, 1)

            total_loss, value_loss, actor_loss, entropy = compute_ppo_loss(logits, value_pred, action, log_prob, value,
                                                                           advantages, targets, clip_eps, vf_coef, ent_coef, normalize_adv, action_mask=seg_action_mask)

            return total_loss, (value_loss, actor_loss, entropy, transformer_out)


        def _compute_minibatch_grads(ts, batch_info):
            init_mem, per_step_masks, obs, action, done, last_done, log_prob, value, last_action, advantages, targets, mb_action_mask = batch_info

            (obs_seg, action_seg, done_seg, log_prob_seg, value_seg,
             last_action_seg, advantages_seg, targets_seg, per_step_masks_seg, action_mask_seg) = segment_rollout_for_window_grad(obs, action, done, log_prob, value, last_action,
                                                                                                                                  advantages, targets, per_step_masks, num_steps, window_grad, action_mask=mb_action_mask)

            def _loss_fn(params):
                def _segment_step(carry, segment_idx):
                    current_mem, current_mem_mask, total_loss_acc, vl_acc, al_acc, ent_acc = carry

                    seg_obs = jax.tree.map(lambda x: x[segment_idx], obs_seg)
                    seg_action = action_seg[segment_idx]
                    seg_done = done_seg[segment_idx]
                    seg_log_prob = log_prob_seg[segment_idx]
                    seg_value = value_seg[segment_idx]
                    seg_last_action = last_action_seg[segment_idx]
                    seg_advantages = advantages_seg[segment_idx]
                    seg_targets = targets_seg[segment_idx]
                    seg_am = action_mask_seg[segment_idx] if action_mask_seg is not None else None

                    segment_data = (seg_obs, seg_action, seg_done, seg_log_prob, seg_value, seg_last_action, seg_advantages, seg_targets, seg_am)
                    seg_mem_mask = per_step_masks_seg[segment_idx, 0]

                    seg_loss, (seg_vl, seg_al, seg_ent, transformer_out) = _compute_segment_loss( params, segment_data, current_mem, seg_mem_mask, ts)

                    new_mem = jnp.roll(current_mem, -window_grad, axis=1)
                    update = transformer_out.astype(new_mem.dtype)
                    new_mem = new_mem.at[:, window_mem - window_grad:, :, :].set(update[:, :, None, :])

                    next_mem_mask = per_step_masks_seg[segment_idx, -1]

                    return (new_mem, next_mem_mask, total_loss_acc + seg_loss, vl_acc + seg_vl, al_acc + seg_al, ent_acc + seg_ent), None

                init_mem_mask = per_step_masks_seg[0, 0]
                init_carry = (init_mem, init_mem_mask, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

                (_, _, total_loss, vl, al, ent), _ = jax.lax.scan(_segment_step, init_carry, jnp.arange(num_segments))

                return total_loss / num_segments, (vl / num_segments, al / num_segments, ent / num_segments)

            grads, (value_loss, actor_loss, entropy) = jax.grad(_loss_fn, has_aux=True)(ts.params)
            return grads, (jnp.mean(value_loss + actor_loss), value_loss, actor_loss, entropy)


        def _update_minibatch_with_accumulation(carry, minibatch_group):
            ts, accum_grads = carry

            def _accumulate_single(inner_carry, single_minibatch):
                inner_accum_grads, inner_loss_accum = inner_carry
                grads, (total_loss, value_loss, actor_loss, entropy) = _compute_minibatch_grads(ts, single_minibatch)

                if inner_accum_grads is None:
                    new_accum_grads = grads
                else:
                    new_accum_grads = accumulate_grads(inner_accum_grads, grads)

                new_loss_accum = (
                    inner_loss_accum[0] + total_loss,
                    inner_loss_accum[1] + value_loss,
                    inner_loss_accum[2] + actor_loss,
                    inner_loss_accum[3] + entropy)
                return (new_accum_grads, new_loss_accum), None

            init_loss_accum = (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
            (final_grads, final_loss_accum), _ = jax.lax.scan(_accumulate_single, (None, init_loss_accum), minibatch_group)

            avg_g = average_grads(final_grads, gradient_accumulation_steps)
            grad_norm = optax.global_norm(avg_g)
            ts = ts.apply_gradients(grads=avg_g)

            avg_loss = ( final_loss_accum[0] / gradient_accumulation_steps, final_loss_accum[1] / gradient_accumulation_steps,
                         final_loss_accum[2] / gradient_accumulation_steps, final_loss_accum[3] / gradient_accumulation_steps, grad_norm)

            return (ts, None), avg_loss

        def _update_minibatch_no_accumulation(ts, batch_info):
            grads, (total_loss, value_loss, actor_loss, entropy) = _compute_minibatch_grads(ts, batch_info)
            grad_norm = optax.global_norm(grads)
            ts = ts.apply_gradients(grads=grads)
            return ts, (total_loss, value_loss, actor_loss, entropy, grad_norm)


        def _update_epoch(update_state, unused):
            ts, traj_batch, advantages, targets, rng_e, init_mem, per_step_masks = update_state

            rng_e, shuffle_rng = jax.random.split(rng_e)
            permutation = jax.random.permutation(shuffle_rng, num_envs)

            batch_traj = (per_step_masks, traj_batch.obs, traj_batch.action, traj_batch.done,
                          traj_batch.last_done, traj_batch.log_prob, traj_batch.value,
                          traj_batch.last_action, advantages, targets, traj_batch.action_mask)

            shuffled_traj = shuffle_batch(batch_traj, permutation)
            minibatches_traj = split_into_minibatches(shuffled_traj, num_minibatches)

            shuffled_mem = jnp.take(init_mem, permutation, axis=0)
            mem_shape = shuffled_mem.shape
            mb_size = mem_shape[0] // num_minibatches
            minibatches_mem = jnp.reshape(shuffled_mem, (num_minibatches, mb_size) + mem_shape[1:])

            minibatches = (minibatches_mem,) + minibatches_traj

            if gradient_accumulation_steps > 1:
                minibatches_grouped = split_minibatches_for_accumulation(minibatches, gradient_accumulation_steps)
                (ts, _), loss_info = jax.lax.scan(_update_minibatch_with_accumulation, (ts, None), minibatches_grouped)
            else:
                ts, loss_info = jax.lax.scan(_update_minibatch_no_accumulation, ts, minibatches)

            return (ts, traj_batch, advantages, targets, rng_e, init_mem, per_step_masks), loss_info


        def _run_validation(ts, rng_v):
            if eval_env is None:
                return {
                    "eval/episode_return_mean": jnp.array(0.0),
                    "eval/episode_return_max": jnp.array(0.0),
                    "eval/episode_return_min": jnp.array(0.0),
                    "eval/episode_length_mean": jnp.array(0.0),
                    "eval/total_episodes": jnp.array(0.0),
                    "eval/total_reward": jnp.array(0.0)
                }
            return run_eval_rollout(ts, eval_env, eval_env_params, rng_v, num_eval_envs, num_eval_steps, memory_manager, use_action_masking=use_action_masking, action_dim=num_actions)


        def _update_step(runner_state, update_step_idx):
            (train_state_u, env_state_u, mem,
             last_obs, done, last_action, last_action_mask,
             step_env, rng_u, avg_ret_ema) = runner_state

            #save memory
            start_mem = mem

            #rollout
            env_runner_state = runner_state
            env_runner_state, traj_batch = jax.lax.scan(_env_step, env_runner_state, None, num_steps)

            (train_state_u, env_state_u, mem, last_obs, done, last_action, last_action_mask, step_env, rng_u, avg_ret_ema) = env_runner_state

            #last value gae
            _, _, last_val = memory_manager.forward_eval(train_state_u.params, mem, last_obs, last_action)

            train_state_u = train_state_u.replace(timesteps=train_state_u.timesteps + num_steps * num_envs)

            advantages, targets = compute_gae(traj_batch, last_val, gamma, gae_lambda)

            # Normalize advantages
            if normalize_adv:
                advantages = normalize_advantages(advantages)

            #prepare transformer batch
            batch_data = prepare_batch_transformer(traj_batch, advantages, targets, start_mem.data)

            init_mem_batch, per_step_masks_batch = batch_data[0], batch_data[1]

            update_state = (train_state_u, traj_batch, advantages, targets, rng_u, init_mem_batch, per_step_masks_batch)

            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, update_epochs)
            train_state_u = update_state[0]
            rng_u = update_state[4]

            metric = compute_metrics(traj_batch, train_state_u, total_timesteps)

            total_loss, value_loss, actor_loss, entropy, grad_norm = loss_info
            metric["total_loss"] = jnp.mean(total_loss)
            metric["value_loss"] = jnp.mean(value_loss)
            metric["actor_loss"] = jnp.mean(actor_loss)
            metric["entropy"] = jnp.mean(entropy)
            metric["grad_norm"] = jnp.mean(grad_norm)

            #action mask metrics
            if use_action_masking and traj_batch.action_mask is not None:
                mask_sum = traj_batch.action_mask.sum(axis=-1)  #[T, B]
                metric["action_mask_valid_mean"] = jnp.mean(mask_sum)
                metric["action_mask_valid_min"] = jnp.min(mask_sum)
                metric["action_mask_valid_max"] = jnp.max(mask_sum)

            avg_ret = metric.get("returned_episode_returns", jnp.array(0.0)).astype(jnp.float32)
            avg_ret_ema = update_avg_return_ema(avg_ret_ema, avg_ret)

            train_state_u = train_state_u.replace(n_updates=train_state_u.n_updates + 1)

            #validation
            if validation_enabled and validation_interval > 0:
                should_validate = (train_state_u.n_updates % validation_interval) == 0
                rng_u, rng_eval = jax.random.split(rng_u)

                eval_metrics = jax.lax.cond( should_validate,
                                             lambda _: _run_validation(train_state_u, rng_eval), lambda _: eval_zeros, None)
                metric.update(eval_metrics)

            if log_wandb:
                def callback(metric, step):
                    wandb.log(metric, step=int(step))
                jax.debug.callback(callback, metric, metric["update_steps"])

            new_runner_state = (train_state_u, env_state_u, mem, last_obs, done, last_action, last_action_mask, step_env, rng_u, avg_ret_ema)
            return new_runner_state, metric


        rng, init_rng = jax.random.split(rng)

        mem = memory_manager.init_memory(num_envs)

        #initial action mask, all valid for first step (correct mask arrives after first step)
        if use_action_masking:
            init_action_mask = jnp.ones((num_envs, num_actions), dtype=jnp.bool_)
        else:
            init_action_mask = None

        runner_state = (train_state, env_state, mem, obsv, jnp.zeros((num_envs,), dtype=bool),
                        jnp.zeros((num_envs,), dtype=jnp.int32), init_action_mask, 0, init_rng, jnp.zeros((), dtype=jnp.float32))

        #build a zeros template matching _run_validation output so jax.lax.cond branches match
        if validation_enabled and validation_interval > 0:
            eval_template = jax.eval_shape(_run_validation, train_state, rng)
            eval_zeros = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), eval_template)

        runner_state, metric = jax.lax.scan(_update_step, runner_state, jnp.arange(num_updates))

        return {"runner_state": runner_state, "metrics": metric}
