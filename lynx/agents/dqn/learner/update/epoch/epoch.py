from typing import Any, Tuple

import jax
import optax
from omegaconf import DictConfig

from lynx.agents.dqn.learner.update.epoch.loss import create_batch_loss_fn
from lynx.types.types import OnlineAndTarget, Transition


def create_update_epoch_fn(
    apply_fn,
    update_fn: optax.TransformUpdateFn,
    buffer,
    config: DictConfig,
):
    def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
        """Update the network for a single epoch."""
        # SAMPLE TRANSITIONS
        params, opt_states, buffer_state, key = update_state
        key, sample_key = jax.random.split(key)
        transition_sample = buffer.sample(buffer_state, sample_key)
        transitions: Transition = transition_sample.experience

        _q_loss_fn = create_batch_loss_fn(apply_fn, config)

        # CALCULATE Q LOSS
        q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(
            params.online,
            params.target,
            transitions,
        )

        # Compute the parallel mean (pmean) over the batch.
        # This calculation is inspired by the Anakin architecture demo notebook.
        # available at https://tinyurl.com/26tdzs5x
        # This pmean could be a regular mean as the batch axis is on the same device.
        q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
        q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

        # UPDATE Q PARAMS AND OPTIMISER STATE
        q_updates, q_new_opt_state = update_fn(q_grads, opt_states)
        q_new_online_params = optax.apply_updates(params.online, q_updates)
        # Target network polyak update.
        new_target_q_params = optax.incremental_update(
            q_new_online_params, params.target, config.train.hparams.tau
        )
        q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)

        # PACK NEW PARAMS AND OPTIMISER STATE
        new_params = q_new_params
        new_opt_state = q_new_opt_state

        # PACK LOSS INFO
        loss_info = {
            **q_loss_info,
        }
        return (new_params, new_opt_state, buffer_state, key), loss_info

    return _update_epoch
