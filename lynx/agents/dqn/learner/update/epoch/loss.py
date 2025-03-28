import chex
import jax.numpy as jnp
import rlax
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig

from lynx.types.types import Transition


def create_batch_loss_fn(
    q_apply_fn,
    config: DictConfig,
):
    def _q_loss_fn(
        q_params: FrozenDict,
        target_q_params: FrozenDict,
        transitions: Transition,
    ) -> jnp.ndarray:
        q_tm1 = q_apply_fn(q_params, transitions.obs).preferences
        q_t = q_apply_fn(target_q_params, transitions.next_obs).preferences

        # Cast and clip rewards.
        discount = 1.0 - transitions.done.astype(jnp.float32)
        d_t = (discount * config.train.hparams.gamma).astype(jnp.float32)
        r_t = jnp.clip(
            transitions.reward,
            -config.train.hparams.max_abs_reward,
            config.train.hparams.max_abs_reward,
        ).astype(jnp.float32)
        a_tm1 = transitions.action

        # Compute Q-learning loss.
        batch_loss = _q_learning(
            q_tm1,
            a_tm1,
            r_t,
            d_t,
            q_t,
            config.train.hparams.huber_loss_parameter,
        )

        loss_info = {
            "q_loss": batch_loss,
        }

        return batch_loss, loss_info

    return _q_loss_fn


def _q_learning(
    q_tm1: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_t: chex.Array,
    huber_loss_parameter: chex.Array,
) -> jnp.ndarray:
    """Computes the double Q-learning loss. Each input is a batch."""
    batch_indices = jnp.arange(a_tm1.shape[0])
    # Compute Q-learning n-step TD-error.
    target_tm1 = r_t + d_t * jnp.max(q_t, axis=-1)
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)

    return jnp.mean(batch_loss)
