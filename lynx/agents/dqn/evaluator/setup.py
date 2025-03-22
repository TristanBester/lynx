import chex
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from lynx.agents.dqn.evaluator.evaluator import create_evaluator


def setup_evaluator(eval_env, apply_fn, key, config):
    """Setup the evaluator."""
    act_fn = _get_distribution_act_fn(apply_fn)
    param_fn = _reshape_params

    # Accepts distributed parameters and normal key (runs VMAP internally)
    evaluator = create_evaluator(eval_env, param_fn, act_fn, config)
    evaluator = jax.pmap(evaluator, axis_name="device")

    # These are keys which can be used with the evaluator
    key, *eval_keys = jax.random.split(key, jax.device_count() + 1)
    eval_keys = jnp.stack(eval_keys)
    eval_keys = eval_keys.reshape(jax.device_count(), -1)
    return evaluator, eval_keys


def _get_distribution_act_fn(apply_fn):
    def act_fn(
        params: FrozenDict, observation: chex.Array, key: chex.PRNGKey
    ) -> chex.Array:
        """Get the action from the distribution."""
        pi = apply_fn(params, observation)
        action = pi.mode()
        return action

    return act_fn


def _reshape_params(x):
    """Reshape the parameters to be used in the evaluator."""
    return jax.tree_util.tree_map(lambda x: x[0, ...], x)  # type: ignore
