import chex
import jax
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig

from lynx.agents.dqn.learner.warmup.step import get_step_env_fn
from lynx.types.types import OnlineAndTarget


def setup_warmup_fn(
    env: Environment,
    apply_fn,
    params: OnlineAndTarget,
    buffer,
    config: DictConfig,
):
    def warmup(
        env_states,
        timesteps: TimeStep,
        buffer_states,
        key: chex.PRNGKey,
    ):
        step_env = get_step_env_fn(env, apply_fn, params)

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (env_states, timesteps, keys), traj_batch = jax.lax.scan(
            step_env, (env_states, timesteps, key), None, config.train.warmup_steps
        )

        # Add the trajectory to the buffer.
        buffer_states = buffer.add(buffer_states, traj_batch)
        return env_states, timesteps, keys, buffer_states

    batch_warmup_step = jax.vmap(
        warmup, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0), axis_name="batch"
    )
    return batch_warmup_step
