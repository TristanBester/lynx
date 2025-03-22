from typing import Any

import jax
from jumanji.env import Environment

from lynx.types.types import OnlineAndTarget, Transition


def get_step_env_fn(env: Environment, apply_fn, params: OnlineAndTarget):
    def _env_step(carry, _: Any):
        env_state, last_timestep, key = carry

        # SELECT ACTION
        key, policy_key = jax.random.split(key)
        actor_policy = apply_fn(params.online, last_timestep.observation)
        action = actor_policy.sample(seed=policy_key)

        # STEP ENVIRONMENT
        env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

        # LOG EPISODE METRICS
        done = timestep.last().reshape(-1)
        info = timestep.extras["episode_metrics"]
        next_obs = timestep.extras["next_obs"]

        transition = Transition(
            last_timestep.observation, action, timestep.reward, done, next_obs, info
        )
        return (env_state, timestep, key), transition

    return _env_step
