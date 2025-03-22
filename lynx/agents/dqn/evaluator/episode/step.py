import jax
import jax.numpy as jnp

from lynx.types.types import EvalState


def create_env_step(env, act_fn, params):
    def _env_step(eval_state: EvalState) -> EvalState:
        """Step the environment."""
        # PRNG keys.
        key, env_state, last_timestep, step_count, episode_return = eval_state

        # Select action.
        key, policy_key = jax.random.split(key)

        action = act_fn(
            params,
            jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, ...], last_timestep.observation
            ),
            policy_key,
        )

        # Step environment.
        env_state, timestep = env.step(env_state, action.squeeze())

        # Log episode metrics.
        episode_return += timestep.reward
        step_count += 1
        eval_state = EvalState(key, env_state, timestep, step_count, episode_return)
        return eval_state

    return _env_step
