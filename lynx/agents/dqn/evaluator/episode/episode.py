import jax
import jax.numpy as jnp

from lynx.agents.dqn.evaluator.episode.cond import create_cond_fn
from lynx.agents.dqn.evaluator.episode.step import create_env_step
from lynx.types.types import EvalState


def create_eval_one_episode(env, act_fn):
    def eval_one_episode(params, key):
        step_env = create_env_step(env, act_fn, params)
        cond_fn = create_cond_fn()

        key, subkey = jax.random.split(key)
        env_state, timestep = env.reset(subkey)
        init_eval_state = EvalState(
            env_state=env_state,
            timestep=timestep,
            episode_return=jnp.zeros((), dtype=jnp.float32),
            step_count=jnp.zeros((), dtype=jnp.int32),
            key=key,
        )

        final_state = jax.lax.while_loop(cond_fn, step_env, init_eval_state)

        eval_statistics = {
            "episode_return": final_state.episode_return,
            "episode_length": final_state.step_count,
        }
        return eval_statistics

    return eval_one_episode
