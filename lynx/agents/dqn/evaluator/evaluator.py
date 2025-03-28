import jax
import jax.numpy as jnp

from lynx.agents.dqn.evaluator.episode import create_eval_one_episode


def create_evaluator(env, param_fn, act_fn, config):
    def evaluate(params, key):
        """Evaluate the agent.

        params: Distributed parameters, device + batch dimension.
        key: Normal PRNG key, shape (2,).
        """
        # Fix shapes
        params = param_fn(params)

        # Create the evaluator
        eval_one_episode = create_eval_one_episode(env, act_fn)
        eval_one_episode = jax.vmap(
            eval_one_episode,
            in_axes=(None, 0),
            axis_name="batch",
        )

        eval_batch = (config.train.eval.num_episodes // jax.device_count()) * 1
        eval_keys = jnp.stack(jax.random.split(key, eval_batch))
        eval_keys = eval_keys.reshape(eval_batch, -1)

        eval_statistics = eval_one_episode(params, eval_keys)
        return eval_statistics

    return evaluate
