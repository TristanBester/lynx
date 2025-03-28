import time

import jax
import jax.numpy as jnp

from lynx.agents.dqn.evaluator import setup_evaluator
from lynx.agents.dqn.learner.setup import setup_learner
from lynx.envs.factories.factory import make
from lynx.logger.logger import LogAggregator, StatisticType


def run_experiment(config, checkpointer):
    key = jax.random.PRNGKey(config.train.config.seed)

    # Setup the environments
    train_env, eval_env = make(config)

    # Create and initialse the learner
    print("Setting up the learner & warming the buffer...")
    key, subkey = jax.random.split(key)
    learn_fn, eval_network, learner_state = setup_learner(train_env, subkey, config)

    # Create and initialise the evaluator
    key, subkey = jax.random.split(key)
    evaluator, eval_keys = setup_evaluator(eval_env, eval_network.apply, subkey, config)

    # Create the logger
    logger = LogAggregator(project_name=config.logger.project)

    timestep = 0
    for _ in range(config.dynamic.eval_count):
        for _ in range(config.dynamic.rollouts_per_eval):
            print("Rollout started...")
            start_time = time.time()
            learner_state, episode_statistics, training_statistics = learn_fn(
                learner_state
            )
            jax.block_until_ready(learner_state)
            elapsed_time = time.time() - start_time

            steps_per_second = config.dynamic.steps_per_rollout / elapsed_time
            timestep += config.dynamic.steps_per_rollout

            terminal_mask = episode_statistics.pop("is_terminal_step")
            logger.log_pytree_mask(
                timestep, episode_statistics, terminal_mask, StatisticType.TRAIN
            )
            logger.log_pytree(timestep, training_statistics, StatisticType.OPT)
            logger.log_scalar(
                timestep, "steps_per_second", steps_per_second, StatisticType.TRAIN
            )

        print("Evaluation started...")
        start_time = time.time()
        eval_statistics = evaluator(learner_state.params.online, eval_keys)
        jax.block_until_ready(eval_statistics)
        elapsed_time = time.time() - start_time

        eval_steps = int(jnp.sum(eval_statistics["episode_length"]))
        steps_per_second = eval_steps / elapsed_time

        logger.log_pytree(timestep, eval_statistics, StatisticType.EVAL)
        logger.log_scalar(
            timestep, "steps_per_second", steps_per_second, StatisticType.EVAL
        )

        if checkpointer is not None:
            checkpointer.save(
                step=timestep,
                params=learner_state.params.online,
                eval_episode_return=jnp.mean(eval_statistics["episode_return"]),
            )
