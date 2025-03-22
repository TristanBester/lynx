import time

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from lynx.agents.dqn.learner.setup import setup_learner
from lynx.agents.dqn.evaluator import setup_evaluator
from lynx.envs.factories.factory import make


def run_experiment(config):
    key = jax.random.PRNGKey(config.experiment.seed)

    # Setup the environments
    train_env, eval_env = make(config)

    # Create and initialse the learner
    key, subkey = jax.random.split(key)
    learn_fn, eval_network, learner_state = setup_learner(train_env, subkey, config)

    # Create and initialise the evaluator
    key, subkey = jax.random.split(key)
    evaluator, eval_keys = setup_evaluator(eval_env, eval_network.apply, subkey, config)

    ##### DEBUG CODE AFTER THIS POINT #####
    steps_per_eval = (
        config.train.rollout_length
        * config.train.batch_size
        * config.train.epochs_per_eval
    )

    for i in range(10):
        start_time = time.time()
        learner_state, episode_statistics, training_statistics = learn_fn(learner_state)
        jax.block_until_ready(learner_state)
        elapsed_time = time.time() - start_time
        steps_per_second = steps_per_eval / elapsed_time

        print("-" * 100)
        print(
            f"Step: {(i+1) * steps_per_eval}, Steps per second: {steps_per_second:2f}"
        )

        is_terminal_step = episode_statistics.pop("is_terminal_step")

        if not jnp.any(is_terminal_step):
            return None

        episode_statistics = jax.tree_util.tree_map(
            lambda x: x[is_terminal_step], episode_statistics
        )

        if episode_statistics is None:
            print("No episode statistics to log")
            return

        print("Episode statistics:")
        for key, value in episode_statistics.items():
            print(f"{key}: {jnp.mean(value):2f}")

        start_time = time.time()
        eval_statistics = evaluator(learner_state.params.online, eval_keys)
        jax.block_until_ready(eval_statistics)
        elapsed_time = time.time() - start_time
        steps_per_eval = int(jnp.sum(eval_statistics["episode_length"]))
        steps_per_second = steps_per_eval / elapsed_time
        print()
        print("Eval statistics:")
        print(f"Steps per second: {steps_per_second:2f}")
        for key, value in eval_statistics.items():
            print(f"{key}: {jnp.mean(value):2f}")
        print("-" * 100)


@hydra.main(
    config_path="/Users/tristan/Projects/lynx/lynx/configs",
    config_name="dqn.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig):
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    pprint(OmegaConf.to_container(cfg, resolve=True))

    run_experiment(cfg)


if __name__ == "__main__":
    hydra_entry_point()
