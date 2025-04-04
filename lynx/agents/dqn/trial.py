import json
import os
import sys
import time

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from lynx.agents.dqn.evaluator import setup_evaluator
from lynx.agents.dqn.learner.setup import setup_learner
from lynx.envs.factories.factory import make
from lynx.logger.logger import LogAggregator, StatisticType


def _load_agent_config():
    cfg = OmegaConf.load("lynx/configs/dqn.yaml")
    OmegaConf.set_struct(cfg, False)
    return cfg


def _load_wandb_trial_hparams(config_path: str) -> DictConfig:
    print(f"Loading config from {config_path}")
    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return OmegaConf.create(config_dict)  # type: ignore
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}") from e


def _merge_config(agent_config, wandb_hparam_config):
    for param, value in wandb_hparam_config.items():
        agent_config.train[param] = value
    return agent_config


def _compute_dynamic_statistics(cfg: DictConfig) -> DictConfig:
    dynamic = OmegaConf.create()

    dynamic.device_count = jax.device_count()
    dynamic.steps_per_rollout = (
        dynamic.device_count * cfg.train.envs_per_device * cfg.train.rollout_length
    )
    dynamic.rollouts_per_eval = (
        cfg.eval.desired_steps_per_eval // dynamic.steps_per_rollout
    )

    if dynamic.rollouts_per_eval == 0:
        # TODO: Handle elegantly
        # FIXME: Handle elegantly
        dynamic.rollouts_per_eval = 1

    dynamic.steps_per_eval = dynamic.steps_per_rollout * dynamic.rollouts_per_eval
    dynamic.updates_per_eval = cfg.train.updates_per_epoch * dynamic.rollouts_per_eval

    dynamic.eval_count = cfg.experiment.total_steps // dynamic.steps_per_eval

    cfg.dynamic = dynamic
    return cfg


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

    # Create the logger
    logger = LogAggregator()

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

        print("Evaluation started...")
        start_time = time.time()
        eval_statistics = evaluator(learner_state.params.online, eval_keys)
        jax.block_until_ready(eval_statistics)
        elapsed_time = time.time() - start_time

        eval_steps = int(jnp.sum(eval_statistics["episode_length"]))
        steps_per_second = eval_steps / elapsed_time

        logger.log_pytree(timestep, eval_statistics, StatisticType.EVAL)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.exists(sys.argv[1]):
        raise ValueError(
            "No hyperparameter config file provided, or file does not exist."
        )

    wandb_hparam_config = _load_wandb_trial_hparams(sys.argv[1])
    agent_config = _load_agent_config()
    config = _merge_config(agent_config, wandb_hparam_config)
    config = _compute_dynamic_statistics(config)  # type: ignore

    pprint(config)

    run_experiment(config)
