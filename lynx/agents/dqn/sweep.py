import time

import hydra
import jax
from carbs import (
    CARBS,
    CARBSParams,
    LinearSpace,
    LogitSpace,
    LogSpace,
    Param,
)
from omegaconf import DictConfig, OmegaConf

from lynx.agents.dqn.evaluator import setup_evaluator
from lynx.agents.dqn.learner.setup import setup_learner
from lynx.envs.factories.factory import make
from lynx.logger.logger import LogAggregator, StatisticType


def setup_sweep_params(config: DictConfig):
    cost_param = _instantiate_param(config.sweep.cost)

    hparams = []
    for param_config in config.sweep.hparams:
        hparams.append(_instantiate_param(param_config))

    return cost_param, hparams


def _instantiate_param(config: DictConfig):
    scale = 4 if config.int else 0.3
    if config.type == "log_space":
        space = LogSpace(
            min=config.min, max=config.max, is_integer=config.int, scale=scale
        )
    elif config.type == "linear":
        space = LinearSpace(
            min=config.min, max=config.max, is_integer=config.int, scale=scale
        )
    elif config.type == "logit":
        space = LogitSpace(
            min=config.min, max=config.max, is_integer=config.int, scale=scale
        )
    else:
        raise ValueError(f"Unknown space type {config.type}")
    return Param(name=config.name, space=space, search_center=config.center)


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
        config.train.rollout_length * config.train.envs_per_device * jax.device_count()
    )
    logger = LogAggregator()

    for i in range(10):
        start_time = time.time()
        learner_state, episode_statistics, training_statistics = learn_fn(learner_state)
        jax.block_until_ready(learner_state)
        elapsed_time = time.time() - start_time
        steps_per_second = steps_per_eval / elapsed_time
        timestep = (i + 1) * steps_per_eval

        terminal_mask = episode_statistics.pop("is_terminal_step")
        logger.log_pytree_mask(
            timestep, episode_statistics, terminal_mask, StatisticType.TRAIN
        )


@hydra.main(
    config_path="/Users/tristan/Projects/lynx/lynx/configs",
    config_name="sweep.yaml",
    version_base="1.2",
)
def main(config: DictConfig):
    cost_param, hyparams = setup_sweep_params(config)
    params = [cost_param] + hyparams

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
    )
    carbs = CARBS(carbs_params, params)

    suggestion = carbs.suggest().suggestion
    del suggestion["suggestion_uuid"]

    suggestion = OmegaConf.create(suggestion)

    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    del cfg["sweep"]

    cfg.train = OmegaConf.merge(cfg.train, suggestion)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
