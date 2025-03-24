import time

import hydra
import jax
import jax.numpy as jnp
from carbs import (
    CARBS,
    CARBSParams,
    LinearSpace,
    LogitSpace,
    LogSpace,
    Param,
    ObservationInParam,
)
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from lynx.agents.dqn.evaluator import setup_evaluator
from lynx.agents.dqn.learner.setup import setup_learner
from lynx.envs.factories.factory import make
from lynx.logger.logger import LogAggregator, StatisticType
import copy


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

    return jnp.mean(eval_statistics["episode_return"]).item()


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

    for trial in range(10):
        print(f"TRIAL: {trial}")
        original_suggestion = carbs.suggest().suggestion
        suggestion = copy.deepcopy(original_suggestion)

        del suggestion["suggestion_uuid"]

        suggestion = OmegaConf.create(suggestion)

        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        del cfg["sweep"]  # type: ignore

        cfg.experiment.total_steps = suggestion["total_steps"]
        cfg.train = OmegaConf.merge(cfg.train, suggestion)
        cfg = _compute_dynamic_statistics(cfg)  # type: ignore
        pprint(OmegaConf.to_container(cfg, resolve=True))

        result = run_experiment(cfg)
        carbs.observe(
            ObservationInParam(
                input=original_suggestion, output=result, cost=suggestion["total_steps"]
            )
        )


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
        dynamic.rollouts_per_eval = 1

    dynamic.steps_per_eval = dynamic.steps_per_rollout * dynamic.rollouts_per_eval
    dynamic.updates_per_eval = cfg.train.updates_per_epoch * dynamic.rollouts_per_eval

    dynamic.eval_count = cfg.experiment.total_steps // dynamic.steps_per_eval

    cfg.dynamic = dynamic
    return cfg


if __name__ == "__main__":
    main()
