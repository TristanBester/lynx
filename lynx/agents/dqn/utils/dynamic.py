import jax
from omegaconf import DictConfig, OmegaConf


def compute_dynamic_statistics(cfg: DictConfig) -> DictConfig:
    dynamic = OmegaConf.create()

    dynamic.device_count = jax.device_count()
    dynamic.steps_per_rollout = (
        dynamic.device_count
        * cfg.train.hparams.envs_per_device
        * cfg.train.hparams.rollout_length
    )
    dynamic.rollouts_per_eval = (
        cfg.train.eval.desired_steps_per_eval // dynamic.steps_per_rollout
    )

    if dynamic.rollouts_per_eval == 0:
        # TODO: Handle elegantly
        dynamic.rollouts_per_eval = 1

    dynamic.steps_per_eval = dynamic.steps_per_rollout * dynamic.rollouts_per_eval
    dynamic.updates_per_eval = (
        cfg.train.hparams.updates_per_epoch * dynamic.rollouts_per_eval
    )

    dynamic.eval_count = cfg.train.config.total_steps // dynamic.steps_per_eval

    cfg.dynamic = dynamic
    return cfg
