import copy
from typing import Tuple

import hydra
import jumanji
from jumanji.env import Environment
from jumanji.registration import _REGISTRY as JUMANJI_REGISTRY
from jumanji.wrappers import AutoResetWrapper
from omegaconf import DictConfig

from lynx.envs.wrappers.conversion.jumanji import JumanjiWrapper
from lynx.envs.wrappers.logging.episode import RecordEpisodeMetrics


def make_jumanji_env(
    env_name: str,
    config: DictConfig,
) -> Tuple[Environment, Environment]:
    """Create a Jumanji environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    # Config generator and select the wrapper.

    # Create envs.
    env_kwargs = dict(copy.deepcopy(config.environment.kwargs))
    if "generator" in env_kwargs:
        generator = env_kwargs.pop("generator")
        generator = hydra.utils.instantiate(generator)
        env_kwargs["generator"] = generator
    env = jumanji.make(env_name, **env_kwargs)
    eval_env = jumanji.make(env_name, **env_kwargs)
    env, eval_env = (
        JumanjiWrapper(env, config.environment.observation_attribute),
        JumanjiWrapper(eval_env, config.environment.observation_attribute),
    )

    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def apply_optional_wrappers(
    envs: Tuple[Environment, Environment], config: DictConfig
) -> Tuple[Environment, Environment]:
    """Apply optional wrappers to the environments.

    Args:
        envs (Tuple[Environment, Environment]): The training and evaluation environments to wrap.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    envs = list(envs)
    if "wrappers" in config.environment and config.environment.wrappers is not None:
        for i in range(len(envs)):
            envs[i] = hydra.utils.instantiate(config.environment.wrappers, env=envs[i])

    return tuple(envs)  # type: ignore


def make(config: DictConfig) -> Tuple[Environment, Environment]:
    """Create environments for training and evaluation..

    Args:
        config (Dict): The configuration of the environment.

    Returns:
        training and evaluation environments.
    """
    env_name = config.environment.name
    if env_name in JUMANJI_REGISTRY:
        envs = make_jumanji_env(env_name, config)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
    envs = apply_optional_wrappers(envs, config)
    return envs
