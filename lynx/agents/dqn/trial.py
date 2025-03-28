import json
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from lynx.agents.dqn.utils import compute_dynamic_statistics, run_experiment


def main():
    # Test if the script is being called correctly
    if len(sys.argv) != 3 or not os.path.exists(sys.argv[2]):
        raise ValueError(
            "No env or hyperparameter config file provided, or file does not exist."
        )

    # Load the agent config and wandb hyperparameter config
    wandb_hparam_config = _load_wandb_trial_hparams(sys.argv[2])

    # Load the agent config
    agent_config = _load_agent_config(sys.argv[1])

    # Merge the agent config with the wandb hyperparameter config
    config = _merge_config(agent_config, wandb_hparam_config)

    # Compute dynamic attributes
    config = compute_dynamic_statistics(config)  # type: ignore

    # Print the config and run the experiment
    pprint(OmegaConf.to_container(config, resolve=True))
    run_experiment(config)


def _load_agent_config(env_name):
    """Load the agent config from the config file."""
    with hydra.initialize(config_path="../../configs/base", version_base="1.2"):
        cfg = hydra.compose(
            config_name="dqn.yaml", overrides=[f"environment={env_name}"]
        )
    OmegaConf.set_struct(cfg, False)
    return cfg


def _load_wandb_trial_hparams(config_path: str) -> DictConfig:
    """Load the wandb hyperparameter config from the file."""
    print(f"Loading config from {config_path}")
    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return OmegaConf.create(config_dict)  # type: ignore
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}") from e


def _merge_config(agent_config, wandb_hparam_config):
    """Override base hyperparameters with wandb sweep hyperparameters."""
    for param, value in wandb_hparam_config.items():
        agent_config.train.hparams[param] = value
    return agent_config


if __name__ == "__main__":
    main()
