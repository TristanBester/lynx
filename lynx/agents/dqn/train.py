import os

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from lynx.agents.dqn.utils import compute_dynamic_statistics, run_experiment
from lynx.checkpoint import Checkpointer


@hydra.main(
    config_path="../../configs/base/",
    config_name="dqn.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig):
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Compute dynamic statistics.
    config = compute_dynamic_statistics(cfg)

    # Create the checkpointer
    checkpointer = Checkpointer(
        model_name=f"dqn-snake-{config.train.experiment.seed}",
        checkpoint_dir=os.path.join(
            os.getcwd(), f"checkpoints/seed-{config.train.experiment.seed}"
        ),
        max_to_keep=1,
        # keep_period=config.dynamic.steps_per_eval,
        keep_period=100_000,  # FIXME: Random number so mod fails and not all stored
    )

    # Print the config and run the experiment.
    pprint(OmegaConf.to_container(config, resolve=True))
    run_experiment(cfg, checkpointer=checkpointer)


if __name__ == "__main__":
    hydra_entry_point()
