import yaml
from fabric import task

import wandb


@task
def init(_):
    with open("config/sweep_config.yaml") as f:
        config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=config, project=config["project"])

    with open("config/sweep_id.yaml", "w") as f:
        yaml.dump({"sweep_id": sweep_id}, f)
