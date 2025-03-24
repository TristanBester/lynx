import os

import wandb


def run_trial(config):
    return config.x**2 + config.y**2


def main():
    print("RUNNING")
    print(os.environ.get("WANDB_PROJECT"))
    wandb.init(project=os.environ.get("WANDB_PROJECT"))
    loss = run_trial(wandb.config)
    wandb.log({"loss": loss})


if __name__ == "__main__":
    main()
