import os

import click
import yaml
from dotenv import load_dotenv
from fabric import Connection
from invoke import run
from jinja2 import Environment, FileSystemLoader

import wandb

load_dotenv()


@click.command()
@click.argument("agent")
@click.argument("operation")
@click.argument("env")
def cli(agent, operation, env):
    if operation != "sweep":
        click.echo("Operation not supported.")
        return

    sweep(agent, env)


def sweep(agent: str, env: str):
    _init(agent, env)
    _compile(agent, env)
    _sync()
    _deploy()


def _init(agent: str, env: str):
    wandb_sweep_config_path = os.path.join(
        os.getcwd(), f"lynx/configs/sweep/{agent}/{env}/sweep.yaml"
    )
    print(f"Loading sweep config from {wandb_sweep_config_path}")

    with open(wandb_sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config["project"])

    if not os.path.exists(".lynx/config"):
        os.makedirs(".lynx/config")

    with open(".lynx/config/sweep.yaml", "w") as f:
        yaml.dump({"sweep_id": sweep_id, "project": sweep_config["project"]}, f)


def _compile(agent: str, env: str):
    _compile_batch_script(agent, env)
    _compile_bash_script(agent, env)


def _compile_batch_script(agent: str, env: str):
    slurm_sweep_config_path = os.path.join(
        os.getcwd(), f"lynx/configs/sweep/{agent}/{env}/slurm.yaml"
    )
    template_dir = os.path.join(os.getcwd(), "cli/templates")
    sweep_config_path = os.path.join(os.getcwd(), ".lynx/config/sweep.yaml")

    template_env = Environment(loader=FileSystemLoader(template_dir))
    template = template_env.get_template("batch_script_template.sh.jinja")

    with open(sweep_config_path) as f:
        sweep_config = yaml.safe_load(f)
    with open(slurm_sweep_config_path) as f:
        slurm_config = yaml.safe_load(f)

    context = {
        "project": sweep_config["project"],
        "sweep_id": sweep_config["sweep_id"],
        "partition": slurm_config["partition"],
        "time_limit": slurm_config["time_limit"],
        "job_name": slurm_config["job_name"],
        "remote_project_dir": slurm_config["remote_project_dir"],
        "entity": slurm_config["entity"],
        "agents_per_node": slurm_config["agents_per_node"],
    }

    batchfile_content = template.render(context)

    with open(".lynx/batch/batch_script.sh", "w") as f:
        f.write(batchfile_content)


def _compile_bash_script(agent: str, env: str):
    slurm_sweep_config_path = os.path.join(
        os.getcwd(), f"lynx/configs/sweep/{agent}/{env}/slurm.yaml"
    )
    template_dir = os.path.join(os.getcwd(), "cli/templates")

    template_env = Environment(loader=FileSystemLoader(template_dir))
    template = template_env.get_template("bash_script_template.sh.jinja")

    with open(slurm_sweep_config_path) as f:
        slurm_config = yaml.safe_load(f)

    context = {
        "remote_project_dir": slurm_config["remote_project_dir"],
        "node_count": slurm_config["node_count"],
    }

    batchfile_content = template.render(context)

    with open(".lynx/bash/bash_script.sh", "w") as f:
        f.write(batchfile_content)


def _sync():
    project_path = "."
    project_name = "lynx"
    cluster_user = os.environ.get("CLUSTER_USER")
    cluster_host = os.environ.get("CLUSTER_HOST")

    rsync_command = (
        f"rsync -avz -e ssh "
        f"--exclude=*.pyc "
        f"--exclude=*.ipynb_checkpoints "
        f"--exclude=.env "
        f"--exclude=*.git "
        f"--exclude=*.tfevents* "
        f"--exclude=*.gitignore "
        f"--exclude=*.DS_Store "
        f"--exclude=.venv* "
        f"--exclude=wandb "
        f"--exclude=__pycache__ "
        f"--exclude=results "
        f"--exclude=*.zip "
        f"--exclude=.ruff_cache "
        f"--exclude=logs "
        f"--exclude=videos "
        f"--exclude=archive "
        f"--exclude=.tox "
        f"{project_path} "
        f"{cluster_user}@{cluster_host}:~/{project_name}"
    )

    # Run the rsync command on the remote cluster using Fabric
    result = run(rsync_command, hide=False)

    # if result.failed:
    #     logger.error("File sync failed.")
    # else:
    #     logger.success("File sync completed successfully.")


def _deploy():
    conn = Connection(
        os.environ.get("CLUSTER_HOST"), user=os.environ.get("CLUSTER_USER")
    )

    with conn.cd("~/lynx/"):
        result = conn.run("bash .lynx/bash/bash_script.sh", warn=True)
        print(result)

        # logger.trace(result.stdout)

        # if result.failed:
        #     logger.error("Job submission failed")
        # else:
        #     logger.success("Job submitted successfully")


if __name__ == "__main__":
    cli()
