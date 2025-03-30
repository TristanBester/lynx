import os

import yaml
from jinja2 import Environment, FileSystemLoader


def compile(agent: str):
    _compile_batch_script(agent)
    _compile_bash_script()


def _compile_batch_script(agent):
    slurm_config_path = os.path.join(os.getcwd(), "lynx/configs/slurm/train.yaml")
    template_dir = os.path.join(os.getcwd(), "cli/train/templates")
    agent_path = f"lynx/agents/{agent}/train.py"

    template_env = Environment(loader=FileSystemLoader(template_dir))
    template = template_env.get_template("batch_script_template.sh.jinja")

    with open(slurm_config_path) as f:
        slurm_config = yaml.safe_load(f)

    context = {
        "partition": slurm_config["partition"],
        "time_limit": slurm_config["time_limit"],
        "job_name": slurm_config["job_name"],
        "remote_project_dir": slurm_config["remote_project_dir"],
        "entity": slurm_config["entity"],
        "agents_per_node": slurm_config["agents_per_node"],
        "agent_path": agent_path,
    }

    batchfile_content = template.render(context)

    if not os.path.exists(".lynx/train"):
        os.makedirs(".lynx/train")

    with open(".lynx/train/batch_script.sh", "w") as f:
        f.write(batchfile_content)


def _compile_bash_script() -> None:
    slurm_config_path = os.path.join(os.getcwd(), "lynx/configs/slurm/train.yaml")
    template_dir = os.path.join(os.getcwd(), "cli/train/templates")

    template_env = Environment(loader=FileSystemLoader(template_dir))
    template = template_env.get_template("bash_script_template.sh.jinja")

    with open(slurm_config_path) as f:
        slurm_config = yaml.safe_load(f)

    context = {
        "remote_project_dir": slurm_config["remote_project_dir"],
        "node_count": slurm_config["node_count"],
    }

    batchfile_content = template.render(context)

    if not os.path.exists(".lynx/train"):
        os.makedirs(".lynx/train")

    with open(".lynx/train/bash_script.sh", "w") as f:
        f.write(batchfile_content)
