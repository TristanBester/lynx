import yaml
from fabric import task
from jinja2 import Environment, FileSystemLoader

from lynx.exp.logger import get_logger

logger = get_logger()


@task
def compile(_):
    logger.info("Compiling the batch files...")
    _compile_batch_script()
    logger.info("Compiling bash script...")
    _compile_bash_script()
    logger.success("Complilation complete.")


def _compile_batch_script():
    template_dir = "./templates"
    env = Environment(loader=FileSystemLoader(template_dir))

    template = env.get_template("batch_script_template.sh.jinja")

    with open("config/job_config.yaml") as f:
        job_config = yaml.safe_load(f)

    context = {
        "partition": job_config["partition"],
        "time_limit": job_config["time_limit"],
        "job_name": job_config["job_name"],
        "experiment_script": job_config["experiment_script"],
        "remote_project_dir": job_config["remote_project_dir"],
        "seed_count": job_config["seed_count"],
        "concurrency_limit": job_config["concurrency_limit"],
        "agents_per_node": job_config["agents_per_node"],
    }

    batchfile_content = template.render(context)

    with open("batch_script.sh", "w") as f:
        f.write(batchfile_content)


def _compile_bash_script():
    template_dir = "./templates"
    env = Environment(loader=FileSystemLoader(template_dir))

    template = env.get_template("bash_script_template.sh.jinja")

    with open("config/job_config.yaml") as f:
        job_config = yaml.safe_load(f)

    context = {
        "remote_project_dir": job_config["remote_project_dir"],
        "node_count": job_config["node_count"],
    }

    batchfile_content = template.render(context)

    with open("bash_script.sh", "w") as f:
        f.write(batchfile_content)
