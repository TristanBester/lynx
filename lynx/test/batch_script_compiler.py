import yaml
from jinja2 import Environment, FileSystemLoader


def compile_batch_script():
    template_dir = "./templates"
    env = Environment(loader=FileSystemLoader(template_dir))

    template = env.get_template("batch_script_template.sh.jinja")

    with open("config/sweep_id.yaml") as f:
        sweep_id = yaml.safe_load(f)
    with open("config/sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)
    with open("config/job_config.yaml") as f:
        job_config = yaml.safe_load(f)

    context = {
        "project": sweep_config["project"],
        "sweep_id": sweep_id["sweep_id"],
        "partition": job_config["partition"],
        "time_limit": job_config["time_limit"],
        "job_name": job_config["job_name"],
        "remote_project_dir": job_config["remote_project_dir"],
        "entity": job_config["entity"],
        "agents_per_node": job_config["agents_per_node"],
    }

    batchfile_content = template.render(context)

    with open("batch_script.sh", "w") as f:
        f.write(batchfile_content)


def compile_bash_script():
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


if __name__ == "__main__":
    compile_batch_script()
    compile_bash_script()
