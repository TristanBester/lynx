def train(agent):
    _compile(agent)
    # _sync()
    # _deploy()


def _compile(agent):
    _compile_batch_script(agent)
    _compile_bash_script(agent)


def _compile_batch_script(agent):
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
