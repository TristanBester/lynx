import os

from dotenv import load_dotenv
from fabric import task
from invoke import run

from lynx.test.logger import get_logger

load_dotenv()
logger = get_logger()


@task
def sync(c):
    logger.info("Starting file sync with remote host...")

    project_path = "../../"
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

    if result.failed:
        logger.error("File sync failed.")
    else:
        logger.success("File sync completed successfully.")
