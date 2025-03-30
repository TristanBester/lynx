import os

from dotenv import load_dotenv
from invoke import run

load_dotenv()


def download():
    project_path = "."
    project_name = "lynx"
    cluster_user = os.environ.get("CLUSTER_USER")
    cluster_host = os.environ.get("CLUSTER_HOST")

    rsync_command = (
        f"rsync -avz -e ssh "
        f"{cluster_user}@{cluster_host}:~/{project_name}/checkpoints "
        f"{project_path}"
    )

    # Run the rsync command on the remote cluster using Fabric
    result = run(rsync_command, hide=False)
    print(result)
