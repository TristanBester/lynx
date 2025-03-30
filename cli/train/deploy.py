import os

from dotenv import load_dotenv
from fabric import Connection

load_dotenv()


def deploy():
    conn = Connection(
        os.environ.get("CLUSTER_HOST"), user=os.environ.get("CLUSTER_USER")
    )

    with conn.cd("~/lynx/"):
        result = conn.run("bash .lynx/train/bash_script.sh", warn=True)
        print(result)

        # logger.trace(result.stdout)

        # if result.failed:
        #     logger.error("Job submission failed")
        # else:
        #     logger.success("Job submitted successfully")
