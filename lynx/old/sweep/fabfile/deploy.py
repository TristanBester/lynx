import os

from dotenv import load_dotenv
from fabric import Connection, task

from lynx.sweep.logger import get_logger

load_dotenv()
logger = get_logger()


@task
def deploy(_):
    logger.info("Submitting slurm jobs to cluster...")

    conn = Connection(
        os.environ.get("CLUSTER_HOST"), user=os.environ.get("CLUSTER_USER")
    )

    with conn.cd("~/lynx/lynx/sweep"):
        logger.info("Submitting job via SLURM")
        result = conn.run("bash bash_script.sh", warn=True)

        logger.trace(result.stdout)

        if result.failed:
            logger.error("Job submission failed")
        else:
            logger.success("Job submitted successfully")
