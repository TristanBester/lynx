import click

from cli.download import download
from cli.train import train


@click.command()
@click.argument("operation")
@click.argument("agent", required=False)
def cli(operation: str, agent: str | None = None) -> None:
    if operation == "train" and agent is not None:
        train(agent)
    elif operation == "download":
        download()
    else:
        raise NotImplementedError(f"Operation {operation} not implemented")
