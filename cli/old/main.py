
import click
from dotenv import load_dotenv

load_dotenv()


@click.command()
@click.argument("agent")
@click.argument("operation")
@click.argument("env")
def cli(agent, operation, env):
    if operation == "sweep":
        sweep(agent, env)
    elif operation == "train":
        train(agent)
    else:
        click.echo("Operation not supported.")


if __name__ == "__main__":
    cli()
