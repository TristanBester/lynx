from cli.train.compile import compile
from cli.train.deploy import deploy
from cli.train.sync import sync


def train(agent: str) -> None:
    compile(agent=agent)
    sync()
    deploy()
