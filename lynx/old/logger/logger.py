from enum import Enum

import chex
import jax
import jax.numpy as jnp
from dotenv import load_dotenv

import wandb

load_dotenv()


class StatisticType(Enum):
    TRAIN = "train"
    EVAL = "eval"
    OPT = "optim"


class LogAggregator:
    def __init__(self, project_name: str | None = None):
        self.log_backends = [WandbBackend(project_name=project_name)]
        # self.log_backends = [ConsoleBackend()]
        self.summary_statistics = {
            "mean": jnp.mean,
            "max": jnp.max,
            "min": jnp.min,
            "std": jnp.std,
        }

    def log_scalar(
        self,
        timestep: int,
        key: str,
        value: chex.Numeric,
        statistic_type: StatisticType,
    ):
        name = f"{statistic_type.value}/{key}"
        processed_statistics = {name: value}

        for backend in self.log_backends:
            backend.log(timestep, processed_statistics, statistic_type)

    def log_pytree(
        self, timestep: int, statistics: chex.ArrayTree, statistic_type: StatisticType
    ):
        processed_statistics = {}
        for key, value in statistics.items():
            for (
                summary_statistic_name,
                summary_statistic_fn,
            ) in self.summary_statistics.items():
                name = f"{statistic_type.value}/{key}/{summary_statistic_name}"
                summary_value = summary_statistic_fn(value)
                processed_statistics[name] = float(summary_value)

        for backend in self.log_backends:
            backend.log(timestep, processed_statistics, statistic_type)

    def log_pytree_mask(
        self,
        timestep: int,
        statistics: chex.ArrayTree,
        mask: chex.Array,
        statistic_type: StatisticType,
    ):
        if not jnp.any(mask):
            print("WARNING: No episodes completed...")
            return

        masked_statistics = jax.tree_util.tree_map(lambda x: x[mask], statistics)

        # Compile statistics
        processed_statistics = {}
        for key, value in masked_statistics.items():
            for (
                summary_statistic_name,
                summary_statistic_fn,
            ) in self.summary_statistics.items():
                name = f"{statistic_type.value}/{key}/{summary_statistic_name}"
                summary_value = summary_statistic_fn(value)
                processed_statistics[name] = float(summary_value)

        # Log statistics
        for backend in self.log_backends:
            backend.log(timestep, processed_statistics, statistic_type)

    def stop(self):
        pass


class ConsoleBackend:
    def log(
        self,
        timestep: int,
        statistics: dict[str, chex.Numeric],
        statistic_type: StatisticType,
    ):
        self._print_type_header(statistic_type)
        for key, value in statistics.items():
            print(f"{key}: {value}")
        print()

    def _print_type_header(self, statistic_type: StatisticType):
        print(f"{'-' * 100}")
        print(f"{statistic_type.value} statistics:")
        print(f"{'-' * 100}")


class WandbBackend:
    def __init__(self, project_name: str | None = None):
        self.run = wandb.init(
            project=project_name,
            tags=["dqn", "cpu"],
            group="dqn",
        )

    def log(
        self,
        timestep: int,
        statistics: dict[str, chex.Numeric],
        statistic_type: StatisticType,
    ):
        statistics["timestep"] = timestep
        wandb.log(statistics)

    def stop(self):
        self.run.finish()
