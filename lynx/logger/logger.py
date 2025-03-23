import os
from enum import Enum

import chex
import jax
import jax.numpy as jnp
import neptune
import wandb
from dotenv import load_dotenv

load_dotenv()


class StatisticType(Enum):
    TRAIN = "train"
    EVAL = "eval"
    OPT = "optim"


class LogAggregator:
    def __init__(self):
        self.log_backends = [ConsoleBackend()]
        self.summary_statistics = {
            "mean": jnp.mean,
            "max": jnp.max,
            "min": jnp.min,
            "std": jnp.std,
        }

    def log_scalar(self, timestep: int, key: str, value: chex.Numeric):
        pass

    def log_pytree(self, timestep: int, statistics: chex.ArrayTree):
        pass

    def log_pytree_mask(
        self,
        timestep: chex.Numeric,
        statistics: chex.ArrayTree,
        mask: chex.Array,
        statistic_type: StatisticType,
    ):
        masked_statistics = jax.tree_util.tree_map(lambda x: x[mask], statistics)

        # Compile statistics
        processed_statistics = {"timestep": timestep}
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
            backend.log(processed_statistics, statistic_type)

    def stop(self):
        pass


class ConsoleBackend:
    def log(
        self,
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


class NeptuneBackend:
    def __init__(self):
        self.run = neptune.init_run(
            project="tristanbester/SnakeAI",
            api_token=os.getenv("NEPTUNE_API_KEY"),
        )

    def log(self, statistics: dict[str, chex.Numeric], _: StatisticType):
        for key, value in statistics.items():
            self.run[key].append(value)

    def stop(self):
        self.run.stop()


class WandbBackend:
    def __init__(self):
        self.run = wandb.init(
            project="lynx",
        )

    def log(
        self,
        statistics: dict[str, chex.Numeric],
        _: StatisticType,
    ):
        wandb.log(statistics)

    def stop(self):
        self.run.finish()
