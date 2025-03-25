import absl.logging as absl_logging
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint

absl_logging.set_verbosity(absl_logging.ERROR)


class Checkpointer:
    def __init__(
        self, model_name: str, checkpoint_dir: str, max_to_keep: int, keep_period: int
    ):
        """Constructor.

        Args:
            model_name (str): Name of the model.
            checkpoint_dir (str): Directory to save checkpoints.
            max_to_keep (int): Number of checkpoints to keep.
            keep_period (Optional[int], optional):
                If set, will not delete any checkpoint where
                checkpoint_step % keep_period == 0. Defaults to None.
        """
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep  # Number of checkpoints to keep
        self.keep_period = keep_period

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(
            create=True,
            best_fn=lambda x: x["eval_episode_return"],
            best_mode="max",
            max_to_keep=max_to_keep,
            keep_period=keep_period,
        )
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            directory=self.checkpoint_dir,
            checkpointers=checkpointer,
            options=options,
        )

    def save(self, step: int, params, eval_episode_return: float | jnp.ndarray) -> None:
        """Save the checkpoint."""
        # Unreplicate the state before saving
        params = jax.tree_util.tree_map(lambda x: x[(0,) * 2], params)

        # Convert JAX arrays to NumPy arrays
        # FIXME: orbax shouldn't require casting to numpy to work
        params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
        eval_episode_return = float(eval_episode_return)  # Convert to Python float

        self.checkpoint_manager.save(
            step=step,
            items={"params": params},
            metrics={"eval_episode_return": eval_episode_return},
        )

    def restore(self):
        checkpoint = self.checkpoint_manager.restore(
            self.checkpoint_manager.latest_step()
        )
        return checkpoint["params"]
