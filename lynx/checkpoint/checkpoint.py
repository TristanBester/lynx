import os

import absl.logging as absl_logging
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from huggingface_hub import HfApi, snapshot_download

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

    def upload_to_hf(self, repo_name: str, folder_path: str):
        """Upload the checkpoint to Hugging Face."""
        api = HfApi()
        api.upload_folder(folder_path=folder_path, repo_id=repo_name)

    def download_from_hf(self, repo_name: str, path_in_repo: str):
        print(f"Downloading checkpoint from {repo_name} at {path_in_repo}")
        snapshot_download(
            repo_id=repo_name,
            allow_patterns=path_in_repo + "/*",
            local_dir=".",
        )
        print(f"Downloaded checkpoint from {repo_name} at {path_in_repo}")

    def upload_best_to_hf(self, repo_name: str, path_in_repo: str):
        print(f"Uploading best checkpoint to {repo_name} at {path_in_repo}")
        checkpoint_path = os.path.join(
            str(self.checkpoint_manager.directory),
            str(self.checkpoint_manager.best_step()),
        )
        api = HfApi()
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_name,
            path_in_repo=path_in_repo,
        )
        print(f"Uploaded best checkpoint to {repo_name} at {path_in_repo}")
