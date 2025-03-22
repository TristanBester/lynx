from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper


@chex.dataclass
class RecordEpisodeMetricsState:
    """State of the `LogWrapper`."""

    env_state: Any
    key: chex.PRNGKey
    # Temporary variables to keep track of the episode return and length.
    running_count_episode_return: chex.Numeric
    running_count_episode_length: chex.Numeric
    # Final episode return and length.
    episode_return: chex.Numeric
    episode_length: chex.Numeric


class RecordEpisodeMetrics(Wrapper):
    """Record the episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """Reset the environment."""
        key, reset_key = jax.random.split(key)
        state, timestep = self._env.reset(reset_key)
        state = RecordEpisodeMetricsState(
            env_state=state,
            key=key,
            running_count_episode_return=jnp.array(0.0, dtype=float),
            running_count_episode_length=jnp.array(0, dtype=int),
            episode_return=jnp.array(0.0, dtype=float),
            episode_length=jnp.array(0, dtype=int),
        )
        timestep.extras["episode_metrics"] = {
            "episode_return": jnp.array(0.0, dtype=float),
            "episode_length": jnp.array(0, dtype=int),
            "is_terminal_step": jnp.array(False, dtype=bool),
        }
        return state, timestep

    def step(
        self,
        state: RecordEpisodeMetricsState,
        action: chex.Array,
    ) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """Step the environment."""
        env_state, timestep = self._env.step(state.env_state, action)

        done = timestep.last()
        not_done = 1 - done

        # Counting episode return and length.
        new_episode_return = state.running_count_episode_return + timestep.reward
        new_episode_length = state.running_count_episode_length + 1

        # Previous episode return/length until done and then the next episode return.
        episode_return_info = (
            state.episode_return * not_done + new_episode_return * done
        )
        episode_length_info = (
            state.episode_length * not_done + new_episode_length * done
        )

        timestep.extras["episode_metrics"] = {
            "episode_return": episode_return_info,
            "episode_length": episode_length_info,
            "is_terminal_step": done,
        }

        state = RecordEpisodeMetricsState(
            env_state=env_state,
            key=state.key,
            running_count_episode_return=new_episode_return * not_done,
            running_count_episode_length=new_episode_length * not_done,
            episode_return=episode_return_info,
            episode_length=episode_length_info,
        )
        return state, timestep
