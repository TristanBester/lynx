import chex
import jax.numpy as jnp

from jumanji.env import Environment
from jumanji.environments.logic.sliding_tile_puzzle.types import (
    Observation as SlidingTilePuzzleObservation,
)
from jumanji.specs import Array, Spec


class PuzzleAgentViewTransformer:
    def __call__(self, observation: SlidingTilePuzzleObservation) -> chex.Array:
        puzzle = observation.puzzle.ravel()
        empty = observation.empty_tile_position.ravel()
        return jnp.concatenate([puzzle, empty])

    def get_observation_spec(self, env: Environment) -> Spec:
        puzzle_shape = jnp.asarray(env.observation_spec.puzzle.shape)
        empty_tile_shape = jnp.asarray(env.observation_spec.empty_tile_position.shape)
        agent_view_size = puzzle_shape.prod() + empty_tile_shape.prod()
        return Array(shape=(agent_view_size,), dtype=float)
