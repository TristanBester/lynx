import chex
import jax.numpy as jnp

from jumanji.env import Environment
from jumanji.environments.routing.snake.types import Observation as SnakeObservation
from jumanji.specs import Array, Spec


class SnakeAgentViewTransformer:
    def __call__(self, observation: SnakeObservation) -> chex.Array:
        grid = observation.grid.ravel()
        return grid.astype(jnp.float32)

    def get_observation_spec(self, env: Environment) -> Spec:
        grid_shape = jnp.asarray(env.observation_spec.grid.shape)
        agent_view_size = grid_shape.prod()
        return Array(shape=(agent_view_size,), dtype=jnp.float32)
