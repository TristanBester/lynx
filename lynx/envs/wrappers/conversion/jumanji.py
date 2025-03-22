from functools import cached_property
from typing import Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.specs import Array, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from lynx.types.types import Observation


class JumanjiWrapper(Wrapper):
    def __init__(
        self,
        env: Environment,
        observation_attribute: str,
    ) -> None:
        self._env = env
        self._observation_attribute = observation_attribute
        self._num_actions = self.action_spec.num_values

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        if hasattr(timestep.observation, "action_mask"):
            legal_action_mask = timestep.observation.action_mask.astype(float)
        else:
            legal_action_mask = jnp.ones((self._num_actions,), dtype=float)
        if self._observation_attribute:
            agent_view = timestep.observation._asdict()[
                self._observation_attribute
            ].astype(float)
        else:
            agent_view = timestep.observation
        obs = Observation(agent_view, legal_action_mask, state.step_count)
        timestep_extras = timestep.extras
        if not timestep_extras:
            timestep_extras = {}
        timestep = timestep.replace(
            observation=obs,
            extras=timestep_extras,
        )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        if hasattr(timestep.observation, "action_mask"):
            legal_action_mask = timestep.observation.action_mask.astype(float)
        else:
            legal_action_mask = jnp.ones((self._num_actions,), dtype=float)
        if self._observation_attribute:
            agent_view = timestep.observation._asdict()[
                self._observation_attribute
            ].astype(jnp.float32)
        else:
            agent_view = timestep.observation
        obs = Observation(agent_view, legal_action_mask, state.step_count)
        timestep_extras = timestep.extras
        if not timestep_extras:
            timestep_extras = {}
        timestep = timestep.replace(
            observation=obs,
            extras=timestep_extras,
        )
        return state, timestep

    @cached_property
    def observation_spec(self) -> Spec:
        if self._observation_attribute:
            agent_view_spec = Array(
                shape=self._env.observation_spec.__dict__[
                    self._observation_attribute
                ].shape,
                dtype=float,
            )
        else:
            agent_view_spec = self._env.observation_spec

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=agent_view_spec,
            action_mask=Array(shape=(self._num_actions,), dtype=float),
            step_count=Array(shape=(), dtype=int),
        )
