from typing import Any, Dict, NamedTuple, Optional

import chex
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep


class Observation(NamedTuple):
    """The observation that the agent sees.

    agent_view: the agent's view of the environment.
    action_mask: boolean array specifying which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_view: chex.Array  # (num_obs_features,)
    action_mask: chex.Array  # (num_actions,)
    step_count: Optional[chex.Array] = None  # (,)


class OnlineAndTarget(NamedTuple):
    online: FrozenDict
    target: FrozenDict


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    info: Dict


class OffPolicyLearnerState(NamedTuple):
    params: Any
    opt_state: Any
    buffer_state: Any
    key: chex.PRNGKey
    env_states: Any
    timesteps: TimeStep


class EvalState(NamedTuple):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: Any
    timestep: TimeStep
    step_count: chex.Array
    episode_return: chex.Array
