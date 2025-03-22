import chex
from flax import linen as nn

from lynx.types.types import Observation


class AgentViewEncoder(nn.Module):
    """Only Observation Input."""

    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        agent_view = observation.agent_view
        return agent_view
