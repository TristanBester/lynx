import distrax
from flax import linen as nn

from lynx.types.types import Observation


class Actor(nn.Module):
    """Simple Feedforward Actor Network."""

    encoder: nn.Module
    backbone: nn.Module
    head: nn.Module

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.DistributionLike:
        embedding_observation = self.encoder(observation)
        backbone_output = self.backbone(embedding_observation)
        return self.head(backbone_output)
