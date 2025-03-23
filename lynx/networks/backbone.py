from typing import Sequence

import chex
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal


class MLPBackbone(nn.Module):
    """MLP Backbone."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(
                layer_size,
                kernel_init=self.kernel_init,
            )(x)
            x = nn.silu(x)
        return x
