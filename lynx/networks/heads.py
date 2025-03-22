import chex
import distrax
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal


class QNetworkHead(nn.Module):
    action_dim: int
    epsilon: float = 0.1
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.EpsilonGreedy:
        q_values = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)
