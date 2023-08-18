import flax
import flax.linen as nn
import jax.numpy as jnp
from jax.random import PRNGKey

class MLP(nn.Module):                    # create a Flax Module dataclass
  out_dims: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(128)(x)                 # create inline Flax Module submodules
    x = nn.relu(x)
    x = nn.Dense(self.out_dims)(x)       # shape inference
    return x

model = MLP(out_dims=10)                 # instantiate the MLP model

x = jnp.empty((4, 28, 28, 1))            # generate random data
for i in range(1000):
    variables = model.init(PRNGKey(42), x)   # initialize the weights
    y = model.apply(variables, x)            # make forward pass
