from flax import linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp
from jax import lax, random


class FiLM(nn.Module):

    @nn.compact
    def __call__(self, x, condition):
        film_params = nn.Dense(2*x.shape[-1], name="film_generator")(condition)
        film_params = film_params.reshape((film_params.shape[0], 2, -1))
        gamma, beta = film_params[:, 0, :], film_params[:, 1, :]
        residual = gamma * x + beta
        return residual + x

if __name__ == '__main__':
    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (4, 5))
    acts = random.uniform(key1, (4, 8))
    model = FiLM()
    params = model.init(key2, x, acts)
    out = model.apply(params, x, acts)
    print(out)


