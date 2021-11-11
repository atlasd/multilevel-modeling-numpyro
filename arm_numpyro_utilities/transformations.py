from jax import numpy as jnp

def logistic(arr):
    return 1 / (1 + jnp.exp(-arr))
