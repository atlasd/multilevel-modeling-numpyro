import numpyro
from jax import random

def get_nuts_mcmc(
    model,
    num_samples,
    num_warmup,
    model_kwargs,
    num_chains=1,
    rng_key_int=0
):
    nuts = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(

        nuts,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup
    )
    mcmc.run(**model_kwargs, rng_key=random.PRNGKey(rng_key_int))
    return mcmc
