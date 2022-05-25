from functools import partial
import logging
import numpy as np
from typing import Union, Tuple

import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger

jnp_array = jax.jit(jnp.array)
jnp_cov = jax.jit(partial(jnp.cov,rowvar=False, bias=True))

ndarray = Union[np.ndarray, jnp.ndarray]

"""
CEM with multivariate normal proposal distribution, P, using diagonal covariance matrix cuz i didn't know how to handle NaN full covariance matrix
P is represented by mu and sigma, rather than an object like distrax or tfp
"""
class CrossEntropyMethod(NEAlgorithm):
    def __init__(
        self,
        param_size: int,
        pop_size: int = 64,
        elite_size: int = 10,
        stdev_init: float = 1.0,
        noisy: bool = True, # adds noise to the covariance matrix: https://ieeexplore.ieee.org/document/6796865
        seed: int = 0,
        logger: logging.Logger = None
    ):
        self.logger = create_logger(name='CrossEntropyMethod') if logger is None else logger
        self.param_size = param_size
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.rng_key = jax.random.PRNGKey(seed = seed)

        self.mu = jnp.zeros(param_size)
        self.sigma = jnp.eye(param_size) * stdev_init
        self.params = jnp.zeros((pop_size, param_size))
    
        def ask_fn(key: jnp.ndarray, mu, sigma) -> Tuple[jnp.ndarray, ndarray]:
            key, subkey = jax.random.split(key)
            params = jax.random.multivariate_normal(subkey, mu, sigma, shape=[self.pop_size])
            return key, params

        def tell_fn(fitness: ndarray, params: ndarray, rng_key) -> ndarray:
            params = params[fitness.argsort(axis=0)] 
            params = params[-self.elite_size:]
            mu = params.mean(axis=0)
            # sigma = jnp_cov(params) # full covariance
            sigma = params.var(axis=0)# diagonal covariance as a vector
            if noisy: 
                sigma += abs(jax.random.normal(rng_key,sigma.shape)) # Noisy CEM
            return mu, jnp.diagflat(sigma)

        self.ask_fn = jax.jit(ask_fn)
        self.tell_fn = jax.jit(tell_fn)

    def ask(self) -> jnp.ndarray:
        self.rng_key, self.params = self.ask_fn(self.rng_key, self.mu, self.sigma)
        return self.params

    def tell(self, fitness: ndarray) -> None:
        self.mu, self.sigma = self.tell_fn(fitness, self.params, self.rng_key)
        self._best_params = self.mu

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp_array(self._best_params)
    
    @best_params.setter
    def best_params(self, params: ndarray) -> None:
        self.params = jnp.repeat(params[None, :], self.pop_size, axis=0)
        self._best_params = jnp.array(params, copy=True)

