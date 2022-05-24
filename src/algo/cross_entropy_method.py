from functools import partial
import logging
import numpy as np
from typing import Union, Tuple

import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger

jnp_array = jax.jit(jnp.array)
jnp_cov = jax.jit(partial(jnp.cov,rowvar=False))

ndarray = Union[np.ndarray, jnp.ndarray]

"""
CEM with multivariate normal proposal distribution, P
P is represented by mu and sigma, rather than an object like distrax or tfp
"""
class CrossEntropyMethod(NEAlgorithm):
    
    def __init__(
        self,
        param_size: int,
        pop_size: int = 64,
        elite_size: int = 10,
        seed: int = 0,
        logger: logging.Logger = None
    ):
        self.logger = create_logger(name='CrossEntropyMethod') if logger is None else logger
        self.param_size = param_size
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.rng_key = jax.random.PRNGKey(seed = seed)
        self.mu = jnp.zeros(param_size)
        self.sigma = jnp.eye(param_size)
    
        def ask_fn(key: jnp.ndarray) -> Tuple[jnp.ndarray, ndarray]:
            key, subkey = jax.random.split(key)
            params = jax.random.multivariate_normal(subkey, self.mu, self.sigma, shape=[self.pop_size-1])
            params = jnp.vstack((params, self.mu))
            return key, params

        def tell_fn(fitness: ndarray, params: ndarray) -> ndarray:
            params = params[fitness.argsort(axis=0)] 
            params = params[-self.elite_size:]
            best_param = params[-1]
            mu = params.mean(axis=0)
            sigma = jnp_cov(params)
            return mu, sigma, best_param

        self.ask_fn = jax.jit(ask_fn)
        self.tell_fn = jax.jit(tell_fn)

    def ask(self) -> jnp.ndarray:
        self.rng_key, self.params = self.ask_fn(self.rng_key)
        return self.params

    def tell(self, fitness: ndarray) -> None:
        self.mu, self.sigma, self._best_params = self.tell_fn(fitness, self.params)

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp_array(self._best_params)
    
    @best_params.setter
    def best_params(self, params: ndarray) -> None:
        self.params = jnp.repeat(params[None, :], self.pop_size, axis=0)
        self._best_params = jnp.array(params, copy=True)

