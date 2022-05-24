import logging
import numpy as np
from typing import Union, Tuple

import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger

"""
*Truncation
*Each parent is randomly chosen from the top K chromosomes

*Tournament
*Each parent is the fittest among K randomly chosen chromosomes

*Roulette Wheel
*Each parent is chosen with a prob proportional to its performance relative to the population
"""
jnp_array = jax.jit(jnp.array)
ndarray = Union[np.ndarray, jnp.ndarray]

"""
Simple genetic algorithm for neuroevolution with gaussian mutation and no crossover
selection method can be chosen from [truncation, tournament, roulette]
eliteism is applied in the mutation stage
"""

class GeneticAlgorithm(NEAlgorithm):

    def __init__(self,
                 param_size: int,
                 selection: str = "truncation",
                 elitesm: bool = True, # not toggleable yet, defaults to yes
                 pop_size: int = 64,
                 elite_size: int = None,
                 sigma: float = 0.01,
                 seed: int = 0,
                 logger: logging.Logger = None
                 ):
    
        self.logger = create_logger(name='GeneticAlgortihm') if logger is None else logger

        self.param_size = param_size
        self.pop_size = pop_size
        self.sigma = sigma
        self.params = jnp.zeros((pop_size, param_size))
        self._best_params = None        
        self.rng_key = jax.random.PRNGKey(seed = seed)

        #controls top K in truncation, random K  in tournament. not used for roulette
        self.elite_size = elite_size 

        #* ask() + elitism
        def gaussian_mutation(key: jnp.ndarray, params: ndarray) -> Tuple[jnp.ndarray, ndarray]:
            key, subkey = jax.random.split(key)
            gaussian_noise = jax.random.normal(subkey, shape = (self.pop_size-1, self.param_size))
            null_noise = jnp.zeros((1, self.param_size)) # placeholder for elitism
            perturbations = jnp.vstack((gaussian_noise, null_noise))
            next_generation = params + perturbations * self.sigma
            return key, next_generation

        #* tell() choices: truncation, tournament, roulette
        def truncation(fitness: ndarray, params: ndarray) -> ndarray:
            params = params[fitness.argsort(axis=0)] # params sorted from least to most fit
            best_param = params[-1]

            params = params[-self.elite_size:]
            params = jax.random.choice(self.rng_key, params, shape=[self.pop_size-1])
            params = jnp.vstack((params, best_param))
            return params
        
        def tournament(fitness: ndarray, params: ndarray) -> ndarray:
            params = params[fitness.argsort(axis=0)] # params sorted from least to most fit
            best_param = params[-1]

            winner_idxes = jax.random.choice(
                self.rng_key,
                self.pop_size,
                shape=[self.pop_size-1, self.elite_size]
                ).max(axis=1)
            params = params[winner_idxes]
            params = jnp.vstack((params, best_param))
            return params
        
        def roulette(fitness: ndarray, params: ndarray) -> ndarray:
            inds = fitness.argsort(axis=0)
            params = params[inds] # params sorted from least to most fit
            best_param = params[-1]

            p = jax.nn.softmax(fitness[inds])
            params = jax.random.choice(self.rng_key, params, shape=[self.pop_size-1], p = p)
            params = jnp.vstack((params, best_param))
            return params


        self.ask_fn = jax.jit(gaussian_mutation)
        if selection == "truncation":
            self.tell_fn = jax.jit(truncation)
        elif selection == "tournament":
            self.tell_fn = jax.jit(tournament)
        elif selection == "roulette":
            self.tell_fn = jax.jit(roulette)

    def ask(self) -> jnp.ndarray:
        self.rng_key, self.params = self.ask_fn(self.rng_key, self.params)
        return self.params

    def tell(self, fitness: ndarray) -> None:
        self.params = self.tell_fn(fitness, self.params)
        self._best_params = self.params[-1]

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp_array(self._best_params)
    
    @best_params.setter
    def best_params(self, params: ndarray) -> None:
        self.params = jnp.repeat(params[None, :], self.pop_size, axis=0)
        self._best_params = jnp.array(params, copy=True)

