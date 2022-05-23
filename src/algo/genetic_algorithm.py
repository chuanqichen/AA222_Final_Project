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
- Take top K chromies to produce next generation of N individuals. (K < N)
- repeat N-1 times:
    - sample a parent from top K, mutate it
- For last individual: 
    - select the best parent from top k (elitism)
    - to ensure we get true elite, evaluate the K chromies additional x times


*Tournament
*Each parent is the fittest among K randomly chosen chromosomes

*Roulette Wheel
*Each parent is chosen with a prob proportional to its performance relative to the population
Softmax Function is a common approach. 
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
            perturbations = jax.random.normal(subkey, shape = (self.pop_size-1, self.param_size))
            perturbations = jnp.vstack((perturbations, jnp.zeros(1, self.param_size)))
            next_generation = params + perturbations * self.sigma
            return key, next_generation

        #* tell() choices: truncation, tournament, roulette
        def truncation(fitness: ndarray, params: ndarray) -> ndarray:
            params = params[fitness.argsort(axis=0)] # params sorted from least to most fit
            best_param = params[-1]
            params = params[-self.elite_size:] # params truncated to most fit K individuals
            # sample pop_size individuals from elite population
            params = jax.random.choice(self.rng_key, params, shape=[self.pop_size-1])
            params = jnp.vstack((params, best_param))
            return params

        self.ask_fn = jax.jit(gaussian_mutation)
        self.tell_fn = jax.jit(truncation)

    def ask(self) -> jnp.ndarray:
        self.rng_key, self.params = self.ask_fn(self.rng_key, self.params)
        return self.params

    def tell(self, fitness: ndarray) -> None:
        self.params = self.tell_fn(fitness, self.params)
        self._best_params = self.params[-1]

    @property
    def best_params(self) -> jnp.ndarray:
        return self.jnp_array(self._best_params)
    
    @best_params.setter
    def best_params(self, params: ndarray) -> None:
        self.params = jnp.repeat(params[None, :], self.pop_size, axis=0)
        self._best_params = jnp.array(params, copy=True)

