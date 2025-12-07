"""
DEPRECATED: This module is deprecated and not actively maintained.
It is excluded from test coverage requirements.
"""
from dataclasses import dataclass
from typing import Union, List, Dict, Optional
from copy import deepcopy
from pathlib import Path
import os
import logging

import numpy as np
from numpy.random import default_rng  
import keras
from keras.layers import Lambda
from keras import backend as K
from keras.callbacks import Callback
from keras import losses
from keras.layers import Layer
import json
import pickle


import gravyflow as gf

def mutate_value(value, A, B, std_dev_fraction):
    # Calculate the standard deviation as a fraction of the range
    std_dev = std_dev_fraction * (B - A)

    # Mutate the value
    mutated_value = value + np.random.normal(0, std_dev)

    # Clamp the result to be within [A, B]
    mutated_value = max(min(mutated_value, B), A)

    return mutated_value
    

@dataclass
class HyperParameter:
    distribution: gf.Distribution
    seed : int = None
    value: Union[int, float, str] = None
    
    def __post_init__(self):

        if self.seed is None:
            self.seed = gf.Defaults.seed

        self.rng = default_rng(self.seed)
        
        if isinstance(self.distribution, HyperParameter):
            self.distribution = self.distribution.distribution
            self.value = self.distribution.value
        elif not isinstance(self.distribution, gf.Distribution):
            self.distribution = gf.Distribution(value=self.distribution, type_=gf.DistributionType.CONSTANT)

        self.distribution.reseed(
            self.rng.integers(1E10)
        )
        
        self.randomize()

    def reseed(self, seed):
        self.seed = seed
        self.rng = default_rng(self.seed)

        self.distribution.reseed(
            self.rng.integers(1E10)
        )

    def randomize(self):
        """
        Randomizes this hyperparameter based on its possible_values.
        """
        self.value = self.distribution.sample()[0]

    def mutate(self, mutation_rate: float, mutation_strength : float = 0.1):
        """
        Returns a new HyperParameter with a mutated value, based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_param: New HyperParameter instance with potentially mutated value.
        """
        if self.rng.random() < mutation_rate:
            match self.distribution.type_:
                case gf.DistributionType.CONSTANT:      
                    pass     
                case gf.DistributionType.UNIFORM:
                    self.value = mutate_value(
                        self.value, self.distribution.min_, self.distribution.max_, mutation_strength
                    )
                case gf.DistributionType.NORMAL:
                    self.value = mutate_value(
                        self.value, self.distribution.min_, self.distribution.max_, mutation_strength
                    )
                case gf.DistributionType.CHOICE:
                    self.value = self.distribution.sample()[0]
                case gf.DistributionType.LOG:
                    log_value = mutate_value(
                        np.log(self.value), self.distribution.min_, self.distribution.max_, mutation_strength
                    )
                    self.value = 10 ** log_value
                case gf.DistributionType.POW_TWO:
                    power_low, power_high = map(int, np.log2((self.distribution.min_, self.distribution.max_)))
                    log_value = mutate_value(
                        np.log2(self.value), power_low, power_high, mutation_strength
                    )
                    self.value = 2**log_value
            
            if self.distribution.dtype == int:

                if self.distribution.type_ == gf.DistributionType.LOG:
                    raise ValueError(
                        "Cannot convert log values to ints."
                    )
                elif self.distribution.type_ == gf.DistributionType.CHOICE:
                    raise ValueError(
                        "Cannot convert choice values to ints."
                    )

                self.value = int(self.value)

    def crossover(self, other, crossover_rate = 0.5):
        if (self.rng.random() < crossover_rate):
            self.distribution = other.distribution
            self.value = other.value

class HyperInjectionGenerator: 

    def __init__(
            self,
            min_ : HyperParameter,
            max_ : HyperParameter,
            mean : HyperParameter,
            std : HyperParameter,
            distribution : HyperParameter,
            chance : HyperParameter,
            generator : HyperParameter
        ):

        self.min_ = min_
        self.max_ = max_
        self.mean = mean
        self.std = std
        self.distribution = distribution
        self.chance = chance
        self.generator = deepcopy(generator)

        self.genes = [
            self.min_,
            self.max_,
            self.mean,
            self.std,
            self.distribution,
            self.chance,
            self.generator
        ]

    def reseed(self, seed):
        self.seed = seed
        self.rng = default_rng(seed)
        
        for gene in self.genes:
            gene.reseed(self.rng.integers(1E10))

    def randomize(self):
        """
        Randomizes this hyperparameter based on its possible_values.
        """
        for gene in self.genes:
            gene.randomize()

    def mutate(self, mutation_rate):
        for gene in self.genes:
            gene.mutate(mutation_rate)

    def crossover(self, other, crossover_rate = 0.5):
        for old, new in zip(self.genes, other.genes):
            old.crossover(new, crossover_rate)

    def return_generator(self):
        generator = self.generator.value
        
        generator.scaling_method.value.min_ = self.min_.value
        generator.scaling_method.value.max_ = self.max_.value
        generator.scaling_method.value.mean = self.mean.value
        generator.scaling_method.value.std = self.std.value
        generator.scaling_method.value.distribution = self.distribution.value
        generator.injection_chance = self.chance.value

        return generator
        
class ModelGenome:

    def __init__(
        self,
        optimizer : HyperParameter,
        batch_size : HyperParameter,
        learning_rate: HyperParameter,
        injection_generators : Dict[str, HyperInjectionGenerator],
        noise_type : HyperParameter,
        exclude_glitches : HyperParameter,
        onsource_duration_seconds : HyperParameter, 
        offsource_duration_seconds : HyperParameter,
        sample_rate_hertz : HyperParameter,
        num_layers : HyperParameter,
        layer_genomes : List,
        seed : int = None
    ):
        
        # Training genes:
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.injection_generators = injection_generators

        #Noise Genes:
        self.exclude_glitches = exclude_glitches
        self.noise_type = noise_type

        # Temporal properties:
        self.onsource_duration_seconds = onsource_duration_seconds
        self.offsource_duration_seconds = offsource_duration_seconds
        self.sample_rate_hertz = sample_rate_hertz

        # Architecture genome:
        self.num_layers = num_layers
        self.layer_genomes = layer_genomes

        if seed is None:
            seed = gf.Defaults.seed

        # Gene list:
        self.genes = [
            self.optimizer,
            self.batch_size,
            self.learning_rate,
            self.noise_type,
            self.exclude_glitches,
            self.onsource_duration_seconds,
            self.offsource_duration_seconds,
            self.sample_rate_hertz,
            self.num_layers
        ] + self.layer_genomes + [gen["hp"] for gen in self.injection_generators.values()]

        self.reseed(seed)

    def reseed(self, seed): 

        # Randomisation:
        self.seed = seed
        self.rng = default_rng(self.seed)

        for gene in self.genes:
            gene.reseed(self.rng.integers(1E10))

        for layer_genome in self.layer_genomes:
            for possibility in layer_genome.distribution.possible_values:
                possibility.reseed(self.rng.integers(1E10))
    
    def randomize(self):
        for gene in self.genes:
            gene.randomize()

        for layer_genome in self.layer_genomes:
            for possibility in layer_genome.distribution.possible_values:
                possibility.randomize()

    def mutate(self, mutation_rate):
        for gene in self.genes:
            gene.mutate(mutation_rate)    

        for layer_genome in self.layer_genomes:
            for possibility in layer_genome.distribution.possible_values:
                possibility.mutate(mutation_rate)
    
    def crossover(self, genome, crossover_rate = 0.5):
        for old, new in zip(self.genes, genome.genes):
            old.crossover(new, crossover_rate)
    
    # Using pickle here for now:
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)