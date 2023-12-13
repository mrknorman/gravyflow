from dataclasses import dataclass
from typing import Union, List, Dict, Optional
from copy import deepcopy
from pathlib import Path
import os
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras.layers import Lambda
from keras import backend as K
from keras.callbacks import Callback
from tensorflow.keras import losses
from tensorflow.keras.layers import Layer
import json

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
    value: Union[int, float, str] = None
    
    def __post_init__(self):
        
        if isinstance(self.distribution, HyperParameter):
            self.distribution = self.distribution.distribution
            self.value = self.distribution.value
        elif not isinstance(self.distribution, gf.Distribution):
            self.distribution = gf.Distribution(value=self.distribution, type_=gf.DistributionType.CONSTANT)
        
        self.randomize()

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
        if np.random.random() < mutation_rate:
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
                    self.value = self.distribution.sample()
                case gf.DistributionType.LOG:
                    log_value = mutate_value(
                        np.log(self.value), self.distribution.min_, self.distribution.max_, mutation_strength
                    )
                    self.value = 10 ** log_value
                case gf.DistributionType.POW_TWO:
                    power_low, power_high = map(int, np.log2((self.distribution.min_, self.distribution.max_)))
                    log_value = mutate_value(
                        np.ln(self.value), power_low, power_high, mutation_strength
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

class ModelGenome:

    def __init__(
        self,
        optimizer : HyperParameter,
        batch_size : HyperParameter,
        learning_rate: HyperParameter,
        num_layers : HyperParameter,
        layer_genomes : List
    ):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.layer_genomes = layer_genomes

        self.genes = [
            self.optimizer,
            self.batch_size,
            self.learning_rate,
            self.num_layers,
         ] + self.layer_genomes

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
        
    def crossover(self, genome):
        new_genes = []
        for old_gene, new_gene in zip(self.genes, genome.genes):
            new_genes.append(np.choice(old_gene, new_gene))
        
        self.genes = new_genes

    # Using pickle here for now:
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
