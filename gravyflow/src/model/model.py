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
import json
import time
import pickle
import datetime

logger = logging.getLogger(__name__)

import numpy as np
from tqdm import tqdm
import keras
from keras import ops
from keras import layers
from keras import losses
from keras.layers import Layer
import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax import lax

import gravyflow as gf

def gamma_log_prob(x, concentration, rate):
    return jstats.gamma.logpdf(x, a=concentration, scale=1/rate)

def folded_normal_log_prob(x, loc, scale):
    # Folded normal distribution: |Y| where Y ~ N(loc, scale)
    # PDF(x) = (1/sigma * sqrt(2pi)) * (exp(-(x-mu)^2/2sigma^2) + exp(-(x+mu)^2/2sigma^2)) for x >= 0
    # We use logsumexp for numerical stability
    
    # Constants
    log_sqrt_2pi = jnp.log(jnp.sqrt(2 * jnp.pi))
    log_sigma = jnp.log(scale)
    
    term1 = -0.5 * ((x - loc) / scale) ** 2
    term2 = -0.5 * ((x + loc) / scale) ** 2
    
    # log(exp(a) + exp(b)) = a + log(1 + exp(b-a))
    log_sum = ops.logsumexp(ops.stack([term1, term2], axis=-1), axis=-1)
    
    return log_sum - log_sigma - log_sqrt_2pi

def trunc_normal_log_prob(x, loc, scale, low, high):
    return jstats.truncnorm.logpdf(x, a=(low - loc) / scale, b=(high - loc) / scale, loc=loc, scale=scale)

def beta_prime_log_prob(x, alpha, beta):
    # Beta prime: f(x) = x^(alpha-1) * (1+x)^(-alpha-beta) / B(alpha, beta)
    # log f(x) = (alpha-1)log(x) - (alpha+beta)log(1+x) - log B(alpha, beta)
    # log B(alpha, beta) = lgamma(alpha) + lgamma(beta) - lgamma(alpha+beta)
    
    log_prob = (alpha - 1) * jnp.log(x) - (alpha + beta) * jnp.log(1 + x)
    log_beta_func = lax.lgamma(alpha) + lax.lgamma(beta) - lax.lgamma(alpha + beta)
    return log_prob - log_beta_func

def gamma_nll(y_true, y_pred):
    alpha, beta = ops.split(y_pred, 2, axis=-1)
    return -gamma_log_prob(y_true, alpha, beta)

def folded_normal_nll(y_true, y_pred):
    loc, scale = ops.split(y_pred, 2, axis=-1)
    return -folded_normal_log_prob(y_true, loc, scale)

def trunc_normal_nll(y_true, y_pred, low=0.0, high=100.0):
    loc, scale = ops.split(y_pred, 2, axis=-1)
    return -trunc_normal_log_prob(y_true, loc, scale, low, high)

def beta_prime_nll(y_true, y_pred):
    alpha, beta = ops.split(y_pred, 2, axis=-1)
    return -beta_prime_log_prob(y_true, alpha, beta)


from numpy.random import default_rng  

import gravyflow as gf


class IndependentGamma(layers.Layer):
    """An independent Gamma Keras layer that outputs parameters."""

    def __init__(self, event_shape=(), **kwargs):
        super().__init__(**kwargs)
        self.event_shape = event_shape

    def call(self, inputs):
        alpha, beta = ops.split(inputs, 2, axis=-1)
        alpha = ops.softplus(alpha) + 1.0e-5
        beta = ops.softplus(beta) + 1.0e-5
        return ops.concatenate([alpha, beta], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"event_shape": self.event_shape})
        return config

class IndependentFoldedNormal(layers.Layer):
    """An independent folded normal Keras layer that outputs parameters."""

    def __init__(self, event_shape=(), **kwargs):
        super().__init__(**kwargs)
        self.event_shape = event_shape

    def call(self, inputs):
        loc, scale = ops.split(inputs, 2, axis=-1)
        loc = loc + 1.0e-6
        scale = scale + 1.0e-6
        
        # In original code:
        # loc = tf.math.softplus(loc)
        # scale = tf.math.softplus(scale)
        # But wait, original code:
        # loc_params = tf.cast(loc_params, dtype = tf.float32)  + 1.0E-6
        # scale_params = tf.cast(scale_params, dtype = tf.float32) + 1.0E-6
        # return tfd.Normal(loc=tf.math.softplus(loc), scale=tf.math.softplus(scale))
        # So yes, softplus is applied.
        
        loc = ops.softplus(loc)
        scale = ops.softplus(scale)
        return ops.concatenate([loc, scale], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"event_shape": self.event_shape})
        return config

class IndependentTruncNormal(layers.Layer):
    """An independent truncated normal Keras layer that outputs parameters."""

    def __init__(self, event_shape=(), low=0.0, high=100.0, **kwargs):
        super().__init__(**kwargs)
        self.event_shape = event_shape
        self.low = low
        self.high = high

    def call(self, inputs):
        loc, scale = ops.split(inputs, 2, axis=-1)
        loc = loc + 1.0e-5
        # Fixing the bug from original code where scale used loc_params
        scale = scale + 1.0e-5 
        
        # Original: scale=tf.math.softplus(scale)
        scale = ops.softplus(scale)
        
        return ops.concatenate([loc, scale], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "event_shape": self.event_shape,
            "low": self.low,
            "high": self.high
        })
        return config

class IndependentBetaPrime(layers.Layer):
    """An independent Beta prime Keras layer that outputs parameters."""

    def __init__(self, event_shape=(), **kwargs):
        super().__init__(**kwargs)
        self.event_shape = event_shape

    def call(self, inputs):
        alpha, beta = ops.split(inputs, 2, axis=-1)
        alpha = ops.softplus(alpha)
        beta = ops.softplus(beta)
        return ops.concatenate([alpha, beta], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"event_shape": self.event_shape})
        return config


def adjust_features(features, labels):
    labels['INJECTION_MASKS'] = labels['INJECTION_MASKS'][0]
    return features, labels

@dataclass
class BaseLayer:
    layer_type: str = "Base"
    seed : int = None
    activation: Union[gf.HyperParameter, str] = None
    mutable_attributes: List = None

    def reseed(self, seed):
        
        rng = default_rng(seed)
        for attribute in self.mutable_attributes:
            attribute.reseed(rng.integers(1E10))
    
    def randomize(self):
        """
        Randomizes all mutable attributes of this layer.
        """
        for attribute in self.mutable_attributes:
            attribute.randomize()
            
    def mutate(self, mutation_rate: float):
        """
        Returns a new layer with mutated hyperparameters based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_layer: New BaseLayer instance with potentially mutated hyperparameters.
        """
        for attribute in self.mutable_attributes:
            attribute.mutate(mutation_rate)

    def crossover(self, other, crossover_rate: float = 0.5):
        """
        Returns a new layer with mutated hyperparameters based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_layer: New BaseLayer instance with potentially mutated hyperparameters.
        """
        for old, new in (self.mutable_attributes, other.mutable_attributes):
            old.crossover(new, crossover_rate)

class Reshape(Layer):
    def __init__(self, reshaping_mode = "depthwise", **kwargs):
        super(Reshape, self).__init__(**kwargs)
        
        self.reshaping_mode = reshaping_mode

    def call(self, inputs):
        # Reshape the input tensor based on the specified mode
        if self.reshaping_mode == 'lengthwise':
            # (num_batches, num_features * num_steps, 1)
            return ops.reshape(inputs, [ops.shape(inputs)[0], -1, 1])
        elif self.reshaping_mode == 'depthwise':
            # (num_batches, num_steps, num_features)
            return ops.transpose(inputs, axes=[0, 2, 1])
        elif self.reshaping_mode == 'heightwise':
            # (num_batches, num_features, num_steps, 1)
            return ops.expand_dims(inputs, axis=-1)
        else:
            raise ValueError("Invalid reshaping mode")

    def get_config(self):
        config = super(Reshape, self).get_config()
        config.update({"reshaping_mode": self.reshaping_mode})
        return config

@dataclass
class DenseLayer(BaseLayer):
    units: gf.HyperParameter = 64
    activation: gf.HyperParameter = "relu"
    dropout_present : Union[gf.HyperParameter, bool] = None 
    dropout_value : Union[gf.HyperParameter, str] = None
    batch_normalisation_present : Union[gf.HyperParameter, bool] = None

    def __init__(
            self, 
            units: Union[gf.HyperParameter, int] = 64, 
            dropout_present : Union[gf.HyperParameter, bool] = False,
            dropout_value : Union[gf.HyperParameter, str] = 0.0,
            batch_normalisation_present : Union[gf.HyperParameter, bool] = False,
            activation: Union[gf.HyperParameter, str] = "relu"
        ):
        """
        Initializes a DenseLayer instance.
        
        Args:
        ---
        units : Union[gf.HyperParameter, int]
            gf.HyperParameter specifying the number of units in this layer.
        activation : Union[gf.HyperParameter, int]
            gf.HyperParameter specifying the activation function for this layer.
        """

        self.rng = default_rng(self.seed)

        self.layer_type = "Dense"
        self.activation = gf.HyperParameter(
            activation, seed=self.rng.integers(1E10)
        )
        self.units = gf.HyperParameter(
            units, seed=self.rng.integers(1E10)
        )
        self.dropout_present = gf.HyperParameter(
            dropout_present, seed=self.rng.integers(1E10)
        )
        self.dropout_value = gf.HyperParameter(
            dropout_value, seed=self.rng.integers(1E10)
        )
        self.batch_normalisation_present = gf.HyperParameter(
            batch_normalisation_present, seed=self.rng.integers(1E10)
        )
        self.mutable_attributes = [
            self.activation, 
            self.units, 
            self.dropout_present, 
            self.dropout_value, 
            self.batch_normalisation_present
        ]

@dataclass
class FlattenLayer(BaseLayer):

    def __init__(
            self
        ):
        """
        Initializes a DenseLayer instance.
        
        Args:
        ---
        units : Union[gf.HyperParameter, int]
            gf.HyperParameter specifying the number of units in this layer.
        activation : Union[gf.HyperParameter, int]
            gf.HyperParameter specifying the activation function for this layer.
        """
        self.rng = default_rng(self.seed)
        self.layer_type = "Flatten"
        self.mutable_attributes = []

@dataclass
class ConvLayer(BaseLayer):
    filters: gf.HyperParameter = 16
    kernel_size: gf.HyperParameter = 16
    strides: gf.HyperParameter = 1
    activation: gf.HyperParameter = "relu"
    dropout_present : Union[gf.HyperParameter, bool] = None 
    dropout_value : Union[gf.HyperParameter, str] = None
    batch_normalisation_present : Union[gf.HyperParameter, bool] = None
    pooling_present : Union[gf.HyperParameter, bool] = None
    pooling_size : Union[gf.HyperParameter, int] = None 
    pooling_stride : Union[gf.HyperParameter, int] = None
    
    def __init__(self, 
            filters: gf.HyperParameter = 16, 
            kernel_size: gf.HyperParameter = 16, 
            activation: gf.HyperParameter = "relu", 
            strides: gf.HyperParameter = 1,
            dilation: gf.HyperParameter = 0,
            dropout_present : Union[gf.HyperParameter, bool] = False,
            dropout_value : Union[gf.HyperParameter, str] = 0.0,
            batch_normalisation_present : Union[gf.HyperParameter, bool] = False,
            pooling_present : Union[gf.HyperParameter, bool] = False,
            pooling_size : Union[gf.HyperParameter, int] = 4,
            pooling_stride : Union[gf.HyperParameter, int] = 1
        ):
        """
        Initializes a ConvLayer instance.
        
        Args:
        filters: gf.HyperParameter specifying the number of filters in this layer.
        kernel_size: gf.HyperParameter specifying the kernel size in this layer.
        activation: gf.HyperParameter specifying the activation function for this layer.
        strides: gf.HyperParameter specifying the stride length for this layer.
        """
        self.rng = default_rng(self.seed)
        self.layer_type = "Convolutional"
        self.activation = gf.HyperParameter(
            activation, seed=self.rng.integers(1E10)
        )
        self.filters = gf.HyperParameter(
            filters, seed=self.rng.integers(1E10)
        )
        self.kernel_size = gf.HyperParameter(
            kernel_size, seed=self.rng.integers(1E10)
        )
        self.strides = gf.HyperParameter(
            strides, seed=self.rng.integers(1E10)
        )
        self.dilation = gf.HyperParameter(
            dilation, seed=self.rng.integers(1E10)
        )
        self.dropout_present = gf.HyperParameter(
            dropout_present, seed=self.rng.integers(1E10)
        )
        self.dropout_value = gf.HyperParameter(
            dropout_value, seed=self.rng.integers(1E10)
        )
        self.batch_normalisation_present = gf.HyperParameter(
            batch_normalisation_present, seed=self.rng.integers(1E10)
        )
        self.pooling_present = gf.HyperParameter(
            pooling_present, seed=self.rng.integers(1E10)
        )
        self.pooling_size = gf.HyperParameter(
            pooling_size, seed=self.rng.integers(1E10)
        )
        self.pooling_stride = gf.HyperParameter(
            pooling_stride, seed=self.rng.integers(1E10)
        )

        self.padding = gf.HyperParameter("same", seed=self.rng.integers(1E10))
        
        self.mutable_attributes = [
            self.activation, 
            self.filters, 
            self.kernel_size, 
            self.strides, 
            self.dilation, 
            self.dropout_present, 
            self.dropout_value, 
            self.batch_normalisation_present,
            self.pooling_present,
            self.pooling_size,
            self.pooling_stride
        ]
        
@dataclass
class PoolLayer(BaseLayer):
    pool_size: gf.HyperParameter = 4
    strides: gf.HyperParameter = 4
    
    def __init__(self, 
        pool_size: gf.HyperParameter = 4, 
        strides: Optional[Union[gf.HyperParameter, int]] = None
        ):
        """
        Initializes a PoolingLayer instance.
        
        Args:
        pool_size: gf.HyperParameter specifying the size of the pooling window.
        strides: gf.HyperParameter specifying the stride length for moving the pooling window.
        """
        self.rng = default_rng(self.seed)
        self.layer_type = "Pooling"
        self.pool_size = gf.HyperParameter(pool_size, seed=self.rng.integers(1E10))
        
        if strides is None:
            self.strides = self.pool_size
        else:
            self.strides = gf.HyperParameter(strides, seed=self.rng.integers(1E10))
        
        self.padding = gf.HyperParameter("same", seed=self.rng.integers(1E10))
        self.mutable_attributes = [self.pool_size, self.strides]
        
class DropLayer(BaseLayer):
    rate: gf.HyperParameter = 0.5
    
    def __init__(self, rate: Union[gf.HyperParameter, float]):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: gf.HyperParameter specifying the dropout rate for this layer.
        """
        self.rng = default_rng(self.seed)
        self.layer_type = "Dropout"
        self.rate = gf.HyperParameter(rate, seed=self.rng.integers(1E10))
        self.mutable_attributes = [self.rate]

class BatchNormLayer(BaseLayer):
    
    def __init__(self):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: gf.HyperParameter specifying the dropout rate for this layer.
        """
        self.rng = default_rng(self.seed)
        self.layer_type = "BatchNorm"

class WhitenLayer(BaseLayer):
    
    def __init__(self):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: gf.HyperParameter specifying the whitening for this layer.
        """
        self.rng = default_rng(self.seed)
        self.layer_type = "Whiten"
        self.mutable_attributes = []

class WhitenPassLayer(BaseLayer):
    
    def __init__(self):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: gf.HyperParameter specifying the whitening for this layer.
        """
        self.rng = default_rng(self.seed)
        self.layer_type = "WhitenPass"
        self.mutable_attributes = []

def cap_value(x):
    return ops.clip(x, 1.0e-5, 1000)  # values will be constrained to [-1, 1]

from gravyflow.src.utils.numerics import ensure_even

class Model:
    def __init__(
        self, 
        name : str,
        layers: List[BaseLayer], 
        optimizer: str, 
        loss: str, 
        input_configs : Union[List[Dict], Dict],
        output_config : dict,
        training_config : dict = None,
        dataset_args : dict = None,
        batch_size: int = None,
        model_path : Path = None,
        metrics : list = [],
        genome : gf.ModelGenome = None,
        seed : int = None
    ):
        """
        Initializes a Model instance.
        
        Args:
        layers: List of BaseLayer instances making up the model.
        optimizer: Optimizer to use when training the model.
        loss: Loss function to use when training the model.
        batch_size: Batch size to use when training the model.
        """

        self.training_dataset = None

        if dataset_args is not None:
            self.training_dataset = gf.Dataset(
                **deepcopy(dataset_args)
            ).map(adjust_features)

        if seed is None:
            seed = gf.Defaults.seed

        self.name = name
        self.path = model_path
        self.genome = genome
        self.rng = default_rng(seed)

        if batch_size is None:
            batch_size = gf.Defaults.num_examples_per_batch

        if optimizer is None:
            optimizer = "adam"

        if loss is None:
            if output_config["type"] == "normal":
                loss = folded_normal_nll
            else:
                loss = losses.BinaryCrossentropy()
        
        self.layers = layers
        self.batch_size = gf.HyperParameter(
            batch_size, 
            seed=self.rng.integers(1E10)
        )
        self.optimizer = gf.HyperParameter(
            optimizer, 
            seed=self.rng.integers(1E10)
        )
        self.loss = gf.HyperParameter(
            loss, 
            seed=self.rng.integers(1E10)
        )
        self.training_config = training_config
        self.loaded = False
        
        self.fitness = []
        self.metrics = []

        self.build(
            input_configs,
            output_config,
            model_path,
            metrics
        )

    @classmethod
    def from_genome(
        cls,
        genome : gf.ModelGenome,
        name : str,
        input_configs : dict, 
        output_config : dict,
        training_config : dict,
        dataset_args : dict,
        model_path : Path,
        metrics : List,
        seed : int = None
    ):
        if seed is None:
            seed = gf.Defaults.seed

        layers = []            
        for i in range(genome.num_layers.value):
            layers.append(
                deepcopy(genome.layer_genomes[i].value)
            )
        
        training_config["model_path"] = model_path
        training_config["learning_rate"] = genome.learning_rate.value
        training_config["batch_size"] = genome.batch_size.value
        training_config["onsource_duration_seconds"] = genome.onsource_duration_seconds.value
        training_config["offsource_duration_seconds"] = genome.offsource_duration_seconds.value
        training_config["sample_rate_hertz"] = genome.sample_rate_hertz.value

        dataset_args["num_examples_per_batch"] = genome.batch_size.value
        dataset_args["onsource_duration_seconds"] = genome.onsource_duration_seconds.value
        dataset_args["offsource_duration_seconds"] = genome.offsource_duration_seconds.value
        dataset_args["sample_rate_hertz"] = genome.sample_rate_hertz.value

        # Use WindowSpec for consistent sample count calculation
        window_spec = gf.WindowSpec.from_params(
            sample_rate_hertz=genome.sample_rate_hertz.value,
            onsource_duration_seconds=genome.onsource_duration_seconds.value,
            offsource_duration_seconds=genome.offsource_duration_seconds.value
        )
        num_onsource_samples = window_spec.num_onsource_samples
        num_offsource_samples = window_spec.num_offsource_samples

        if genome.layer_genomes[0].value.layer_type == "WhitenPass":
            dataset_args["input_variables"] = [
                gf.ReturnVariables.ONSOURCE
            ]
            input_configs = [
                {
                    "name" : gf.ReturnVariables.ONSOURCE.name,
                    "shape" : (1, num_onsource_samples,)
                }
            ]

        else:
            input_configs = [
                {
                    "name" : gf.ReturnVariables.ONSOURCE.name,
                    "shape" : (1, num_onsource_samples,)
                },
                {
                    "name" : gf.ReturnVariables.OFFSOURCE.name,
                    "shape" : (1, num_offsource_samples,)
                }
            ]

        gf.Defaults.set(
            num_examples_per_batch=dataset_args["num_examples_per_batch"],
            sample_rate_hertz=dataset_args["sample_rate_hertz"],
            onsource_duration_seconds=dataset_args["onsource_duration_seconds"],
            offsource_duration_seconds=dataset_args["offsource_duration_seconds"]
        )

        output_config = {
            "name" : gf.ReturnVariables.INJECTION_MASKS.name,
            "type" : "binary"
        }

        dataset_args = deepcopy(dataset_args)

        # This is currently quite specific:
        dataset_args["waveform_generators"] = {
            name : { 
                "generator" : generator["hp"].return_generator(),
                "excluded" : generator["excluded"],
                "exclusive" : generator["exclusive"]
            } for name, generator in genome.injection_generators.items()
        }
        dataset_args["noise_obtainer"].noise_type = genome.noise_type.value
        
        # Log training_config
        for key, value in training_config.items():
            logger.info(f"Training config: {key} = {value}")

        # Log dataset_args
        for key, value in dataset_args.items():
            logger.info(f"Dataset args: {key} = {value}")

        # Create an instance of DenseModel with num_neurons list
        model = cls(
            name=name,
            layers=layers, 
            optimizer=genome.optimizer.value, 
            loss=gf.HyperParameter(losses.BinaryCrossentropy(), seed=100), 
            input_configs=input_configs, 
            output_config=output_config,
            training_config=training_config,
            dataset_args=dataset_args,
            batch_size=genome.batch_size.value,
            model_path=model_path,
            metrics=metrics,
            genome=genome,
            seed=seed
        )

        return model

    @classmethod
    def from_config(
        cls, 
        model_name : str,
        model_config_path : str, 
        num_ifos : int, 
        optimizer,
        loss,
        num_onsource_samples : int = None, 
        num_offsource_samples : int = None,
        model_path : Path = Path("./models"),
        seed : int = None
    ):
        """
        Loads a model configuration from a file and sets up the model according to the configuration.

        Args:
        - model_config_path (str): Path to the model configuration file.
        - num_ifos (int): Number of interferometers.
        - num_onsource_samples (int): Number of on-source samples.
        - num_offsource_samples (int): Number of off-source samples.
        - gf (module): A module containing custom layer classes and functions.

        Returns:
        - tuple: A tuple containing input configurations, output configuration, and hidden layers.
        """

        if seed is None:
            seed = gf.Defaults.seed

        if num_onsource_samples is None:
            # Use WindowSpec.default() for consistent calculation
            num_onsource_samples = gf.WindowSpec.default().num_onsource_samples
        if num_offsource_samples is None:
            num_offsource_samples = gf.WindowSpec.default().num_offsource_samples

        with open(model_config_path) as file:
            model_config = json.load(file)

        # Replace placeholders in input configurations
        input_configs = model_config["inputs"]
        replacements = {
            "num_ifos": num_ifos,
            "num_onsource_samples": num_onsource_samples,
            "num_offsource_samples": num_offsource_samples
        }
        input_configs = gf.replace_placeholders(input_configs, replacements=replacements)

        # Get output configuration
        output_config = model_config["outputs"][0]

        # Process each layer and create corresponding layer objects
        hidden_layers = []
        for index, layer_config in enumerate(model_config["layers"]):
            name = layer_config.get("name", f"layer_{index}")

            if "type" not in layer_config:
                raise Exception(f"Layer: {name} missing type!")

            layer_type = layer_config["type"]
            # Match layer type and add appropriate layer
            match layer_type:       
                case "Whiten":
                    hidden_layers.append(gf.WhitenLayer())
                case "WhitenPass":
                    hidden_layers.append(gf.WhitenPassLayer())
                case "Flatten":
                    hidden_layers.append(gf.FlattenLayer())
                case "Dense":
                    hidden_layers.append(gf.DenseLayer(
                        units=layer_config.get("num_neurons", 64), 
                        activation=layer_config.get("activation", "relu")
                    ))
                case "Conv":
                    hidden_layers.append(gf.ConvLayer(
                        filters=layer_config.get("num_filters", 16), 
                        kernel_size=layer_config.get("filter_size", 16), 
                        activation=layer_config.get("activation", "relu"),
                        strides=layer_config.get("filter_stride", 1), 
                        dilation=layer_config.get("filter_dilation", 0)
                    ))
                case "Pool":
                    hidden_layers.append(gf.PoolLayer(
                        pool_size=layer_config.get("size", 16),
                        strides=layer_config.get("stride", 16)
                    ))
                case "Drop":
                    hidden_layers.append(gf.DropLayer(
                        layer_config.get("rate", 0.5)
                    ))
                case _:
                    raise ValueError(f"Layer type '{layer_type}' not recognized")

            # Add dropout layer if specified
            if hasattr(layer_config, "dropout_present"):
                if layer_config.dropout_present:
                    hidden_layers.append(gf.DropLayer(
                        layer_config.get("dropout_value", 0.5)
                    ))
            if hasattr(layer_config, "batch_normalisation_present"):
                if layer_config.batch_normalisation_present:
                    hidden_layers.append(gf.BatchNormLayer(BaseLayer))
            if hasattr(layer_config, "pooling_present"):
                if layer_config.pooling_present:
                    hidden_layers.append(
                        gf.PoolLayer(
                            pool_size=layer_config.get("pooling_size", 16),
                            strides=layer_config.get("pooling_stride", 16)
                        )
                    )

        model = cls(
            model_name,
            hidden_layers, 
            input_configs=input_configs, 
            output_config=output_config,
            optimizer=optimizer, 
            loss=loss,
            model_path=model_path, 
            seed=seed
        )

        return model

    @classmethod
    def load(
            cls,
            name : str,
            model_load_path : Path,
            input_configs : Union[List[Dict], Dict],
            output_config : dict,
            num_ifos : int = None,
            optimizer: str = None, 
            loss: str = None, 
            training_config : dict = None,
            hidden_layers = None,
            model_config_path = None,
            num_onsource_samples : int = None, 
            num_offsource_samples : int = None,
            model_path : Path = None,
            force_overwrite : bool = False,
            load_genome : bool = False, 
            dataset_args : Union[Dict, None]  = None,
            genome : None = None,
            seed : int = None
        ):
        
        if num_onsource_samples is None:
            # Use WindowSpec.default() for consistent calculation
            num_onsource_samples = gf.WindowSpec.default().num_onsource_samples
        if num_offsource_samples is None:
            num_offsource_samples = gf.WindowSpec.default().num_offsource_samples

        if model_path is None:
            model_path = model_load_path

        if hidden_layers is not None and model_config_path is not None:
            logger.warning("When attempting to load model, hidden layers and model_config_path are both not none. Using hidden layers.")
        
        blueprint_exists = True
        if hidden_layers is not None:
            model = cls(
                name,
                hidden_layers, 
                input_configs=input_configs, 
                output_config=output_config,
                training_config=training_config,
                optimizer=optimizer, 
                loss=loss,
                model_path=model_path,
                seed=seed
            )
        elif model_config_path is not None:
            model = cls.from_config(
                name,
                model_config_path=model_config_path, 
                num_ifos=num_ifos, 
                optimizer=optimizer,
                loss=loss,
                num_onsource_samples=num_onsource_samples, 
                num_offsource_samples=num_offsource_samples,
                model_path=model_path,
                seed=seed
            )
        elif genome is not None:
            model = cls.from_genome(
                genome=genome, 
                name=name,
                input_configs=input_configs, 
                output_config=output_config,
                training_config=training_config,
                dataset_args=dataset_args, 
                model_path=model_path,
                metrics=[],
                seed=seed
            )
        elif load_genome:
            genome = gf.ModelGenome.load(model_path / "genome")

            model = cls.from_genome(
                genome=genome, 
                name=name,
                input_configs=input_configs, 
                output_config=output_config,
                training_config=training_config,
                dataset_args=dataset_args, 
                model_path=model_path,
                metrics=[],
                seed=seed
            )
        else:  
            blueprint_exists = False
            model = cls(
                name,
                [], 
                input_configs=input_configs, 
                output_config=output_config,
                optimizer=optimizer, 
                loss=loss,
                model_path=model_path,
                training_config=training_config,
                seed=seed
            )
        
        # Check if the model file exists
        if os.path.exists(model_path) and not force_overwrite:
            try:
                # Try to load the model
                logger.info(f"Loading model from {model_path}")
                loaded_model = keras.models.load_model(model_path)
                model.model = loaded_model
                model.loaded=True

                return model

            except Exception as e:
                logger.error(f"Error loading model: {e}")
                if blueprint_exists:
                    logger.info("Using new model...")
                    return model
                elif load_genome is True:
                    genome_path = Path(f"{model_path}/genome")
                    if genome_path.exists():
                        genome = gf.ModelGenome.load(genome_path)
                        
                        return cls.from_genome(
                            genome=genome, 
                            name=name,
                            input_configs=input_configs, 
                            output_config=output_config,
                            training_config=training_config,
                            dataset_args=dataset_args, 
                            model_path=model_path,
                            metrics=[],
                            seed=seed
                        )
                    else: 
                        raise ValueError("No genome exists!")
                else:
                    raise ValueError("No default model blueprint exists!")
        else:
            # If the model doesn't exist, build a new one
            if blueprint_exists:
                logger.info("No saved model found. Using new model...")
                return model
            else:
                raise ValueError("No default model blueprint exists!")

    def build(
            self, 
            input_configs : Union[List[Dict], Dict],
            output_config : dict,
            model_path : Path = None,
            metrics : list = []
        ):

        """
        Builds the model.
        
        Args:
        input_configs: Dict
            Dictionary containing input information.
        output_config: Dict
            Dictionary containing input information.
        model_path: Path
            Path to save model data.
        """        

        self.model_path = model_path

        if not isinstance(input_configs, list):
            input_configs = [input_configs]
        
        # Create input tensors based on the provided configurations
        inputs = {
            config["name"]: keras.Input(shape=config["shape"], name=config["name"]) for config in input_configs
        }

        # The last output tensor, starting with the input tensors
        last_output_tensors = list(inputs.values())

        for layer in self.layers:
            new_layers = self.build_hidden_layer(layer)
            for new_layer in new_layers:
                # Apply the layer to the last output tensor(s)
                if isinstance(new_layer, gf.Whiten):
                    # Whiten expects a list of tensors
                    last_output_tensors = [new_layer(last_output_tensors)]
                else:
                    # Apply the layer to each of the last output tensors (assuming they can all be processed by this layer)
                    last_output_tensors = [new_layer(tensor) for tensor in last_output_tensors]
        
        # Build output layer
        output_tensor = self.build_output_layer(last_output_tensors[-1], output_config)

        self.model = keras.Model(inputs=inputs, outputs=output_tensor)
        
        # If metrics is empty use best guess
        if not metrics:
            match output_config["type"]:
                case "normal":
                    metrics = [keras.metrics.RootMeanSquaredError()]
                case "binary":
                    metrics = [keras.metrics.BinaryAccuracy()]
        
        # Compile model
        self.model.compile(
            optimizer = self.optimizer.value, 
            loss = self.loss.value,
            metrics = metrics
        )
    
    def build_hidden_layer(
            self,
            layer : BaseLayer
        ):

        new_layers = []

        # Get layer type:
        match layer.layer_type:       
            case "Whiten":
                new_layers.append(gf.Whiten())
                new_layers.append(gf.Reshape())
            case "WhitenPass":
                new_layers.append(gf.WhitenPass())
                new_layers.append(gf.Reshape())
            case "Flatten":
                new_layers.append(layers.Flatten())
            case "Dense":
                new_layers.append(layers.Dense(
                        layer.units.value, 
                        activation=layer.activation.value
                    ))
            case "Convolutional":
                new_layers.append(layers.Conv1D(
                        layer.filters.value, 
                        (layer.kernel_size.value,), 
                        strides=(layer.strides.value,), 
                        activation=layer.activation.value,
                        padding = layer.padding.value
                    ))
            case "Pooling":
                new_layers.append(layers.MaxPool1D(
                        (layer.pool_size.value,),
                        strides=(layer.strides.value,),
                        padding = layer.padding.value
                    ))
            case "Dropout":
                new_layers.append(layers.Dropout(
                        layer.rate.value
                    ))
            case "BatchNorm":
                new_layers.append(layers.BatchNormalization())
            case _:
                raise ValueError(
                    f"Layer type '{layer.layer_type.value}' not recognized"
                )
            
        if hasattr(layer, "dropout_present"):
            if layer.dropout_present.value:
                new_layers.append(layers.Dropout(
                    layer.dropout_value.value
                ))
        
        if hasattr(layer, "batch_normalisation_present"):
            if layer.batch_normalisation_present.value:
                new_layers.append(layers.BatchNormalization())

        if hasattr(layer, "pooling_present"):
            if layer.pooling_present:
                new_layers.append(
                    layers.MaxPool1D(
                        (layer.pooling_size.value,),
                        strides=(layer.pooling_stride.value,),
                        padding = layer.padding.value
                    )
                )
        
        # Return new layer type:
        return new_layers
            
    def build_output_layer(self, last_output_tensor, output_config):
        # Flatten the last output tensor
        #x = tf.keras.layers.Flatten()(last_output_tensor)

        x = last_output_tensor

        # Based on the output type, add the final layers functionally
        if output_config["type"] == "normal":
            x = layers.Dense(
                    2, 
                    activation='linear', 
                    dtype='float32', 
                    bias_initializer=keras.initializers.Constant([1.0, 2.0])
                )(x)
            output_tensor = IndependentFoldedNormal(1, name=output_config["name"])(x)
        elif output_config["type"] == "binary":
            x = layers.Flatten()(x)
            output_tensor = layers.Dense(
                                1, 
                                activation='sigmoid', 
                                dtype='float32',
                                name=output_config["name"]
                            )(x)
        else:
            raise ValueError(f"Unsupported output type: {output_config['type']}")

        return output_tensor
        
    def train(
        self, 
        training_dataset = None, 
        validation_dataset = None,
        validate_args : dict = None,
        training_config: dict = None,
        force_retrain : bool = True,
        max_epochs_per_lesson = None,
        callbacks = None,
        heart = None
    ):
        """
        Trains the model.
        
        Args:
        training_dataset: Dataset to train on.
        num_epochs: Number of epochs to train for.
        """ 

        if validation_dataset is None and validate_args is None:
            raise ValueError("No validation dataset!")
        
        if validation_dataset is not None and not validate_args:
            raise ValueError("Validation argument and validateion datasets")

        if validate_args is not None:

            if self.genome is not None:
                validate_args["onsource_duration_seconds"] = self.genome.onsource_duration_seconds.value
                validate_args["offsource_duration_seconds"] = self.genome.offsource_duration_seconds.value
                validate_args["sample_rate_hertz"] = self.genome.sample_rate_hertz.value
                validate_args["num_examples_per_batch"] = 32

                if self.genome.layer_genomes[0].value.layer_type == "WhitenPass":
                    validate_args["input_variables"] = [
                        gf.ReturnVariables.ONSOURCE
                    ]

            validation_dataset = gf.Dataset(
                **deepcopy(validate_args),
                group="validate"
            )

        if training_dataset is None:
            training_dataset = self.training_dataset
        elif self.training_dataset is not None:
            raise ValueError("Warning train dataset passed even though internal dataset it set.")

        if training_config is None:
            if self.training_config is not None:
                training_config = self.training_config
            else:
                raise ValueError("Missing training config")

        if callbacks is None:
            callbacks = []

        gf.ensure_directory_exists(self.model_path)

        checkpoint_monitor = "val_loss"
        if not force_retrain:
            model_path = self.model_path

            history_data = gf.load_history(self.model_path)
            if history_data != {}:
                best_metric = min(history_data[checkpoint_monitor]) #assuming loss for now
                best_epoch = np.argmin(history_data[checkpoint_monitor])
                initial_epoch = len(history_data[checkpoint_monitor])

                if initial_epoch - best_epoch > training_config["patience"]:
                    logger.info(
                        f"Model already completed training. Skipping! Current epoch {initial_epoch}, best epoch {best_epoch}."
                    )
                    self.model = keras.models.load_model(
                        self.model_path
                    )
                    self.metrics.append(history_data)
                    
                    return False
            else:
                initial_epoch = 0
                model_path = None
                best_metric = None
        else:
            initial_epoch = 0
            model_path = None
            best_metric = None
            gf.save_dict_to_hdf5({}, self.model_path / "history.hdf5", True)

        if self.genome is not None:
            self.genome.save(self.model_path / "genome")
        
        if max_epochs_per_lesson is None:
            current_max_epoch = training_config["max_epochs"]
        else:
            current_max_epoch = initial_epoch + max_epochs_per_lesson

        early_stopping = gf.EarlyStoppingWithLoad(
                monitor  = checkpoint_monitor,
                patience = training_config["patience"],
                model_path=model_path
            )

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            self.model_path,
            monitor=checkpoint_monitor,
            save_best_only=True,
            save_freq="epoch", 
            initial_value_threshold=best_metric
        )

        if force_retrain:
            logger.info("Forcing retraining!")
        else:
            logger.info(f"Resuming from {initial_epoch}, current history : {history_data}")
        
        history_saver = gf.CustomHistorySaver(self.model_path, force_overwrite=force_retrain)
        wait_printer = gf.PrintWaitCallback(early_stopping)
        
        callbacks += [
            early_stopping,
            model_checkpoint,
            history_saver,
            wait_printer
            #tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)
        ]
        
        num_batches = training_config["num_examples_per_epoc"] // self.batch_size.value
        num_validation_batches = training_config["num_validation_examples"] // 32
        
        verbose : int = 1
        if gf.is_redirected():
            verbose : int = 2
            
            if heart is not None:
                callbacks += [gf.HeartbeatCallback(heart, 32)]
        else:
            logger.info("Training... or at least trying to.")

        self.metrics.append(
            self.model.fit(
                training_dataset,
                validation_data=validation_dataset,
                validation_steps=num_validation_batches,
                epochs=current_max_epoch, 
                initial_epoch = initial_epoch,
                steps_per_epoch = num_batches,
                callbacks = callbacks,
                batch_size = self.batch_size.value,
                verbose=verbose
            )
        )

        if heart is not None:
            heart.beat()

        gf.save_dict_to_hdf5(
            self.metrics[0].history, 
            self.model_path / "metrics", 
            force_overwrite=False
        )

        return True
            
    def validate(
        self, 
        validate_args : dict,
        efficiency_config : dict,
        far_config : dict,
        roc_config : dict,
        model_path : Path,
        heart : None
    ):
        validation_file_path : Path = Path(model_path) / "validation_data.h5"

        if self.genome is not None:
            validate_args["onsource_duration_seconds"] = self.genome.onsource_duration_seconds.value
            validate_args["offsource_duration_seconds"] = self.genome.offsource_duration_seconds.value
            validate_args["sample_rate_hertz"] = self.genome.sample_rate_hertz.value

            if self.genome.layer_genomes[0].value.layer_type == "WhitenPass":
                validate_args["input_variables"] = [
                    gf.ReturnVariables.ONSOURCE
                ]
        
        # Validate model:
        validator = gf.Validator.validate(
                self.model, 
                self.name,
                dataset_args=deepcopy(validate_args),
                efficiency_config=efficiency_config,
                far_config=far_config,
                roc_config=roc_config,
                checkpoint_file_path=validation_file_path,
                heart=heart
            )

        validator.plot(
            model_path / "validation_plots.html"
        )

    def test(self, validation_datasets, num_batches: int):
        """
        Tests the model.
        
        Args:
        validation_datasets: Dataset to test on.
        batch_size: Batch size to use when testing.
        """
        
        self.fitness.append(1.0 / self.model.evaluate(validation_datasets, steps=num_batches)[0])
        
        return self.fitness[-1]
        
    def summary(self):
        """
        Prints a summary of the model.
        """
        self.model.summary()
    
    @staticmethod
    def crossover(parent1: 'Model', parent2: 'Model') -> 'Model':
        """
        Creates a new model whose hyperparameters are a combination of two parent models.
        The child model is then returned.
        """
        # Determine the shorter and longer layers lists
        short_layers, long_layers = (parent1.layers, parent2.layers) if len(parent1.layers) < len(parent2.layers) else (parent2.layers, parent1.layers)

        # Choose a random split point in the shorter layers list
        split_point = np.random.randint(1, len(short_layers))

        # Choose which parent to take the first part from
        first_part, second_part = (short_layers[:split_point], long_layers[split_point:]) if np.random.random() < 0.5 else (long_layers[:split_point], short_layers[split_point:])
        child_layers = first_part + second_part

        child_model = Model(child_layers, parent1.optimizer, parent1.loss, parent1.batch_size)

        return child_model
    
    def mutate(self, mutation_rate: float) -> 'Model':
        """
        Returns a new model with mutated layers based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_model: New Model instance with potentially mutated layers.
        """
        mutated_layers = [layer.mutate(mutation_rate) for layer in self.layers]
        mutated_model = Model(mutated_layers, self.optimizer, self.loss, self.batch_size)

        return mutated_model
@dataclass
class PopulationSector:
    name : str
    save_directory : Path
    models : List = None
    num_models : int = 0
    fitnesses : float = None
    accuracies : List = None
    losses : List = None

    mean_accuracy_history : List = None
    mean_fitness_history : List = None
    mean_loss_history : List = None

    def __post_init__(self):
        if self.models is None:
            self.models = []
        self.num_models = len(self.models)
        self.fitnesses = list(np.ones(self.num_models))
        self.accuracies = list(np.zeros(self.num_models))
        self.losses = list(np.ones(self.num_models))

        self.mean_accuracy_history = []
        self.mean_fitness_history = []
        self.mean_loss_history = []

    def add(self, new_model):
        self.models.append(new_model)
        self.fitnesses.append(1)
        self.accuracies.append(0)
        self.losses.append(1)
        self.num_models += 1

    def transfer(self, population, index):
        self.models.append(population.models.pop(index))
        self.fitnesses.append(population.fitnesses.pop(index))
        self.num_models += 1
    
    def save(self):
        np.save(self.save_directory / f"{self.name}_fitnesses", self.fitnesses)
        np.save(self.save_directory / f"{self.name}_accuracies", self.fitnesses)
        np.save(self.save_directory / f"{self.name}_losses", self.losses)

        np.save(self.save_directory / f"{self.name}_fitness_history", self.mean_accuracy_history)
        np.save(self.save_directory / f"{self.name}_accuracy_history", self.mean_fitness_history)
        np.save(self.save_directory / f"{self.name}_loss_history", self.mean_loss_history)

class Population:
    def __init__(
        self, 
        num_population_members: int,
        default_genome: gf.ModelGenome,
        population_directory_path : Path = None,
        seed : int = None
    ):

        if population_directory_path is None:
            current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            population_directory_path = Path(f"./population_{current_timestamp}")

        self.repo = gf.get_current_repo()

        self.generation = 0

        self.num_population_members = num_population_members
        self.population_directory_path = Path(population_directory_path)
        self.default_genome = default_genome
        self.current_id = 0

        self.orchard = PopulationSector("orchard", population_directory_path)
        self.nursary = PopulationSector("nursary", population_directory_path)
        self.lumberyard = PopulationSector("lumberyard", population_directory_path)

        if seed == None:
            seed = gf.Defaults.seed

        self.rng = default_rng(seed)

        self.initilize()
        self.save()
    
    def save(self, path = None):
        if path is None:
            path = self.population_directory_path / "checkpoint.pkl"

        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        if path.exists():
            with open(path, 'rb') as file:
                return pickle.load(file)
        else:
            return None

    def load_model(self):
        model_number = self.current_id
        self.current_id += 1

        model_name = f"model_{model_number}"
        model_path = self.population_directory_path / f"generation_{self.generation}/{model_name}"
        
        try:
            genome = gf.ModelGenome.load(model_path / "genome")
        except:
            raise FileNotFoundError(f"Model genome {self.generation}/{model_name} does not exist.")
        
        model = {
            "name" : model_name,
            "number" : model_number,
            "path" : model_path,
            "genome" : genome
        }

        self.nursary.add(model)

    def add_model(self, genome):

        model_number = self.current_id

        model_name = f"model_{model_number}"
        model_path = self.population_directory_path / f"generation_{self.generation}/{model_name}"

        gf.ensure_directory_exists(model_path)
        model = {
            "name" : model_name,
            "number" : model_number,
            "path" : model_path,
            "genome" : genome
        }

        genome.save(model_path / "genome")

        self.nursary.add(model)
        self.current_id += 1

    def initilize(self):
        if self.generation == 0:
            for j in tqdm(range(self.num_population_members)): 
                if not Path(
                    self.population_directory_path / f"generation_{self.generation}/model_{j}/genome"
                ).exists():
                    self.random_sapling()
                else:
                    self.load_model()
                
    def roulette_wheel_selection(self, fitnesses):
        """
        Performs roulette wheel selection on the population.

        Args:
            population (list): The population of individuals.
            fitnesses (list): The fitness of each individual in the population.

        Returns:
            The selected individual from the population.
        """

        # Convert the fitnesses to probabilities.
        total_fit = sum(fitnesses)
        prob = [fit/total_fit for fit in fitnesses]

        # Calculate the cumulative probabilities.
        cumulative_probs = np.cumsum(prob)

        # Generate a random number in the range [0, 1).
        r = np.random.rand()

        # Find the index of the individual to select.
        for i in range(len(fitnesses)):
            if r <= cumulative_probs[i]:
                return i

        # If we've gotten here, just return the last individual in the population.
        # This should only happen due to rounding errors, and should be very rare.
        return 0

    def train_generation(self):

        initial_processes = []
        for model in self.nursary.models:    
            if not Path(model["path"] / "validation_plots.html").exists():
                initial_processes.append(gf.Process(
                    f"python train.py --path {model['path']}", model["name"], 
                    tensorflow_memory_mb=1000, 
                    cuda_overhead_mb=500, 
                    initial_restart_count=1
                ))
            else:
                logger.info(
                    f"Model {model['path']} has already completed validation, skipping."
                )

        if initial_processes:
            manager = gf.Manager(
                initial_processes,
                max_restarts=8,
                restart_timeout_seconds=3600.0,
                process_start_wait_seconds=3.0,
                management_tick_length_seconds=5.0,
                max_num_concurent_processes=60,
                log_directory_path = (
                    self.population_directory_path / f"generation_{self.generation}/logs/"
                )
            )

            interval_seconds : float = 60.0
            last_save_time = time.time()  # Tracks the last save time.

            while manager:
                manager()  # Call the manager function.
        
        if not initial_processes:
            logger.info(
                f"Generation {self.generation} empty or completed. Skipping."
            )

    def load_config(
        self,
        path : Path
    ):
        """Loads configuration parameters from a JSON file."""
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def train(
        self, 
        num_generations
    ):  

        current_dir = Path(__file__).resolve().parent.parent

        email_config_path = Path("./gravyflow/alert_settings.json")
        email_config = self.load_config(email_config_path)

        gf.send_email(
            f"Started optimising {self.population_directory_path}..", 
            (f"Started optimization of {self.num_population_members} population members over {num_generations}"
            " generations"), 
            email_config['recipient_email'], 
            Path(email_config['email_config_path'])
        )

        logger.info("Starting training...")

        """
        self.generation = 5
        self.current_id = 5000

        self.nursary = PopulationSector("nursary", self.population_directory_path)
        for _ in range(self.num_population_members):
            if (self.population_directory_path / f"generation_{self.generation}/model_{self.current_id}/genome").exists():
                self.load_model()
            else:
                logger.warning("Can't find file!")
        """

        for generation_index in range(self.generation, num_generations):
            logger.info(f"Training Generation: {self.generation}")
            
            self.train_generation()
            self.generation += 1

            self.nursary.models=sorted(self.nursary.models, key=lambda x: x['number'])
            self.nursary.fitnesses = self.load_and_calculate_fitness(
                generation_index
            )

            logger.info(
                f"Current Fitnesses: {self.nursary.fitnesses}"
            )

            gf.send_email(
                f"Generation {self.generation} completed.", 
                f"Completed generation, current fitnesses {self.nursary.fitnesses}", 
                email_config['recipient_email'], 
                Path(email_config['email_config_path'])
            )

            while self.nursary.models:
                self.orchard.transfer(self.nursary, -1)
                        
            for _ in range(self.num_population_members):
                if (self.population_directory_path / f"/generation_{self.generation}/model_{self.current_id}/genome").exists():
                    self.load_model()
                else:
                    self.germinate_sapling()

            while self.orchard.models:
                self.lumberyard.transfer(self.orchard, -1)

            self.save()
    
    def germinate_sapling(self):

        parent_a_index = self.roulette_wheel_selection(self.orchard.fitnesses)
        parent_b_index = self.roulette_wheel_selection(self.orchard.fitnesses)

        parent_a = self.orchard.models[parent_a_index]
        parent_b = self.orchard.models[parent_b_index]

        #Crossover
        new_genome = deepcopy(parent_a["genome"])
        new_genome.reseed(self.rng.integers(1E10))
        new_genome.crossover(deepcopy(parent_b["genome"]))

        #Mutate
        new_genome.mutate(0.05)

        logger.info('Germinated new model.')

        self.add_model(new_genome)

    def random_sapling(self):
        new_genome = deepcopy(self.default_genome)  
        new_genome.reseed(self.rng.integers(1E10))
        new_genome.randomize()

        logger.info('Randomised new model.')

        self.add_model(new_genome)    

    def load_and_calculate_fitness(
            self,
            generation, 
            scaling_threshold = 8
        ):

        self.population_directory_path = Path(self.population_directory_path)

        directory_path = self.population_directory_path / f"generation_{generation}/"

        model_indices = []
        for entry in directory_path.iterdir():
            if entry.name.startswith(f"model_"):
                model_indices.append(int(entry.name.split("_")[-1]))

        fitnesses = np.zeros(self.num_population_members)

        for entry in directory_path.iterdir():
            if entry.name.startswith(f"model_"):
                
                model_index = int(entry.name.split("_")[-1]) - self.num_population_members*generation
                try:
                    history = gf.load_history(entry)
                    validator = gf.validate.Validator.load(entry / "validation_data.h5")
                    genome = gf.ModelGenome.load(entry / "genome")
                    
                    score_threshold = list(gf.validate.calculate_far_score_thresholds(
                        validator.far_scores, 
                        genome.onsource_duration_seconds.value,
                        [10E-4]
                    ).values())[0][1]

                    efficiency_data = validator.efficiency_data

                    valid_scores = np.array([])
                    for scaling, scores in zip(efficiency_data["scalings"], efficiency_data["scores"]):
                        if scaling > scaling_threshold:
                            valid_scores = np.append(valid_scores, scores.flatten())
                    
                    fitnesses[model_index] = calculate_fitness(valid_scores, score_threshold, history)
                except Exception as e:
                    logger.error(e)
                    pass

        return list(fitnesses)

def calculate_fitness(valid_scores: List[float], score_threshold : float, history: Dict[str, List[float]]) -> float:
    """
    Calculate the fitness based on the number of valid scores above a threshold,
    the list of valid scores, and the history of validation losses. Sets fitness to
    0 if any term is NaN or infinite.

    Parameters
    ----------
    num_valid_scores_above : int
        Number of valid scores above the threshold.
    valid_scores : List[float]
        List of valid scores.
    history : Dict[str, List[float]]
        History dictionary containing validation loss under the key "val_loss".

    Returns
    -------
    float
        The calculated fitness value or 0 if any term is NaN or infinite.

    """
    num_valid_scores_above = len(valid_scores[valid_scores > score_threshold])

    len_valid_scores = len(valid_scores)
    min_val_loss = np.min(history["val_loss"])
    
    # Pre-calculation checks for NaN or inf
    terms = [num_valid_scores_above, len_valid_scores, min_val_loss]
    if np.any(np.isnan(terms)) or np.any(np.isinf(terms)):
        logger.info("One or more terms are NaN or infinite. Setting fitness to 0.")
        return 0.0
    
    fitness = 1.0 / (1.0 - (num_valid_scores_above / len_valid_scores) + min_val_loss + 1E-8)
    
    # Post-calculation check for NaN or inf in the fitness result
    if np.isnan(fitness) or np.isinf(fitness):
        logger.info("Calculated fitness is NaN or infinite. Setting fitness to 0.")
        return 0.0

    return fitness

def snake_case_to_capitalised_first_with_spaces(text):
    """
    Convert a string from snake_case to Capitalised First With Spaces format.
    """

    text.replace('val', 'validate') if 'val' in text else text
    # Split the string at underscores
    words = text.split('_')
    
    # Capitalise the first letter of each word and join them with spaces
    return ' '.join(word.capitalize() for word in words)

