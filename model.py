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

def negative_loglikelihood(targets, estimated_distribution):
    
    targets = tf.cast(targets, dtype = tf.float32)
    return -estimated_distribution.log_prob(targets)

def negative_loglikelihood_(y_true, y_pred):
    loc, scale = tf.unstack(tf.cast(y_pred, dtype = tf.float32), axis=-1)
    y_true = tf.cast(y_true, dtype = tf.float32)

    truncated_normal = tfp.distributions.TruncatedNormal(
        loc,
        scale + 1.0E-5,
        0.0,
        1000.0,
        validate_args=False,
        allow_nan_stats=True,
        name='TruncatedNormal'
    )
        
    return -truncated_normal.log_prob(y_true)

tfd = tfp.distributions
tfpl = tfp.layers

class IndependentGamma(tfpl.DistributionLambda):
    """An independent Gamma Keras layer."""

    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        super(IndependentGamma, self).__init__(
            lambda t: self.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )
        self._event_shape = event_shape
        self._validate_args = validate_args

    def new(self, params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentGamma'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = tf.reshape(event_shape, [-1])
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape
            ], axis=0)
            alpha_params, beta_params = tf.split(params, 2, axis=-1)
            alpha_params = tf.nn.softplus(tf.cast(alpha_params, dtype = tf.float32)) + 1.0E-5
            beta_params = tf.nn.softplus(tf.cast(beta_params, dtype = tf.float32)) + 1.0E-5
            
            return tfd.Independent(
                tfd.Gamma(
                    concentration=tf.reshape(alpha_params, output_shape),
                    rate=tf.reshape(beta_params, output_shape),
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentGamma_params_size'):
            event_shape = tf.convert_to_tensor(
                event_shape, 
                name='event_shape', 
                dtype_hint=tf.int32
            )
            return np.int32(2) * np.prod(event_shape)

    def get_config(self):
        """Returns the config of this layer."""
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': self.convert_to_tensor_fn,
            'validate_args': self._validate_args
        }
        base_config = super(IndependentGamma, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class IndependentFoldedNormal(tfpl.DistributionLambda):
    """An independent folded normal Keras layer."""

    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        super(IndependentFoldedNormal, self).__init__(
            lambda t: self.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )
        self._event_shape = event_shape
        self._validate_args = validate_args

    def new(self, params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentFoldedNormal'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = tf.reshape(event_shape, [-1])
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape
            ], axis=0)
            loc_params, scale_params = tf.split(params, 2, axis=-1)
            loc_params = tf.cast(loc_params, dtype = tf.float32)  + 1.0E-6
            scale_params = tf.cast(scale_params, dtype = tf.float32) + 1.0E-6

            return tfd.Independent(
                tfd.TransformedDistribution(
                    distribution=tfd.Normal(
                        loc=tf.math.softplus(tf.reshape(loc_params, output_shape)),
                        scale=tf.math.softplus(tf.reshape(scale_params, output_shape)),
                        validate_args=validate_args),
                    bijector=tfb.AbsoluteValue(),
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentFoldedNormal_params_size'):
            event_shape = tf.convert_to_tensor(event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(2) * np.prod(event_shape)

    def get_config(self):
        """Returns the config of this layer."""
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': self.convert_to_tensor_fn,
            'validate_args': self._validate_args
        }
        base_config = super(IndependentFoldedNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IndependentTruncNormal(tfpl.DistributionLambda):
    """An independent truncated normal Keras layer."""

    def __init__(self,
                 event_shape=(),
                 low=  0.000,
                 high= 100.0, #float("inf"),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        self.low = low
        self.high = high
        super(IndependentTruncNormal, self).__init__(
            lambda t: self.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )
        self._event_shape = event_shape
        self._validate_args = validate_args

    def new(self, params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentTruncNormal'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = tf.reshape(event_shape, [-1])
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape
            ], axis=0)
            loc_params, scale_params = tf.split(params, 2, axis=-1)
            loc_params = tf.cast(loc_params, dtype = tf.float32)  + 1.0E-5
            scale_params = tf.cast(loc_params, dtype = tf.float32) + 1.0E-5
            
            return tfd.Independent(
                tfd.TruncatedNormal(
                    loc=tf.reshape(loc_params, output_shape),
                    scale=tf.math.softplus(tf.reshape(scale_params, output_shape)),
                    low=self.low,
                    high=self.high,
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentTruncNormal_params_size'):
            event_shape = tf.convert_to_tensor(event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(2) * np.prod(event_shape)

    def get_config(self):
        """Returns the config of this layer."""
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': self.convert_to_tensor_fn,
            'validate_args': self._validate_args
        }
        base_config = super(IndependentTruncNormal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

tfb = tfp.bijectors
class BetaPrime(tfb.Bijector):
    """Bijector for the beta prime distribution."""
    def __init__(self, validate_args=False, name="beta_prime"):
        super(BetaPrime, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=0, name=name)

    def _forward(self, x):
        return x / (1 - x)

    def _inverse(self, y):
        return y / (1 + y)

    def _forward_log_det_jacobian(self, x):
        return - tf.math.log1p(-x)

    def _inverse_log_det_jacobian(self, y):
        return - tf.math.log1p(y)

class IndependentBetaPrime(tfpl.DistributionLambda):
    """An independent Beta prime Keras layer."""
    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfd.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        super(IndependentBetaPrime, self).__init__(
            lambda t: self.new(t, event_shape, validate_args),
            convert_to_tensor_fn,
            **kwargs
        )
        self._event_shape = event_shape
        self._validate_args = validate_args

    def new(self, params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentBetaPrime'):
            
            params = tf.cast(tf.convert_to_tensor(params, name='params'), tf.float32)
            event_shape = tf.reshape(event_shape, [-1])
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape
            ], axis=0)
            concentration1_params, concentration0_params = tf.split(params, 2, axis=-1)
            concentration1_params = tf.math.softplus(tf.reshape(concentration1_params, output_shape))
            concentration0_params = tf.math.softplus(tf.reshape(concentration0_params, output_shape))

            return tfd.Independent(
                tfd.TransformedDistribution(
                    distribution=tfd.Beta(
                        concentration1=concentration1_params,
                        concentration0=concentration0_params,
                        validate_args=validate_args),
                    bijector=BetaPrime(),
                    validate_args=validate_args),
                reinterpreted_batch_ndims=tf.size(event_shape),
                validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentBetaPrime_params_size'):
            event_shape = tf.convert_to_tensor(event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(2) * np.prod(event_shape)

    def get_config(self):
        """Returns the config of this layer."""
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': self.convert_to_tensor_fn,
            'validate_args': self._validate_args
        }
        base_config = super(IndependentBetaPrime, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def adjust_features(features, labels):
    labels['INJECTION_MASKS'] = labels['INJECTION_MASKS'][0]
    return features, labels

@dataclass
class BaseLayer:
    layer_type: str = "Base"
    activation: Union[gf.HyperParameter, str] = None
    mutable_attributes: List = None
    
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
            return tf.reshape(inputs, [tf.shape(inputs)[0], -1, 1])
        elif self.reshaping_mode == 'depthwise':
            # (num_batches, num_steps, num_features)
            return tf.transpose(inputs, perm=[0, 2, 1])
        elif self.reshaping_mode == 'heightwise':
            # (num_batches, num_features, num_steps, 1)
            return tf.expand_dims(inputs, axis=-1)
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

    def __init__(
            self, 
            units: Union[gf.HyperParameter, int] = 64, 
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
        
        self.layer_type = "Dense"
        self.activation = gf.HyperParameter(activation)
        self.units = gf.HyperParameter(units)
        self.mutable_attributes = [self.activation, self.units]

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
        
        self.layer_type = "Flatten"
        self.mutable_attributes = []

@dataclass
class ConvLayer(BaseLayer):
    filters: gf.HyperParameter = 16
    kernel_size: gf.HyperParameter = 16
    strides: gf.HyperParameter = 1
    
    def __init__(self, 
        filters: gf.HyperParameter = 16, 
        kernel_size: gf.HyperParameter = 16, 
        activation: gf.HyperParameter = "relu", 
        strides: gf.HyperParameter = gf.HyperParameter(1),
        dilation: gf.HyperParameter = gf.HyperParameter(0)
        ):
        """
        Initializes a ConvLayer instance.
        
        Args:
        filters: gf.HyperParameter specifying the number of filters in this layer.
        kernel_size: gf.HyperParameter specifying the kernel size in this layer.
        activation: gf.HyperParameter specifying the activation function for this layer.
        strides: gf.HyperParameter specifying the stride length for this layer.
        """
        self.layer_type = "Convolutional"
        self.activation = gf.HyperParameter(activation)
        self.filters = gf.HyperParameter(filters)
        self.kernel_size = gf.HyperParameter(kernel_size)
        self.strides = gf.HyperParameter(strides)
        self.dilation = gf.HyperParameter(dilation)

        self.padding = gf.HyperParameter("same")
        
        self.mutable_attributes = [self.activation, self.filters, self.kernel_size, self.strides, self.dilation]
        
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
        self.layer_type = "Pooling"
        self.pool_size = gf.HyperParameter(pool_size)
        
        if strides is None:
            self.strides = self.pool_size
        else:
            self.strides = gf.HyperParameter(strides)
        
        self.padding = gf.HyperParameter("same")
        self.mutable_attributes = [self.pool_size, self.strides]
        
class DropLayer(BaseLayer):
    rate: gf.HyperParameter = 0.5
    
    def __init__(self, rate: Union[gf.HyperParameter, float]):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: gf.HyperParameter specifying the dropout rate for this layer.
        """
        self.layer_type = "Dropout"
        self.rate = gf.HyperParameter(rate)
        self.mutable_attributes = [self.rate]

class WhitenLayer(BaseLayer):
    
    def __init__(self):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: gf.HyperParameter specifying the whitening for this layer.
        """
        self.layer_type = "Whiten"
        self.mutable_attributes = []

class WhitenPassLayer(BaseLayer):
    
    def __init__(self):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: gf.HyperParameter specifying the whitening for this layer.
        """
        self.layer_type = "WhitenPass"
        self.mutable_attributes = []

"""
class TruncatedNormal(tf.keras.layers.Layer):
    def __init__(self, event_shape=1, **kwargs):
        super(TruncatedNormal, self).__init__(**kwargs)
        self.event_shape = event_shape

    def build(self, input_shape):
        # Input should be of shape (batch_size, 2) 
        # with the first column as the mean and the second column as the standard deviation
        assert input_shape[-1] == 2

    def call(self, inputs):
        loc, scale = tf.unstack(inputs, axis=-1)
        return loc, scale

    def get_config(self):
        return {'event_shape': self.event_shape}
"""

def cap_value(x):
    return K.clip(x, 1.0e-5, 1000)  # values will be constrained to [-1, 1]

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
        genome : gf.ModelGenome = None
    ):
        """
        Initializes a Model instance.
        
        Args:
        layers: List of BaseLayer instances making up the model.
        optimizer: Optimizer to use when training the model.
        loss: Loss function to use when training the model.
        batch_size: Batch size to use when training the model.
        """ 
        self.train_dataset = None
        if dataset_args is not None:
            self.train_dataset = gf.Dataset(**deepcopy(dataset_args)).map(adjust_features)

        self.name = name

        self.genome = genome

        if batch_size is None:
            batch_size = gf.Defaults.num_examples_per_batch

        self.layers = layers
        self.batch_size = gf.HyperParameter(batch_size)
        self.optimizer = gf.HyperParameter(optimizer)
        self.loss = gf.HyperParameter(loss)
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
        metrics : List 
    ):

        layers = []            
        for i in range(genome.num_layers.value):
            layers.append(
                deepcopy(genome.layer_genomes[i].value)
            )
        
        training_config["model_path"] = model_path
        training_config["learning_rate"] = genome.learning_rate.value

        dataset_args["injection_generators"] = [
            generator.return_generator() for generator in genome.injection_generators 
        ]

        #dataset_args["noise_obtainer"].type = genome.noise_type.value
        dataset_args = deepcopy(dataset_args)
        
        # Create an instance of DenseModel with num_neurons list
        model = cls(
            name=name,
            layers=layers, 
            optimizer=genome.optimizer.value, 
            loss=gf.HyperParameter(losses.BinaryCrossentropy()), 
            input_configs=input_configs, 
            output_config=output_config,
            training_config=training_config,
            dataset_args=dataset_args,
            batch_size=genome.batch_size.value,
            model_path=model_path,
            metrics=metrics,
            genome=genome
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
        model_path : Path = Path("./models")
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

        if num_onsource_samples is None:
            num_onsource_samples = int(
                (gf.Defaults.onsource_duration_seconds + 2.0*gf.Defaults.crop_duration_seconds)*
                gf.Defaults.sample_rate_hertz
            )
        if num_offsource_samples is None:
            num_offsource_samples = int(
                gf.Defaults.offsource_duration_seconds*
                gf.Defaults.sample_rate_hertz
            )

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
            if "dropout" in layer_config:
                hidden_layers.append(gf.DropLayer(
                    layer_config.get("dropout", 0.5)
                ))

        model = cls(
            model_name,
            hidden_layers, 
            input_configs=input_configs, 
            output_config=output_config,
            optimizer=optimizer, 
            loss=loss,
            model_path=model_path
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
            force_overwrite = False,
            load_genome = False, 
            dataset_args= None,
            genome=None
        ):
        
        if num_onsource_samples is None:
            num_onsource_samples = int(
                (gf.Defaults.onsource_duration_seconds + 2.0*gf.Defaults.crop_duration_seconds)*
                gf.Defaults.sample_rate_hertz
            )
        if num_offsource_samples is None:
            num_offsource_samples = int(
                gf.Defaults.offsource_duration_seconds*
                gf.Defaults.sample_rate_hertz
            )

        if model_path is None:
            model_path = model_load_path

        if hidden_layers is not None and model_config_path is not None:
            logging.warning("When attempting to load model, hidden layers and model_config_path are both not none. Using hidden layers.")
        
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
                model_path=model_path
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
                model_path=model_path
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
                metrics=[]
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
            )
        
        # Check if the model file exists
        if os.path.exists(model_load_path) and not force_overwrite:
            try:
                # Try to load the model
                logging.info(f"Loading model from {model_load_path}")
                model.model = tf.keras.models.load_model(model_load_path)
                model.loaded=True

                if load_genome:
                    model.genome = gf.ModelGenome.load(model_path / "genome")

                return model

            except Exception as e:
                logging.error(f"Error loading model: {e}")
                if blueprint_exists:
                    logging.info("Using new model...")
                    return model
                else:
                    raise ValueError("No default model blueprint exists!")
        else:
            # If the model doesn't exist, build a new one
            if blueprint_exists:
                logging.info("No saved model found. Using new model...")
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
            config["name"]: tf.keras.Input(shape=config["shape"], name=config["name"]) for config in input_configs
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

        self.model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        
        # If metrics is empty use best guess
        if not metrics:
            match output_config["type"]:
                case "normal":
                    metrics = [tf.keras.metrics.RootMeanSquaredError()]
                case "binary":
                    metrics = [tf.keras.metrics.BinaryAccuracy()]
        
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
                new_layers.append(gf.WhitenPassthrough())
                new_layers.append(gf.Reshape())

            case "Flatten":
                new_layers.append(tf.keras.layers.Flatten())
            case "Dense":
                new_layers.append(tf.keras.layers.Dense(
                        layer.units.value, 
                        activation=layer.activation.value
                    ))
            case "Convolutional":
                new_layers.append(tf.keras.layers.Conv1D(
                        layer.filters.value, 
                        (layer.kernel_size.value,), 
                        strides=(layer.strides.value,), 
                        activation=layer.activation.value,
                        padding = layer.padding.value
                    ))
            case "Pooling":
                new_layers.append(tf.keras.layers.MaxPool1D(
                        (layer.pool_size.value,),
                        strides=(layer.strides.value,),
                        padding = layer.padding.value
                    ))
            case "Dropout":
                new_layers.append(tf.keras.layers.Dropout(
                        layer.rate.value
                    ))
            case _:
                raise ValueError(
                    f"Layer type '{layer.layer_type.value}' not recognized"
                )
        
        # Return new layer type:
        return new_layers
            
    def build_output_layer(self, last_output_tensor, output_config):
        # Flatten the last output tensor
        #x = tf.keras.layers.Flatten()(last_output_tensor)

        x = last_output_tensor

        # Based on the output type, add the final layers functionally
        if output_config["type"] == "normal":
            x = tf.keras.layers.Dense(
                    2, 
                    activation='linear', 
                    dtype='float32', 
                    bias_initializer=tf.keras.initializers.Constant([1.0, 2.0])
                )(x)
            output_tensor = IndependentFoldedNormal(1, name=output_config["name"])(x)
        elif output_config["type"] == "binary":
            x = tf.keras.layers.Flatten()(x)
            output_tensor = tf.keras.layers.Dense(
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
        train_dataset: tf.data.Dataset = None, 
        validate_dataset: tf.data.Dataset = None,
        training_config: dict = None,
        force_retrain : bool = True,
        max_epochs_per_lesson = None,
        callbacks = None,
        heart = None
        ):
        """
        Trains the model.
        
        Args:
        train_dataset: Dataset to train on.
        num_epochs: Number of epochs to train for.
        """ 

        if validate_dataset is None:
            raise ValueError("No validation dataset!")

        if train_dataset is None:
            train_dataset = self.train_dataset

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
                best_epoch = np.argmin(history_data[checkpoint_monitor]) + 1
                initial_epoch = len(history_data[checkpoint_monitor])

                if initial_epoch - best_epoch > training_config["patience"]:
                    logging.info(
                        f"Model already completed training. Skipping! Current epoch {initial_epoch}, best epoch {best_epoch}."
                    )
                    self.model = tf.keras.models.load_model(
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
            logging.info("Forcing retraining!")
        else:
            logging.info(f"Resuming from {initial_epoch}, current history : {history_data}")
        
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
        num_validation_batches = training_config["num_validation_examples"] // self.batch_size.value
        
        verbose : int = 1
        if gf.is_redirected():
            verbose : int = 2
            
            if heart is not None:
                callbacks += [gf.HeartbeatCallback(heart, 32)]

        self.metrics.append(
            self.model.fit(
                train_dataset,
                validation_data=validate_dataset,
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
        dataset_arguments : dict,
        efficiency_config : dict,
        far_config : dict,
        roc_config : dict,
        model_path : Path,
        heart : None
    ):
        validation_file_path : Path = Path(model_path) / "validation_data.h5"
        
        # Validate model:
        validator = gf.Validator.validate(
                self.model, 
                self.name,
                dataset_args=deepcopy(dataset_arguments),
                efficiency_config=efficiency_config,
                far_config=far_config,
                roc_config=roc_config,
                checkpoint_file_path=validation_file_path,
                heart=heart
            )

        validator.plot(
            model_path / "validation_plots.html"
        )

    def test(self, validation_datasets: tf.data.Dataset, num_batches: int):
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
        self.accuracies.append(population.accuracies.pop(index))
        self.losses.append(population.losses.pop(index))
        self.num_models += 1

    def tick(self):
        self.mean_accuracy_history.append(np.mean(self.accuracies))
        self.mean_loss_history.append(np.mean(self.losses))
        self.mean_fitness_history.append(np.mean(self.fitnesses))
        self.num_models = len(self.models)

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
        initial_population_size: int, 
        max_population_size: int,
        default_genome: gf.ModelGenome,
        training_config: dict,
        dataset_args : dict,
        num_onsource_samples : int = None,
        num_offsource_samples : int = None,
        num_ifos : int = 1,
        population_directory_path : Path = Path("./population/"),
        metrics : List = []
    ):
        if num_onsource_samples is None:
            self.num_onsource_samples = int((gf.Defaults.onsource_duration_seconds + 2.0*gf.Defaults.crop_duration_seconds) * gf.Defaults.sample_rate_hertz)
        
        if num_offsource_samples is None:
            self.num_offsource_samples = int(gf.Defaults.offsource_duration_seconds * gf.Defaults.sample_rate_hertz)

        self.initial_population_size = initial_population_size
        self.current_population_size = initial_population_size
        self.max_population_size = max_population_size
        self.population_directory_path = population_directory_path
        self.metrics = metrics
        self.default_genome = default_genome
        self.training_config = training_config
        self.dataset_args = dataset_args
        self.num_ifos = num_ifos
        self.current_id = 0

        self.orchard = PopulationSector("orchard", population_directory_path)
        self.nursary = PopulationSector("nursary", population_directory_path)
        self.initilize()

    def add_model(self, genome):

        input_configs = [
            {
                "name" : gf.ReturnVariables.ONSOURCE.name,
                "shape" : (self.num_ifos, self.num_onsource_samples,)
            },
            {
                "name" : gf.ReturnVariables.OFFSOURCE.name,
                "shape" : (self.num_ifos, self.num_offsource_samples,)
            }
        ]
        
        output_config = {
            "name" : gf.ReturnVariables.INJECTION_MASKS.name,
            "type" : "binary"
        }

        model_number = self.current_id
        self.current_id += 1

        model_name = f"model_{model_number}"

        model = gf.Model.load(
            name=model_name,
            model_load_path=Path(f"./population/{model_name}"),
            genome=genome,
            num_ifos=self.num_ifos,
            training_config=self.training_config,
            input_configs=input_configs,
            output_config=output_config,
            dataset_args=self.dataset_args
        )

        self.nursary.add(model)
        model.summary()
        
    def initilize(self):
        for j in range(self.initial_population_size): 
            self.random_sapling()
            
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
    
    def train(
        self, 
        num_generations,
        dataset_args,
        num_validate_examples,
        num_examples_per_batch,
        max_num_epochs_per_generation=1
    ):  

        test_args = deepcopy(dataset_args)
        test_args["group"] = "test"
        test_args["seed"] = 1984
        test_dataset = gf.Dataset(**test_args).map(adjust_features)

        for i in range(self.current_population_size):
            logging.info(f"Training model: {i}")
            model = self.nursary.models[i]

            if not model.loaded:
                model.train(
                    validate_dataset=test_dataset, 
                    max_epochs_per_lesson=max_num_epochs_per_generation,
                    force_retrain=True
                )
                self.nursary.fitnesses[i] = 1.0/self.nursary.models[i].metrics[-1].history['val_loss'][-1]

            model.metrics = []
            model.model = None

            self.nursary.save()

        current_dir = Path(__file__).resolve().parent.parent
        initial_processes = [
            gf.Process(f"python train.py", name, tensorflow_memory_mb=4000, cuda_overhead_mb=2000, initial_restart_count=1)
            for name in [f"model_{i}" for i in range(self.current_population_size)]
        ]

        manager = gf.Manager(
            initial_processes,
            max_restarts=20,
            restart_timeout_seconds=3600.0, 
            process_start_wait_seconds=1.0, 
            management_tick_length_seconds=5.0,
            max_num_concurent_processes=8,
            log_directory_path = Path(f"{current_dir}/population/logs/")
        )

        while manager:
            manager()
            manager.tabulate()

        quit()
                
        while 1:     
            i = self.roulette_wheel_selection(self.nursary.fitnesses)
            logging.info(f"Training model: {self.nursary.models[i].name}")

            is_alive = self.nursary.models[i].train(
                validate_dataset=test_dataset, 
                max_epochs_per_lesson=max_num_epochs_per_generation, 
                force_retrain=False
            )   

            if not isinstance(self.nursary.models[i].metrics[-1], dict):
                self.nursary.fitnesses[i] = 1.0/self.nursary.models[i].metrics[-1].history['val_loss'][-1]
                self.nursary.accuracies[i] = self.nursary.models[i].metrics[-1].history['val_binary_accuracy'][-1]
            else:
                self.nursary.fitnesses[i] = 1.0/self.nursary.models[i].metrics[-1]['val_loss'][-1]
                self.nursary.accuracies[i] = self.nursary.models[i].metrics[-1]['val_binary_accuracy'][-1]

            # Birth if conditions are correct:  
            # Flip if random or offspring
            if not is_alive:
                self.orchard.transfer(self.nursary, i)

            plant_condition = self.nursary.num_models < self.max_population_size
            exploration_rate = 0.5
            if plant_condition:
                if np.random.random() < exploration_rate and self.orchard.num_models > 1: 
                    self.germinate_sapling()
                else:
                    self.random_sapling()

                self.nursary.models[-1].train(
                    validate_dataset=test_dataset, 
                    max_epochs_per_lesson=max_num_epochs_per_generation, 
                    force_retrain=True
                )   
            
            if self.nursary.num_models == 0:
                break

            self.nursary.tick()
            self.orchard.tick()

            self.nursary.save()
            self.orchard.save()

            print("Nursary History:", self.nursary.mean_accuracy_history)
            print("Orchard History:", self.orchard.mean_accuracy_history)

        print("Final scores:", self.orchard.fitnesses)
        print("Final Accuracies:", self.orchard.accuracies)

    def germinate_sapling(self):
        
        parent_a = self.orchard.models[self.roulette_wheel_selection(self.orchard.fitnesses)]
        parent_b = self.orchard.models[self.roulette_wheel_selection(self.orchard.fitnesses)]

        #Crossover
        new_genome = parent_a.genome.crossover(parent_b.genome)

        #Mutate
        new_genome.mutate()

        logging.info('Germinated new model.')

        self.add_model(new_genome)

    def random_sapling(self):
        new_genome = deepcopy(self.default_genome)  
        new_genome.randomize()

        logging.info('Randomised new model.')

        self.add_model(new_genome)    


