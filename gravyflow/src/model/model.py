"""
``gf.Model``: a network built from layer specifications, its training run and its
run record, and the layer specifications themselves.
"""
from dataclasses import dataclass
from typing import Union, List, Dict, Optional
from copy import deepcopy
from pathlib import Path
import os
import logging
import json

logger = logging.getLogger(__name__)

import numpy as np
import keras
from keras import ops
from keras import layers
from keras import losses
from keras.layers import Layer
import jax.numpy as jnp

import gravyflow as gf
from gravyflow.src.model.hyperparameter import search_stream
from gravyflow.src.rng import keras_draws


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

@keras.saving.register_keras_serializable(package="gravyflow")
def folded_normal_nll(y_true, y_pred):
    loc, scale = ops.split(y_pred, 2, axis=-1)
    return -folded_normal_log_prob(y_true, loc, scale)


@keras.saving.register_keras_serializable(package="gravyflow")
class IndependentFoldedNormal(layers.Layer):
    """An independent folded normal Keras layer that outputs parameters."""

    def __init__(self, event_shape=(), **kwargs):
        super().__init__(**kwargs)
        self.event_shape = event_shape

    def call(self, inputs):
        loc, scale = ops.split(inputs, 2, axis=-1)
        loc = ops.softplus(loc + 1.0e-6)
        scale = ops.softplus(scale + 1.0e-6)
        return ops.concatenate([loc, scale], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"event_shape": self.event_shape})
        return config


@dataclass
class BaseLayer:
    """
    A layer specification: its type and its settings, each a
    ``gf.HyperParameter`` (``mutable_attributes`` lists those a draw may vary).
    """
    layer_type: str = "Base"
    seed : int = None
    activation: Union[gf.HyperParameter, str] = None
    mutable_attributes: List = None

    def reseed(self, seed):

        rng = search_stream(seed, "layer")
        for attribute in self.mutable_attributes:
            attribute.reseed(rng.integers(1E10))

    def randomize(self):
        """
        Randomizes all mutable attributes of this layer.
        """
        for attribute in self.mutable_attributes:
            attribute.randomize()


@keras.saving.register_keras_serializable(package="gravyflow")
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

        self.rng = search_stream(self.seed, "layer")

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
        self.rng = search_stream(self.seed, "layer")
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
        self.rng = search_stream(self.seed, "layer")
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
        self.rng = search_stream(self.seed, "layer")
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
        self.rng = search_stream(self.seed, "layer")
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
        self.rng = search_stream(self.seed, "layer")
        self.layer_type = "BatchNorm"

class WhitenLayer(BaseLayer):
    
    def __init__(self):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: gf.HyperParameter specifying the whitening for this layer.
        """
        self.rng = search_stream(self.seed, "layer")
        self.layer_type = "Whiten"
        self.mutable_attributes = []

class WhitenPassLayer(BaseLayer):
    
    def __init__(self):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: gf.HyperParameter specifying the whitening for this layer.
        """
        self.rng = search_stream(self.seed, "layer")
        self.layer_type = "WhitenPass"
        self.mutable_attributes = []

from gravyflow.src.utils.numerics import ensure_even

def epoch_sized(dataset_args: dict, training_config: dict = None) -> dict:
    """
    A copy of a training dataset's arguments, its length one epoch of the
    training config: ``num_examples_per_epoc // num_examples_per_batch``
    batches. The dataset's length is the one definition of an epoch: Model.train
    lets it set fit's steps, so another explicit ``steps_per_epoch`` is refused.
    """
    args = deepcopy(dataset_args)
    if not training_config or "num_examples_per_epoc" not in training_config:
        return args
    batch = args.get("num_examples_per_batch") or gf.Defaults.num_examples_per_batch
    steps = training_config["num_examples_per_epoc"] // batch
    given = args.setdefault("steps_per_epoch", steps)
    if given != steps:
        raise ValueError(
            f"The training dataset's steps_per_epoch={given} differs from the training config's "
            f"epoch of {steps} batches (num_examples_per_epoc="
            f"{training_config['num_examples_per_epoc']} // num_examples_per_batch={batch}). "
            "Leave steps_per_epoch unset: the training config sets an epoch's length."
        )
    return args


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
            self.training_dataset = gf.Dataset(**epoch_sized(dataset_args, training_config))

        if seed is None:
            seed = gf.Defaults.seed

        self.name = name
        self.path = model_path
        self.seed = seed
        self.rng = search_stream(seed, "model")

        if batch_size is None:
            batch_size = gf.Defaults.num_examples_per_batch

        if optimizer is None:
            optimizer = "adam"

        if loss is None:
            if output_config["type"] == "normal":
                loss = folded_normal_nll
            else:
                # The binary output is a logit (see build_output_layer).
                loss = losses.BinaryCrossentropy(from_logits=True)
        
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
        
        self.metrics = []

        self.build(
            input_configs,
            output_config,
            model_path,
            metrics
        )

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
            seed : int = None
        ):
        """
        The model of a blueprint (``hidden_layers``, else ``model_config_path``)
        with the checkpoint in ``model_path`` (``model_load_path`` by default)
        loaded into it (``restore``); with no blueprint, only a checkpoint.
        """
        
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
        
        if model.restore(force_overwrite):
            return model
        # If the model doesn't exist, build a new one
        if blueprint_exists:
            logger.info("No saved model found. Using new model...")
            return model
        raise ValueError("No default model blueprint exists!")

    def restore(self, force_overwrite : bool = False) -> bool:
        """
        Load the checkpoint in the model's directory into this model, unless
        ``force_overwrite``: True when one was loaded. A checkpoint that exists
        is loaded or refused, never replaced by an untrained model.
        """
        checkpoint = gf.model_file(self.model_path)
        if force_overwrite or not os.path.exists(checkpoint):
            return False
        logger.info(f"Loading model from {self.model_path}")
        try:
            self.model = self._load_checkpoint()
        except Exception as error:
            raise ValueError(
                f"The checkpoint {checkpoint} exists but cannot be loaded. Pass "
                "force_overwrite=True to start an untrained model in its place."
            ) from error
        self.loaded = True
        return True

    @property
    def build_seed(self) -> int:
        """The seed Keras builds (and loads) the model's layers from: a stream of the model's seed."""
        return int(search_stream(self.seed, "model/build").integers(2**63))

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

        # Keras seeds the layers built here (initialisers, dropout) from Python's
        # global random: they draw from the model's seed instead.
        with keras_draws(self.build_seed):
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

            # Named, as the dataset's targets are: Keras 3 matches a dict of targets
            # only to a dict of outputs.
            self.model = keras.Model(inputs=inputs, outputs={output_config["name"]: output_tensor})

        # If metrics is empty use best guess
        if not metrics:
            match output_config["type"]:
                case "normal":
                    metrics = [keras.metrics.RootMeanSquaredError()]
                case "binary":
                    # The output is a logit: positive means signal.
                    metrics = [keras.metrics.BinaryAccuracy(threshold=0.0)]
        
        # Keyed by the output's name, as its target is.
        self.model.compile(
            optimizer = self.optimizer.value,
            loss = {output_config["name"]: self.loss.value},
            metrics = {output_config["name"]: metrics}
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
            # A logit, never a probability: the validator ranks logits, because
            # saturated probabilities tie (sigmoid of the output is P(signal)).
            x = layers.Flatten()(x)
            output_tensor = layers.Dense(
                                1, 
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
        callbacks = None
    ):
        """
        Trains the model.

        The split the model trains and is selected on is its data's
        (``split_record``), and belongs to its weights (``run_record``): weights
        that carry a record keep to its split, and only weights the model
        initialised itself start a new one.

        Args:
        training_dataset: Dataset to train on.
        force_retrain: True starts the run in ``model_path`` again (its history
            is cleared); False resumes it. Either way training continues from
            the weights in memory: neither reinitialises them, so neither lets
            weights that carry a record train on another split. Train such data
            from new weights: a new model, or ``Model.load`` with
            ``force_overwrite=True``.

        Raises:
            StaleCheckpointError: the weights, or the run resumed, record
                another split, or none; or, resuming, ``model_path`` holds a
                checkpoint with no history.
        """

        if validation_dataset is None and validate_args is None:
            raise ValueError("No validation dataset!")
        
        if validation_dataset is not None and not validate_args:
            raise ValueError("Validation argument and validateion datasets")

        if validate_args is not None:
            # The validation dataset is the validate split; arguments naming another refuse.
            validate_args = {**validate_args}
            group = validate_args.pop("group", "validate")
            if group != "validate":
                raise ValueError(
                    f"Model.train validates on the 'validate' split; validate_args names {group!r}."
                )
            validation_dataset = gf.Dataset(**deepcopy(validate_args), group="validate")

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

        from gravyflow.src.rng import platform
        from gravyflow.src.validate.config import require_same_split, split_record, with_run_record
        from gravyflow.src.validate.provenance import StaleCheckpointError

        platform.warn_if_not_pure()
        gf.ensure_directory_exists(self.model_path)
        split = split_record(training_dataset, validation_dataset)
        weights = self.run_record
        if weights is not None:
            require_same_split(weights.get("split"), split, f"Model {self.name!r}'s weights")
        saved = self._saved_run_record()

        checkpoint_monitor = "val_loss"
        if not force_retrain:
            model_path = self.model_path

            history_data = gf.load_history(self.model_path)
            if history_data != {}:
                # Resuming continues the run saved here, whose best weights it may restore.
                require_same_split((saved or {}).get("split"), split, self.model_path)
                best_metric = min(history_data[checkpoint_monitor]) #assuming loss for now
                best_epoch = np.argmin(history_data[checkpoint_monitor])
                initial_epoch = len(history_data[checkpoint_monitor])

                if initial_epoch - best_epoch > training_config["patience"]:
                    logger.info(
                        f"Model already completed training. Skipping! Current epoch {initial_epoch}, best epoch {best_epoch}."
                    )
                    self.model = self._load_checkpoint()
                    self.metrics.append(history_data)
                    
                    return False
            elif gf.model_file(self.model_path).exists():
                # A checkpoint with no history cannot be resumed, and a new run
                # would discard it: only force_retrain does that.
                raise StaleCheckpointError(
                    f"{gf.model_file(self.model_path)} has no history to resume "
                    f"({self.model_path / 'history.hdf5'} is empty). Train with "
                    "force_retrain=True to start a new run there, or use another model_path."
                )
            else:
                initial_epoch = 0
                model_path = None
                best_metric = None
        else:
            initial_epoch = 0
            model_path = None
            best_metric = None
            gf.save_dict_to_hdf5({}, self.model_path / "history.hdf5", True)

        record = {"split": split}
        if saved != record:
            # A new run starts here: an earlier run's checkpoint, of another split or
            # of none, goes first, so run.json always describes the weights beside it.
            gf.model_file(self.model_path).unlink(missing_ok=True)
        self._write_run_record(record)
        with_run_record(self.model, record)

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
            gf.model_file(self.model_path),
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
        
        # A gravyflow dataset takes Keras' epoch; without it each Keras epoch
        # would train the next epoch's batches (see epoch_callback).
        if hasattr(training_dataset, "epoch_callback"):
            callbacks += [training_dataset.epoch_callback()]

        # A dataset's length is its epoch (see epoch_sized); only a dataset
        # without one is given fit's steps from the training config.
        if hasattr(training_dataset, "__len__"):
            epoch_steps = {}
        else:
            epoch_steps = {
                "steps_per_epoch": training_config["num_examples_per_epoc"] // self.batch_size.value
            }
        num_validation_batches = training_config["num_validation_examples"] // 32
        
        verbose : int = 1
        if gf.is_redirected():
            verbose : int = 2
        else:
            logger.info("Training... or at least trying to.")

        self.metrics.append(
            self.model.fit(
                training_dataset,
                validation_data=validation_dataset,
                validation_steps=num_validation_batches,
                epochs=current_max_epoch, 
                initial_epoch = initial_epoch,
                callbacks = callbacks,
                batch_size = self.batch_size.value,
                verbose=verbose,
                # The dataset owns the order: its batches are addressed by
                # (epoch, index) from its seed, and Keras' shuffle would draw
                # from Python's unseeded global random.
                shuffle=False,
                **epoch_steps
            )
        )

        gf.save_dict_to_hdf5(
            self.metrics[0].history, 
            self.model_path / "metrics", 
            force_overwrite=False
        )

        return True

    @property
    def run_record(self) -> Optional[dict]:
        """
        The training record of the model's weights (``run_record_of``): its
        ``split`` is the split record of the data they trained and were selected
        on (``split_record``). None for weights the model initialised and never
        trained; weights loaded from a checkpoint carry the record saved beside
        it (``run.json``, as a run recipe's run directory keeps), or an empty one.
        """
        from gravyflow.src.validate.config import run_record_of

        return run_record_of(self.model)

    def _saved_run_record(self) -> Optional[dict]:
        """The training record saved in the model's directory (``run.json``), or None."""
        path = Path(self.model_path) / "run.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _load_checkpoint(self) -> keras.Model:
        """
        The checkpoint in the model's directory, carrying the record saved beside
        it: an empty one when none was, so loaded weights never pass for weights
        the model initialised, which may start any split's record.
        """
        from gravyflow.src.validate.config import with_run_record

        with keras_draws(self.build_seed):  # a load draws its layers' seeds as a build does
            loaded = keras.models.load_model(gf.model_file(self.model_path))
        return with_run_record(loaded, self._saved_run_record() or {})

    def _write_run_record(self, record: dict) -> None:
        path = Path(self.model_path) / "run.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
        os.replace(tmp, path)

    def validate(
        self,
        validate_args : dict,
        config : "gf.ValidationConfig" = None,
        model_path : Path = None,
        heart = None
    ) -> "gf.Validator":
        """
        Score the model with the validator (its logit output is the score),
        saving ``validation_data.h5`` and ``validation_plots.html`` in
        ``model_path`` (the model's own directory by default). The validator
        reads the record the model's weights carry (``run_record``), so it
        refuses to score the split that selected them or data split by another
        spec. ``heart``, any object with a ``beat()`` method, is beaten as the
        validator works, for a supervisor that watches the process.
        """
        model_path = Path(model_path if model_path is not None else self.model_path)
        validation_file_path : Path = model_path / "validation_data.h5"

        # Validate model:
        validator = gf.Validator.validate(
                self.model, 
                self.name,
                dataset_args=deepcopy(validate_args),
                config=config,
                checkpoint_file_path=validation_file_path,
                heart=heart,
            )

        validator.plot(
            model_path / "validation_plots.html"
        )
        return validator

    def summary(self):
        """
        Prints a summary of the model.
        """
        self.model.summary()
