from dataclasses import dataclass
from typing import Union, List, Dict, Optional
from copy import deepcopy
from pathlib import Path
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras.layers import Lambda
from keras import backend as K
from keras.callbacks import Callback

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


@dataclass
class HyperParameter:
    possible_values: Dict[str, Union[str, List[Union[int, float]]]]
    value: Union[int, float, str] = None
    
    def __post_init__(self):
        self.randomize()

    def randomize(self):
        """
        Randomizes this hyperparameter based on its possible_values.
        """
        value_type = self.possible_values['type']
        possible_values = self.possible_values['values']
        
        if value_type == 'list':
            self.value = np.random.choice(possible_values)
        elif value_type == 'power_2_range':
            power_low, power_high = map(int, np.log2(self.possible_values['values']))
            power = np.random.randint(power_low, power_high + 1)
            self.value = 2**power
        elif value_type == 'int_range':
            low, high = self.possible_values['values']
            self.value = np.random.randint(low, high + 1)
        elif value_type == 'float_range':
            self.value = np.random.uniform(*possible_values)

    def mutate(self, mutation_rate: float):
        """
        Returns a new HyperParameter with a mutated value, based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_param: New HyperParameter instance with potentially mutated value.
        """
        mutated_param = deepcopy(self) 
        if np.random.random() < mutation_rate:
            value_type = self.possible_values['type']
            possible_values = self.possible_values['values']
            if value_type in ['int_range', 'power_2_range', 'float_range']:
                mutation = np.random.normal(scale=(possible_values[1] - possible_values[0]) / 10)
                new_value = self.value + mutation
                # Make sure new value is in the allowed range
                new_value = min(max(new_value, *possible_values))
                mutated_param.value = new_value
            else:
                mutated_param.randomize()
            return mutated_param
        else:
            return deepcopy(self)
        
def hp(N):
    return HyperParameter({'type': 'list', 'values': [N]})

def ensure_hp(parameter):
    return parameter if isinstance(parameter, HyperParameter) else hp(parameter)

@dataclass
class BaseLayer:
    layer_type: str
    activation: Union[HyperParameter, str]
    mutable_attributes: List
    
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
        mutated_layer = deepcopy(self)
        for attribute_name in mutated_layer.mutable_attributes:
            mutated_value = getattr(mutated_layer, attribute_name).mutate(mutation_rate)
            setattr(mutated_layer, attribute_name, mutated_value)
        return mutated_layer

@dataclass
class DenseLayer(BaseLayer):
    units: HyperParameter
    activation: HyperParameter

    def __init__(
            self, 
            units: Union[HyperParameter, int], 
            activation: Union[HyperParameter, str] = "relu"
        ):
        """
        Initializes a DenseLayer instance.
        
        Args:
        ---
        units : Union[HyperParameter, int]
            HyperParameter specifying the number of units in this layer.
        activation : Union[HyperParameter, int]
            HyperParameter specifying the activation function for this layer.
        """
        
        self.layer_type = "Dense"
        self.activation = ensure_hp(activation)
        self.units = ensure_hp(units)
        self.mutable_attributes = [self.activation, self.units]

@dataclass
class ConvLayer(BaseLayer):
    filters: HyperParameter
    kernel_size: HyperParameter
    strides: HyperParameter
    
    def __init__(self, 
        filters: HyperParameter, 
        kernel_size: HyperParameter, 
        activation: HyperParameter, 
        strides: HyperParameter = hp(1)
        ):
        """
        Initializes a ConvLayer instance.
        
        Args:
        filters: HyperParameter specifying the number of filters in this layer.
        kernel_size: HyperParameter specifying the kernel size in this layer.
        activation: HyperParameter specifying the activation function for this layer.
        strides: HyperParameter specifying the stride length for this layer.
        """
        self.layer_type = "Convolutional"
        self.activation = ensure_hp(activation)
        self.filters = ensure_hp(filters)
        self.kernel_size = ensure_hp(kernel_size)
        self.strides = ensure_hp(strides)

        self.padding = hp("same")
        
        self.mutable_attributes = [self.activation, self.filters, self.kernel_size, self.strides]
        
@dataclass
class PoolLayer(BaseLayer):
    pool_size: HyperParameter
    strides: HyperParameter
    
    def __init__(self, 
        pool_size: HyperParameter, 
        strides: Optional[Union[HyperParameter, int]] = None
        ):
        """
        Initializes a PoolingLayer instance.
        
        Args:
        pool_size: HyperParameter specifying the size of the pooling window.
        strides: HyperParameter specifying the stride length for moving the pooling window.
        """
        self.layer_type = "Pooling"
        self.pool_size = ensure_hp(pool_size)
        
        if strides is None:
            self.strides = self.pool_size
        else:
            self.strides = ensure_hp(strides)
        
        self.padding = hp("same")
        self.mutable_attributes = [self.pool_size, self.strides]
        
class DropLayer(BaseLayer):
    rate: HyperParameter
    
    def __init__(self, rate: Union[HyperParameter, float]):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: HyperParameter specifying the dropout rate for this layer.
        """
        self.layer_type = "Dropout"
        self.rate = ensure_hp(rate)
        self.mutable_attributes = [self.rate]

class WhitenLayer(BaseLayer):
    
    def __init__(self):
        """
        Initializes a DropLayer instance.
        
        Args:
        rate: HyperParameter specifying the whitening for this layer.
        """
        self.layer_type = "Whiten"

def randomizeLayer(layer_types: List[str], default_layers: Dict[str, BaseLayer]) -> BaseLayer:
    """
    Returns a randomized layer of a random type.
    
    Args:
    layer_types: List of possible layer types.
    default_layers: Dictionary mapping layer types to default layers of that type.
    
    Returns:
    layer: New BaseLayer instance of a random type, with randomized hyperparameters.
    """
    layer_type = np.random.choice(layer_types)
    layer = default_layers[layer_type]
    layer.randomize()
    return layer

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

class ModelBuilder:
    def __init__(
        self, 
        layers: List[BaseLayer], 
        optimizer: str, 
        loss: str, 
        batch_size: int = None
    ):
        """
        Initializes a ModelBuilder instance.
        
        Args:
        layers: List of BaseLayer instances making up the model.
        optimizer: Optimizer to use when training the model.
        loss: Loss function to use when training the model.
        batch_size: Batch size to use when training the model.
        """   

        if batch_size is None:
            batch_size = gf.Defaults.num_examples_per_batch

        self.layers = layers
        self.batch_size = ensure_hp(batch_size)
        self.optimizer = ensure_hp(optimizer)
        self.loss = ensure_hp(loss)
        
        self.fitness = []
        
        self.metrics = []

    def build_model(
        self, 
        input_configs : Union[List[Dict], Dict],
        output_config : dict,
        model_path : Path = None,
        metrics : list = []
    ):
        """
        Builds the model.
        
        Args:
        input_shape: Shape of the input data.
        output_shape: Shape of the output data.
        """        

        self.model_path = model_path

        if not isinstance(input_configs, list):
            input_configs = [input_configs]
        
        # Create input tensors based on the provided configurations
        inputs = {config["name"]: tf.keras.Input(shape=config["shape"], name=config["name"])
                  for config in input_configs}

        # The last output tensor, starting with the input tensors
        last_output_tensors = list(inputs.values())

        for layer in self.layers:
            new_layer = self.build_hidden_layer(layer)

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

        # Get layer type:
        match layer.layer_type:       
            case "Whiten":
                new_layer = gf.Whiten()
            case "Dense":
                new_layer = tf.keras.layers.Dense(
                        layer.units.value, 
                        activation=layer.activation.value
                    )
            case "Convolutional":
                new_layer = tf.keras.layers.Conv1D(
                        layer.filters.value, 
                        (layer.kernel_size.value,), 
                        strides=(layer.strides.value,), 
                        activation=layer.activation.value,
                        padding = layer.padding.value
                    )
            case "Pooling":
                new_layer = tf.keras.layers.MaxPool1D(
                        (layer.pool_size.value,),
                        strides=(layer.strides.value,),
                        padding = layer.padding.value
                    )
            case "Dropout":
                new_layer = tf.keras.layers.Dropout(
                        layer.rate.value
                    )
            case _:
                raise ValueError(
                    f"Layer type '{layer.layer_type.value}' not recognized"
                )
        
        # Return new layer type:
        return new_layer
            
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
            output_tensor = tf.keras.layers.Dense(
                                1, 
                                activation='sigmoid', 
                                dtype='float32',
                                name=output_config["name"]
                            )(x)
        else:
            raise ValueError(f"Unsupported output type: {output_config['type']}")

        return output_tensor
        
    def train_model(
        self, 
        train_dataset: tf.data.Dataset, 
        validate_dataset: tf.data.Dataset,
        training_config: dict,
        force_retrain : bool = True,
        callbacks = None,
        heart = None
        ):
        """
        Trains the model.
        
        Args:
        train_dataset: Dataset to train on.
        num_epochs: Number of epochs to train for.
        """

        if callbacks is None:
            callbacks = []

        checkpoint_monitor = "val_loss"

        if not force_retrain:
            model_path = self.model_path

            history_data = gf.load_history(self.model_path)
            if history_data != {}:
                best_metric = min(history_data[checkpoint_monitor]) #assuming loss for now
                initial_epoch = len(history_data[checkpoint_monitor]) - 1
            else:
                initial_epoch = 0
                model_path = None
                best_metric = None
        else:
            initial_epoch = 0
            model_path = None
            best_metric = None
            gf.save_dict_to_hdf5({}, self.model_path / "history.hdf5", True)

        early_stopping = gf.EarlyStoppingWithLoad(
                monitor  = checkpoint_monitor,
                patience = training_config["patience"],
                start_from_epoch=4,
                model_path=model_path
            )
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            str(training_config["model_path"]),
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
                epochs = training_config["max_epochs"], 
                initial_epoch = initial_epoch,
                steps_per_epoch = num_batches,
                callbacks = callbacks,
                batch_size = self.batch_size.value,
                verbose=verbose
            )
        )
            
    def validate_model(self, test_dataset: tf.data.Dataset):
        pass

    def test_model(self, validation_datasets: tf.data.Dataset, num_batches: int):
        """
        Tests the model.
        
        Args:
        validation_datasets: Dataset to test on.
        batch_size: Batch size to use when testing.
        """
        
        self.fitness.append(1.0 / self.model.evaluate(validation_datasets, steps=num_batches)[0])
        
        return self.fitness[-1]
    
    def check_death(self, patience):
        return np.all(self.fitness[-int(patience)] > self.fitness[-int(patience)+1:])
        
    def summary(self):
        """
        Prints a summary of the model.
        """
        self.model.summary()
        
    @staticmethod
    def crossover(parent1: 'ModelBuilder', parent2: 'ModelBuilder') -> 'ModelBuilder':
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

        child_model = ModelBuilder(child_layers, parent1.optimizer, parent1.loss, parent1.batch_size)

        return child_model
    
    def mutate(self, mutation_rate: float) -> 'ModelBuilder':
        """
        Returns a new model with mutated layers based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_model: New ModelBuilder instance with potentially mutated layers.
        """
        mutated_layers = [layer.mutate(mutation_rate) for layer in self.layers]
        mutated_model = ModelBuilder(mutated_layers, self.optimizer, self.loss, self.batch_size)

        return mutated_model

class Population:
    def __init__(
        self, 
        initial_population_size: int, 
        max_population_size: int,
        genome_template: int,
        input_size : int,
        output_size : int
    ):
        self.initial_population_size = initial_population_size
        self.current_population_size = initial_population_size
        self.max_population_size = max_population_size
        self.population = \
            self.initilize_population(
                genome_template, 
                input_size, 
                output_size
            )
        self.fitnesses = np.ones(self.current_population_size)
        
    def initilize_population(self, genome_template, input_size, output_size):
    
        population = []
        for j in range(self.initial_population_size):   
            layers = []
            
            genome_template['base']['num_layers'].randomize()
            genome_template['base']['optimizer'].randomize()
            self.batch_size = genome_template['base']['batch_size'].randomize()
            
            for i in range(genome_template['base']['num_layers'].value):
                layers.append(
                    randomizeLayer(*genome_template['layers'][i])
                )
                
            # Create an instance of DenseModelBuilder with num_neurons list
            builder = ModelBuilder(
                layers, 
                optimizer = genome_template['base']['optimizer'], 
                loss = hp(negative_loglikelihood), 
                batch_size = genome_template['base']['batch_size']
            )

            # Build the model with input shape (input_dim,)
            builder.build_model(
                input_configs={}, 
                output_config={}
            )
            population.append(builder)
            builder.summary()
            
        return population
    
    def roulette_wheel_selection(self):
        """
        Performs roulette wheel selection on the population.

        Args:
            population (list): The population of individuals.
            fitnesses (list): The fitness of each individual in the population.

        Returns:
            The selected individual from the population.
        """

        # Convert the fitnesses to probabilities.
        total_fit = sum(self.fitnesses)
        prob = [fit/total_fit for fit in self.fitnesses]

        # Calculate the cumulative probabilities.
        cumulative_probs = np.cumsum(prob)

        # Generate a random number in the range [0, 1).
        r = np.random.rand()

        # Find the index of the individual to select.
        for i in range(len(self.population)):
            if r <= cumulative_probs[i]:
                return i

        # If we've gotten here, just return the last individual in the population.
        # This should only happen due to rounding errors, and should be very rare.
        return self.population[-1]
    
    def train_population(
        self, 
        num_generations, 
        num_train_examples, 
        num_validate_examples, 
        num_examples_per_batch, 
        ds
    ):
                
        num_train_batches = int(num_train_examples // num_examples_per_batch)
        
        num_validate_batches = int(num_validate_examples // num_examples_per_batch)
        
        for i in range(self.current_population_size):
            training_ds = ds.take(num_train_batches)
            validation_ds = ds.take(num_validate_batches)
            
            model = self.population[i]
            model.train_model(training_ds, num_train_batches)
            self.fitnesses[i] = \
                model.test_model(validation_ds, num_validate_batches)
        
        print(self.fitnesses)
        
        for _ in range(self.current_population_size*(num_generations - 1)):
            training_ds = ds.take(num_train_batches)
            validation_ds = ds.take(num_validate_batches)
            
            i = self.roulette_wheel_selection()
            self.population[i].train_model(training_ds, num_train_batches)
            self.fitnesses[i] = \
                model.test_model(validation_ds, num_validate_batches)
            
            print("is_alive:", model.check_death(10))
                        
            print(self.fitnesses)
            print(mean(self.fitnesses))