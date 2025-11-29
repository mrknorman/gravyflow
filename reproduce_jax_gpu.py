import os
# Force JAX to use GPU if available
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import tensorflow as tf
import jax
import jax.numpy as jnp
import gravyflow as gf

print("TensorFlow version:", tf.__version__)
print("JAX version:", jax.__version__)

print("TensorFlow GPUs:", tf.config.list_physical_devices('GPU'))
try:
    print("JAX Devices:", jax.devices())
except Exception as e:
    print("JAX Devices Error:", e)

try:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5000, 5000))
    y = jnp.dot(x, x)
    print("JAX computation successful on:", y.device())
except Exception as e:
    print("JAX computation failed:", e)
