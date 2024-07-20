# File: code/custom_metrics.py
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError

@tf.keras.utils.register_keras_serializable()
class CustomMeanSquaredError(MeanSquaredError):
    def __init__(self, name='mean_squared_error', dtype=None):
        super(CustomMeanSquaredError, self).__init__(name=name, dtype=dtype)
