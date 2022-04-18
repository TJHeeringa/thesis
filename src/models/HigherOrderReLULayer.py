import tensorflow as tf
from tensorflow.keras import layers

def grelu(x, power):
    """Generalization of ReLU activation function

    Args:
        x ([type]): Input tensor
        power (int): Power to which to raise the ReLU

    Returns:
        [type]: Output tensor
    """
    return tf.math.pow(tf.nn.relu(x), power)


class GReLU(layers.Layer):
    """Dense Layer that uses GReLU as activation function

    Args:
        units (int): Number of neurons in the layer
        power (int): Power of the GReLU
    """
    def __init__(self, units=32, power=1, **kwargs):
        super(GReLU, self).__init__(**kwargs)
        self.units = units
        self.power = power

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True,
                                name="w")
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True,
                                name="b")

    def get_config(self):
        config = super(GReLU, self).get_config()
        config.update({
            "units": self.units,
            "power": self.power
        })
        return config

    def call(self, inputs):
        return grelu(tf.matmul(inputs, self.w) + self.b, self.power)