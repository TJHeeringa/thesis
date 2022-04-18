import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.HigherOrderReLULayer import GReLU

def HigherOrderReLUNetwork(width, power):
    model = keras.Sequential([
        GReLU(width, power, name="grelu"),
        layers.Dense(1, use_bias=True, name="output")
    ])
    return model


def generate_shallow_neural_network_grelu(m, power, optimizer):
    model = keras.Sequential([
        GReLU(m, power, name="grelu"),
        layers.Dense(1, use_bias=True, name="output")
    ])
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=optimizer
        # optimizer=keras.optimizers.RMSprop(),
        # optimizer=keras.optimizers.Adam(),
        # optimizer=keras.optimizers.SGD(),
    )
    return model