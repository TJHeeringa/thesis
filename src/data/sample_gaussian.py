from math import factorial
import numpy as np
from scipy.stats import multivariate_normal, uniform
from scipy.special import factorial
import tensorflow as tf
import os
import pandas as pd

def taylored_gaussian(x, order):
    taylor_series = 0
    for k in range(int(np.floor(order/2))):
        taylor_series += (((-0.5)**k ) / factorial(k)) * x ** (2*k)
    return np.exp(-0.5 * x ** 2) - taylor_series


# def generate_data(num_samples, residual_order, radius):
#     """Function

#     Args:
#        num_samples (int): number of samples to take
#        residual_order (int): number of Taylor terms that are subtracted
#        radius (int): interval to take samples from, interval is given by [-radius, radius]

#     Returns:
#         training data (tuple): Tuple of x positions and the residual of the Gaussian at those points.
#     """
#     data = uniform(loc=-radius, scale=2*radius).rvs(num_samples)
#     taylor_series = 0
#     for k in range(int(np.floor(residual_order/2))):
#         taylor_series += (((-0.5)**k ) / factorial(k)) * data ** (2*k)
#     labels = np.exp(-0.5 * data ** 2) - taylor_series
#     return (data, labels)


if __name__ == "__main__":
    for r in [1,3]:
        os.makedirs(f"./data/radius={r}")
        for s in range(11):
            os.makedirs(f"./data/radius={r}/order={s:02}")
            for n in range(100,1100,100):
                data = uniform(loc=-r, scale=2*r).rvs(n)
                labels = taylored_gaussian(data, s)
                dataframe = pd.DataFrame({"data": data, "labels": labels})
                dataframe.to_csv(f"./data/radius={r}/order={s:02}/samples={n:04}")

