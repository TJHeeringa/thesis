import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import cm
from scipy.fftpack import fft
from scipy.special import gamma, factorial

pi = tf.constant(np.pi, dtype="float64")
two = tf.constant(2, dtype="float64")
three = tf.constant(3, dtype="float64")
four = tf.constant(4, dtype="float64")

""" works only for r\geq 1 """

delta = tf.constant(0.01, dtype="float64")
r = tf.constant(1, dtype="float64")

s_min = 1
s_max = 16
s_steps = s_max
s = np.linspace(s_min, s_max, s_steps)

barron_norm_fs_squared = pi * tf.pow([two]*len(s), s+2) * tf.pow(gamma((s+1)/2),2)
base_m = 12 * tf.pow(r, 2*(s-1)) * tf.pow(tf.pow([1+r]*len(s), s) / factorial(s), 2) * barron_norm_fs_squared

bound_m = base_m * r
bound_d_n = base_m * 16 * s * (1+r) * tf.sqrt(two*tf.math.log(four))
bound_delta_n = base_m * 6 * r * tf.sqrt(two*tf.math.log(2/delta))
 
plt.figure()
plt.semilogy(s, bound_m, c="r", label=r"$\mathcal{E}_m$")
plt.semilogy(s, bound_d_n, c="m", label=r"$\mathcal{E}_{d,n}$")
plt.semilogy(s, bound_delta_n, c="b", label=r"$\mathcal{E}_{\delta,n}$")
plt.axvline(13, c="g", ls="-.")
plt.axhline(bound_m[0], c="r", ls="--")
plt.axhline(bound_d_n[0], c="m", ls="--")
plt.axhline(bound_delta_n[0], c="b", ls="--")
plt.legend()
plt.xlabel("s")
plt.savefig(f"results/images/r_{r}__upper_bound.pdf")
plt.show()