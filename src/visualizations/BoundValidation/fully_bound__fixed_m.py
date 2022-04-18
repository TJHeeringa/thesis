import numpy as np
import os
import numpy as np
from numpy.core.fromnumeric import ptp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.integrate import quad
from tqdm import tqdm

from src.data.sample_gaussian import taylored_gaussian 


def integrand(x, model, order):
    return tf.pow( taylored_gaussian(x, order) - tf.reshape(model.predict([x]), [-1]), 2 )


m = 500

for r in [1,3]:
    # domain = np.linspace(-r,r,10000)
    # delta_x = domain[2]-domain[1]

    num_losses = []
    plt.figure()
    for s in tqdm(range(1, 11)):
        losses = []
        for n in range(100,1100,100):
            model = keras.models.load_model(f"models/directApproximation/radius={r}/samples={n:04}/order={s:02}__width={m:04}")
            L2_loss, numerical_error = quad(integrand, -r, r, args=(model,0), epsabs=1e-6)
            losses.append(L2_loss)
            num_losses.append(numerical_error)
            # prediction = tf.reshape(model.predict(domain), [-1])
            # gaussian_prediction = taylored_gaussian(domain, 0)
            # integral = np.sum(((gaussian_prediction - prediction) ** 2)* delta_x)
            # losses.append(integral)
        plt.semilogy(range(100,1100,100), losses, label=f"s={s}")
    plt.legend()
    plt.xlabel("n")
    plt.savefig(f"results/images/directApproximation/fully_bound__fixed_m__r_{r}.pdf")

    plt.figure()
    for s in tqdm(range(1, 11)):
        losses = []
        for n in range(100,1100,100):
            model = keras.models.load_model(f"models/taylorApproximation/radius={r}/samples={n:04}/order={s:02}__width={m:04}")
            if r == 1:
                L2_loss, numerical_error = quad(integrand, -r, r, args=(model,s))
            else:
                L2_loss, numerical_error = quad(integrand, -r, r, args=(model,s), epsabs=1e-6)
            losses.append(L2_loss)
            num_losses.append(numerical_error)
            # prediction = tf.reshape(model.predict(domain), [-1])
            # gaussian_prediction = taylored_gaussian(domain, s)
            # integral = np.sum(((gaussian_prediction - prediction) ** 2)* delta_x)
            # losses.append(integral)
        plt.semilogy(range(100,1100,100), losses, label=f"s={s}")
    plt.legend()
    plt.xlabel("n")
    plt.savefig(f"results/images/taylorApproximation/fully_bound__fixed_m__r_{r}.pdf")

    print(num_losses)
plt.show()

