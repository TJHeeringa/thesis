import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import matplotlib.pyplot as plt

s = 4
n = 800
m = 500
r = 3

model = keras.models.load_model(f"models/taylorApproximation/radius={r}/samples={n:04}/order={s:02}__width={m:04}")
# model = keras.models.load_model(f"models/directApproximation/radius={r}/samples={n:04}/order={s:02}__width={m:04}")
# model = keras.models.load_model(f"models/directApproximation/radius={r}/width={m:04}/order={s:02}__samples={n:04}")

domain = np.linspace(-r,r,1000)
function = model.predict(domain)
gaussian = np.exp(-0.5 * domain ** 2)

plt.figure()
plt.plot(domain, function)
plt.plot(domain, gaussian)
# plt.savefig(f"results/images/networks/direct__radius_{r}__width_{m:04}__n_{n:04}__s_{s:02}.pdf")
plt.show()