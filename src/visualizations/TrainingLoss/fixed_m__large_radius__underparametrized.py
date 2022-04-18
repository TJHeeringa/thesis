import matplotlib.pyplot as plt
import pandas as pd

m = 500
n = 800
r = 3

# Direct approx
plt.figure()
for s in range(1,11):   
    loss_dataframe = pd.read_json(f"models/directApproximation/radius={r}/samples={n:04}/order={s:02}__width={m:04}.json", numpy=True)
    loss = loss_dataframe[["loss"]].values
    plt.semilogy(loss, label=f"s={s}")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(f"results/images/directApproximation/fixed_m__large_radius__underparametrized.pdf")

# Taylor approx
plt.figure()
for s in range(1,11):   
    loss_dataframe = pd.read_json(f"models/taylorApproximation/radius={r}/samples={n:04}/order={s:02}__width={m:04}.json", numpy=True)
    loss = loss_dataframe[["loss"]].values
    plt.semilogy(loss, label=f"s={s}")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(f"results/images/taylorApproximation/fixed_m__large_radius__underparametrized.pdf")
plt.show()