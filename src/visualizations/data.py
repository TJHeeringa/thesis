import os
import matplotlib.pyplot as plt
import pandas as pd

r = 1
n = 200
for s in range(11):
    dataframe = pd.read_csv(f"data/radius={r}/order={s:02}/samples={n:04}")
    x = dataframe[["data"]].values
    f = dataframe[["labels"]].values
    plt.figure()
    plt.plot(x,f, "bo")
    plt.savefig(f"results/images/data/radius={r}__order={s:02}__samples={n:04}.pdf")