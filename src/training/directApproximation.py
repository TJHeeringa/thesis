import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
from src.models.HigherOrderReLUNetwork import HigherOrderReLUNetwork
from tqdm import tqdm
from tensorflow import keras

for r in [1,3]:
    print(f" ------------ RADIUS {r} ------------ ")
    for n in tqdm(range(100,1100,100)):
        dataframe = pd.read_csv(f"./data/radius={r}/order=00/samples={n:04}")
        os.makedirs(f"./models/directApproximation/radius={r}/samples={n:04}")
        for s in range(1,11):
            m = 500
            model = HigherOrderReLUNetwork(width=m, power=s)

            model.compile(
                loss=keras.losses.MeanSquaredError(),
                optimizer=keras.optimizers.Adam(),
            )

            history = model.fit(
                x=dataframe[["data"]].values, 
                y=dataframe[["labels"]].values, 
                epochs=500, 
                batch_size=25, 
                verbose=0, 
            )

            history_dataframe = pd.DataFrame(history.history)

            file_name = f"./models/directApproximation/radius={r}/samples={n:04}/order={s:02}__width={m:04}"

            history_dataframe.to_json(file_name + ".json")
            model.save(file_name)

    for m in tqdm(range(100,1100,100)):
        os.makedirs(f"./models/directApproximation/radius={r}/width={m:04}")
        for s in range(1,11):
            n = 500
            dataframe = pd.read_csv(f"./data/radius={r}/order=00/samples={n:04}")

            model = HigherOrderReLUNetwork(width=m, power=s)

            model.compile(
                loss=keras.losses.MeanSquaredError(),
                optimizer=keras.optimizers.Adam(),
            )

            history = model.fit(
                x=dataframe["data"], 
                y=dataframe["labels"], 
                epochs=500, 
                batch_size=25, 
                verbose=0, 
            )

            history_dataframe = pd.DataFrame(history.history)

            file_name = f"./models/directApproximation/radius={r}/width={m:04}/order={s:02}__samples={n:04}"

            history_dataframe.to_json(file_name + ".json")
            model.save(file_name)
