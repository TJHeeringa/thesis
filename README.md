# Thesis

This is the code used for the thesis "The Effect of Higher Order Activation Functions on Infinitely wide Neural Networks". 

## Project structure

| Folder             | Meaning                                                                         |
|--------------------|---------------------------------------------------------------------------------|
| data               | Folder to store the sampled data points.                                        |
| models             | Folder to store the trained neural networks.                                    |
| results/images     | Folder to store the created images.                                             |
| src/data           | Folder with script to sample the gaussian.                                      |
| src/models         | Folder with neural network archictures and components.                          |
| src/training       | Folder with script for creating, training and storing the used neural networks. |
| src/visualizations | Folder with code to create the images used.                                     |


## How to reproduce the results?

To produce the data:
```
python3 ./src/data/sample_gaussian.py
```

To train the networks on the data:
```
python3 -m src.training.directApproximation
python3 -m src.training.taylorApproximation
```

To create the images:
```
python3 -m src.visualizations.data
python3 -m src.visualizations.upper_bound
python3 -m src.visualizations.network
python3 -m src.visualizations.TrainingLoss.fixed_m__small_radius__overparametrized
python3 -m src.visualizations.TrainingLoss.fixed_m__small_radius__underparametrized
python3 -m src.visualizations.TrainingLoss.fixed_m__large_radius__overparametrized
python3 -m src.visualizations.TrainingLoss.fixed_m__large_radius__underparametrized
python3 -m src.visualizations.BoundValidation.fully_bound__fixed_m
python3 -m src.visualizations.BoundValidation.fully_bound__fixed_n
```

NB: Some of these scripts create folders without proper checking on whether the folder exists, so you should remove those before running the code. 