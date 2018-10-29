Machine Learning, Project 1
===========================

### Files provided

  - implementations.py
    This file contains our main machine learning functions and their helper subfunctions. It is split into parts:
        - Helper functions : miscallenious reusable functions
        - Loss functions : losses used within our various models
        - Gradient functions : gradient computing functions used within our models
        - Model training functions : the models implementations 
        - Cross validation functions : the cross validation functions used to compute   optimal hyperparameters for our models
  - run.py
    This file contains the code that generates results.csv, which contains our best model predictions for the testing dataset.
  - benchmark.ipynb
    This notebook contains our history of training models and testing their accuracy, with increasing levels of feature processing, searching for the best performance. It is loosely chronologiscal, from earlier tries to later ones at the bottom.
  - proj1_helpers.py
    This file contains the original helper methods and new methods made to modularize our code.

### Reusability
...

### How to generate predictions.csv
In order to generate predictions.csv you need serialized weights.npy, clean_features.npy and parameters.npy.
To generate those three files, you can use the ridge regression blockof the benchmark notebook, which is our most peformant model.
Alternatively, you can generate the npy files with another model, in the subsequent blocks of the benchmark notebook. Those models represent our previous less performant tries.
