# TMD-AutoML

This repository contains a set of python modules, classes and notebooks that allow the evaluation of the 
travel mode detection algorithm proposed in [US-TransportationMode](https://github.com/vlomonaco/US-TransportationMode) using k-fold cross-validation.

It also allows the evaluation of the improvements in detection performance and cost obtained with the use of the Automated Machine Learning (AutoML)
framework provided by [AutoSklearn](https://automl.github.io/auto-sklearn) and Principal Component Analysis (PCA) class provided by [Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

The evaluation experiments were conducted using smartphone collected data from the [TMD-Dataset](http://cs.unibo.it/projects/us-tm2017/index.html).

# License

The python modules and classes contained in this repository are extended and modified versions of the ones made available in the US-TransportationMode repository.
US-Transportation mode was licensed under MIT License and licensing information can be verified in the [LICENSE.md](./LICENSE.md) file.

# Evaluation Experiments

The evaluation experiments performed using the jupyter notebooks available in this repository have been submitted for publication in a peer-revied journal.
Due to the maximum number of pages that can be included in a journal paper, supplementary tables containing details about the experiments have been made 
made available in this repository:

- [experiments_config.pdf](./experiments_config.pdf): This file details the configurations for AutoSklearn and traditional machine learning algorithms used in each evaluation scenario.

- [ensembles_config.pdf](./ensembles_config.pdf): This file details the ensemble configurations generated with AutoSklearn during each evaluation scenario.

- [experiments_results.pdf](./experiments_results.pdf): This file details the results obtained in each evaluation scenario by the ensembles generated with AutoSklearn and traditional machine learning algorithms.

# Instalation 

To install python modules and notebooks dependencies run:
`pip install -r requirements.txt`

To install AutoSklearn and its dependencies run:
`curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install`
`pip install auto-sklearn`

# Notebooks

- [Data Visualization.ipynb](./Data%20Visualization.ipynb) was used to generate histograms and line plots
- [TravelModeDetection-OriginalDataset.ipynb](./TravelModeDetection-OriginalDataset.ipynb) was used to evaluate the scenarios in which ONLY the original features of the 
dataset were used, with or without PCA.
- [TravelModeDetection-ModifiedDataset.ipynb](./TravelModeDetection-ModifiedDataset.ipynb) was used to evaluate the scenarios in which the Skewness and Kurtosis 
features were used in addition to the ones contained in the original dataset.