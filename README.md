# How to reproduce results

- __Download the data__:
  Go to the [IARAI website](https://www.iarai.ac.at/traffic4cast/2019-competition/) to download the traffic4cast 2019 competition data, upack it.

- __Train or download models:__

  - If necessary, update the data paths in `model_training/unet_config.py` and `model_training/graphnets_config.py`

  - run `model_training/unet_training.py`, `kipfdepth1_training.py`, `model_training/kipfdepth2_training.py` and `model_training/graphresnet_training.py` from within the main directory (e.g., `python ./model_training/unet_training.py`) to train the models (pretrained models are available [here](https://1drv.ms/u/s!Ar5qZtQfLW_Mi-8qO6_MuuOJc4S5Yw?e=VoTdIE)).

- __Run the generalization experiment__:

  - If necessary, update the data paths in `experiment/generalization_config.py`
  - If you want to test your own models, update `config['model_tuple_list']`
  - run `experiment/generalization.py` to test all models on all cities.

- __Plot__:

  - To reproduce the plot `output/performance_nb_params.pdf`run `experiment/plot_performance_vs_nbparams.py` 

All scripts should be executed in the root folder of the repository. E.g., `python experiment/generalization.py`

# Folder structure

- __Code__:
  - __experiment__

    `generalization.py`: This script runs the main experiment and calculates the loss for all Moscow trained models on Istanbul and Berlin. The results are stored in  `output/data_generalization.p'` as a dictionary.  

  - __model_training__
    All scripts and configuration files necessary for the model training
    
    - `*_training.py` files are used to train the corresponding networks
    - `graphnets_config.py` has all necessary configurations for the training of the different graph networks 
    - `unet_config.py` has all necessary configurations for the training of the different U-Nets
    - All training results are stored in `/runs/graphnets` or `/runs/unets` respectively
    
  - __models__

    - `graph_models.py` and `unet.py` contain the definitions for the different models used in this paper.

  - __utils__
    Helper functions for graph image-transformations and neural network training 
- __Data and results__:
  - __data__
    Default directory for the raw data. Raw data has to be downloaded from the [IARAI website](https://www.iarai.ac.at/traffic4cast/2019-competition/).
    
  - __images__
    Graphs used in the paper
    
  - __output__ 
    
    - `data_generalization.p` pickle file with the results of the generalization experiment
    - `performance_nb_params.pdf` Figure 4 from the `Graph-ResNets for short-term traffic forecasts in almost unknown cities` paper.
    
  - __runs__
    Folder that stores the tensorboard logs and the corresponding trained models. The trained models used in the paper are stored in `PMLR_nets` all newly trained UNets are stored in the `unets` folder and all newly trained graph networks are stored in the `graphnets` folder.
    Pretrained networks have to be downloaded from [here](https://1drv.ms/u/s!Ar5qZtQfLW_Mi-8qO6_MuuOJc4S5Yw?e=VoTdIE)
    



### Notes

- We use [PyTorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) to implement graph neural networks
- For reliable results the batch-size for graph networks must be set to 1