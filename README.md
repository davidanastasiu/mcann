## MC-ANN

We present `MC-ANN`,  a Mixture Clustering-Based Attention Neural Network for time Series Forecasting.


## Preliminaries

Experiments were executed in an Anaconda 3 environment with Python 3.8.8. The following will create an Anaconda environment and install the requisite packages for the project.

```bash
conda create -n MCANN python=3.8.8
conda activate MCANN
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
python -m pip install -r requirements.txt
```

## Files organizations

Download the datasets from [here](https://clp.engr.scu.edu/static/datasets/MCANN-datasets.zip) and upzip the files in the data_provider directory. In the ./data_provider/datasets directory, there should now be  5 reservoir sensor (file names end with _sof24.tsv) datasets.

## Parameters setting

--reservoir_sensor: reservoir dataset file name. The file should be csv file.

--rain_sensor: rain dataset file name. The file should be csv file.

--train_volume: train set size.

--hidden_dim: hidden dim of lstm layers.

--atten_dim: to set hidden dim of attention layers.

--layer: number of layers.

--os_s: oversampling steps.

--os_v: oversampling frequency.

--seq_weight: sequence cluster weight.

--watershed: 1 if trained with rain info, else 0.

--model: model name, used to generate the pt file and predicted file names.

--mode: set it to 'train' or 'inference' with an existing pt_file.

--pt_file: if set, the model will be loaded from this pt file, otherwise check the file according to the assigned parameters.

--save: if save the predicted file of testset, set to 1, else 0.

--outf: default value is './output', the model will be saved in the train folder in this directory.

Refer to the annotations in `run.py` for other parameter settings. Default parameters for reproducing are set in the files (file names start with opt and end with .txt) under './models/'.

## Training and Inferencing


The Jupyter notebook example.ipynb shows how to train a model via command line commands and use specific model functions to perform inference on the Stevens Creek sensor dataset.




