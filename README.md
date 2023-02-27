
# D3GlobalBind

#### Developing based on [EQUIBIND](https://arxiv.org/abs/2202.05146)

Main points:
1. Runable on computer with small size of memory (<64 GB) by using disk cache
2. Allow to test different strategies for global feature learning by changing
model_type in the configuration files at configs_clean/RDKitCoords_flexible_self_docking



# Dataset

Preprocessed data (see dataset section in the paper Appendix) is available from [zenodo](https://zenodo.org/record/6408497). \
The files in `data` contain the names for the time-based data split.

If you want to train one of our models with the data then: 
1. download it from [zenodo](https://zenodo.org/record/6408497) 
2. unzip the directory and place it into `data` such that you have the path `data/PDBBind`



Here are the requirements themselves for the case with a CUDA GPU if you want to install them manually instead of using the `environment.yml`:
````
python=3.7
pytorch 1.10
torchvision
cudatoolkit=10.2
torchaudio
dgl-cuda10.2
rdkit
openbabel
biopython
rdkit
biopandas
pot
dgllife
joblib
pyaml
icecream
matplotlib
tensorboard
````


Download the data and place it in the 'dataset' folder
### Training
To train the model yourself, run:

    python train.py --config=configs_clean/RDKitCoords_flexible_self_docking.yml

The model weights are saved in the `runs` directory.\
You can also start a tensorboard server ``tensorboard --logdir=runs`` and watch the model train. \
To evaluate the model on the test set, change the ``run_dirs:`` entry of the config file `inference_file_for_reproduce.yml` to point to the directory produced in `runs`.
Then you can run``python inference.py --config=configs_clean/inference_file_for_reproduce.yml`` as above!
### Testing
To predict binding structures using the provided model weights run: 

    python inference.py --config=configs_clean/inference_file_for_reproduce.yml
