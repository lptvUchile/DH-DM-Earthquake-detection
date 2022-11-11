<img src="http://www.lptv.cl/wp-content/uploads/2017/08/LOGO_2017_740x150.png" width="1020">

# DNN-HMM-Duration-Modelling-Earthquake-Detection

This repository contains the codes of an **end-to-end DNN-HMM based system with duration modelling for robust earthquake detection** proposed in: Nombre_Paper

--------------
## Description

The model proposed is an end-to-end DNN-HMM based scheme, i.e. it does not require previous phase-picking, backed by engineered features and combined with duration modelling of states and seismic events. The proposed engine requires 20 times or so fewer parameters than state-of-the-art methods and therefore needs a smaller training database. Moreover, duration modelling can increase the noise robustness of the detection system significantly, particularly with limited training data.

--------------
## How to install 

pip install -r requirements.txt

or:

conda env create -f env_DNN_HMM.yml

--------------
## Tutorial

The following is a brief example and description of how to use this repository.
The repository is mainly composed of three codes: *features_extraction.py*, *train.py* and *test.py*.

- features_extraction.py: used to obtain the features of the seismic traces, the input of this code corresponds to the path of the database to use (the database should be in the folder *data/name_database/sac*, where name_database is the name of the database to use). The output is a .np file that will be generated in the folder *data/name_database/features*. In case you want to train the model, you will also need the prior probabilities, to get these probabilities you need to use the file *get_prob_prior.npy*, which is in the folder *src/utils/*. 
- train.py: this code is used to train and validate the model. This uses the features generated with the code *features_extraction.py* and the alignment generated by *Main_Algoritmo_ViterbiForzado.py*. The output is the trained DNN, which is saved in the *models* folder, the detection time in each trace and a summary of the resulting metrics can be found in the *Results* folder.
-  test.py: The main aim of this code is to load an already trained model, which is saved in *models* folder and test a new database.

The input to *Main_Algorithm_ViterbiForced.py* are the means, variances and Gaussian weights found in the *models/final.txt* file. This was generated from training a GMM with the toolkit Kaldi, using the traditional recipe. In this case, a monophonema model and a bigrama language model were defined. In addition, a manual segmentation of the training database is incorporated as an initial condition of the GMM training.


--------------

## Links
In this <a href="https://drive.google.com/drive/folders/1wuC61PkiOQijR6jmmMmchectpmvkPFOm?usp=share_link" target="_blank">link</a> you will find the database used