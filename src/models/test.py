# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 12:14:14 2022

@author: Marc
"""
import sys
sys.path.insert(1, '../utils/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Main_Algoritmo_Viterbi import Algoritmo_Viterbi
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import IPython
import pandas as pd

import copy
import random
import time


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


path_probPrior_train = '../../data/NorthChile/features/Probs_Prior_NorthChile_Train.npy'
#path_probPrior_train = '../../data/NorthChile/features/Prob_Prior_NorthChile_Train.npy'


path_modelo = '../../models/model_MLP_HMM_NorthChile.pt'

ref_file_test = '../../data/NorthChile/reference/Referencia_NorthChile_Test.xlsx'

#path_feat_test = '../../data/NorthChile/features/Features_NorthChile_Test.npy'   
 
path_feat_test = '' # Features Matrix. Ouput of file Extraction_Features.py

sac_test = "../../data/NorthChile/sac/Sac_NorthChile_Test.scp"


    
probPriorTrain  = np.load(path_probPrior_train, allow_pickle=True)  
  
#Se define el modelo
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 16)
        self.hidden_fc1 = nn.Linear(16, 16)
        self.hidden_fc2 = nn.Linear(16, 16)
        self.output_fc = nn.Linear(16, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_0 = F.relu(self.input_fc(x))
        h_1 = F.relu(self.hidden_fc1(h_0))
        h_2 = F.relu(self.hidden_fc2(h_1))
        y_pred = self.output_fc(h_2)

        return y_pred, h_2


INPUT_DIM = 918  #Features con contexto
OUTPUT_DIM = 12  #Estados 

model = MLP(INPUT_DIM, OUTPUT_DIM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model.load_state_dict(torch.load(path_modelo))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')



def get_predictions(model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x.float())

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


    
 
    



def DNN2ProbObs(feat_entrada):
    salida_DNN = []
    for traza in feat_entrada:

        set_conjunto = TensorDataset(torch.from_numpy(traza),-1*torch.ones(len(traza))) #El target lo seteo en -1, da lo mismo
        conjunto_iterator = DataLoader(set_conjunto) 
        images, _ , probs = get_predictions(model, conjunto_iterator, device)
        salida_DNN.append(probs)

    calculo_Prob = []
    for ProbTraza in salida_DNN:
        calculo_Prob.append(np.log(ProbTraza)- np.log(probPriorTrain))

    Probs_Observations = []
    for traza in calculo_Prob:  
        ruido = traza[:,0:3]
        evento = traza[:,3:]
        Probs_Observations.append([np.array(ruido),np.array(evento)])


    return Probs_Observations


X_test = np.load(path_feat_test, allow_pickle = True)
Probs_Observations_test = DNN2ProbObs(X_test)


##################################### Algoritmo Viterbi #######################################3

file_viterbi_test = 'results/Viterbi_DNN_test'


phones="../../models/phones_3estados.txt"
transitions_file="../../models/final_16_layers3_s1_lr001_Iquique.mdl"




#test
Algoritmo_Viterbi(ref_file_test, file_viterbi_test, sac_test, phones, transitions_file, Probs_Observations_test, 'test')

print(f'The model has {count_parameters(model):,} trainable parameters')


