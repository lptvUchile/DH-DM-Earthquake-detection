# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 12:14:14 2022

@author: Marc


"""

"""
The goal of this code is to train and validate the DNN.
Also to obtain the results of the model using DNN+HMM, using the training and validation database.
"""
import sys
sys.path.insert(1, '../utils/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import IPython
import pandas as pd
import copy
import random
import time
from Main_Algoritmo_Viterbi import Algoritmo_Viterbi



SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


#Features train and validation
#path_feat_train = '../../data/NorthChile/features/Features_NorthChile_Train.npy'   
#path_feat_val = '../../data/NorthChile/features/Features_NorthChile_Val.npy'
path_feat_train = '' #Features Matrix. Ouput of file Extraction_Features.py
path_feat_val = ''  #Features Matrix. Ouput of file Extraction_Features.py

#labels train and validation
path_label_train = '../../data/NorthChile/features/Probs_NorthChile_Train.npy'
path_label_val = '../../data/NorthChile/features/Probs_NorthChile_Val.npy'

#Prior probability
path_probPrior_train = '../../data/NorthChile/features/Probs_Prior_NorthChile_Train.npy'
path_probPrior_val = '../../data/NorthChile/features/Probs_Prior_NorthChile_Val.npy'

#Earthquakes references
ref_file_train = '../../data/NorthChile/reference/Referencia_NorthChile_Train.xlsx'
ref_file_val = '../../data/NorthChile/reference/Referencia_NorthChile_Val.xlsx'

#path to SAC's
sac_train = "../../data/NorthChile/sac/Sac_NorthChile_Train.scp"
sac_val = "../../data/NorthChile/sac/Sac_NorthChile_Val.scp"



#Features, labels, references, prior prob are loaded
X_train = np.load(path_feat_train, allow_pickle = True)
X_train = np.vstack(X_train) #For DNN training and validation, the temporal order of the features doesn't matter,
                              #that's why are stacked. Notice that it's is only the training of the DNN, not the training of the complete system(DNN+HMM).
X_train = torch.from_numpy(X_train)

y_train_orig = np.load(path_label_train, allow_pickle = True)
y_train = np.argmax(np.vstack(y_train_orig),1)
y_train = torch.from_numpy(y_train)


X_val = np.load(path_feat_val, allow_pickle = True)
X_val = np.vstack(X_val)
X_val = torch.from_numpy(X_val)

y_val_orig = np.load(path_label_val, allow_pickle = True)
y_val = np.argmax(np.vstack(y_val_orig),1)
y_val = torch.from_numpy(y_val)


probPriorTrain  = np.load(path_probPrior_train, allow_pickle=True)
#probPriorVal  = np.load(path_probPrior_val, allow_pickle=True)

BATCH_SIZE = 256
EPOCHS = 100

set_train, set_val = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
train_iterator = DataLoader(set_train, batch_size=BATCH_SIZE, shuffle=True)
val_iterator = DataLoader(set_val, batch_size=BATCH_SIZE, shuffle=True)

INPUT_DIM = 918  #Features
OUTPUT_DIM = 12  #States 


#The DNN is defined
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 16)  
        self.hidden_fc1 = nn.Linear(16,16)
        self.hidden_fc2 = nn.Linear(16,16)
        self.output_fc = nn.Linear(16, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_0 = F.relu(self.input_fc(x))
        h_1 = F.relu(self.hidden_fc1(h_0))
        h_2 = F.relu(self.hidden_fc2(h_1))
        y_pred = self.output_fc(h_2)

        return y_pred,h_2




model = MLP(INPUT_DIM, OUTPUT_DIM)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')



optimizer = optim.Adam(model.parameters(), lr= 0.0001) #Optimizer
criterion = nn.CrossEntropyLoss() #Loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

#Accuracy
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


#Train function
def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for (x, y) in iterator:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x.float())

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


#Validation function
def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x.float())
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



##################################### DNN training  #######################################3
best_valid_loss = float('inf')
Train_loss, Val_loss = np.zeros(EPOCHS), np.zeros(EPOCHS)
Train_acc, Val_acc = np.zeros(EPOCHS), np.zeros(EPOCHS)
for epoch in range(EPOCHS):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, val_iterator, criterion, device)

    if valid_loss < best_valid_loss: 
        best_valid_loss = valid_loss

        torch.save(model.state_dict(), '../../models/model_MLP_HMM.pt') #The DNN is saved

    end_time = time.monotonic()
    
    Train_loss[epoch], Val_loss[epoch] = train_loss, valid_loss
    Train_acc[epoch], Val_acc[epoch] = train_acc, valid_acc
    
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    
model.load_state_dict(torch.load('../../models/model_MLP_HMM.pt')) #The DNN is loaded

#ACC and Loss are saved for different analyses, for example, evaluate overfitting
Acc_DNN = pd.DataFrame(data= {'Acc_Train':Train_acc, 'Acc_Val':Val_acc})
Loss_DNN = pd.DataFrame(data= {'Loss_Train':Train_loss, 'Loss_Val':Val_loss})
Acc_DNN.to_csv('../../reports/Acc_DNN.csv')
Loss_DNN.to_csv('../../reports/Loss_DNN.csv')



def get_predictions(model, iterator, device):

    model.eval()
    feat = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred, _ = model(x.float())
            y_prob = F.softmax(y_pred, dim=-1)
            feat.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    feat = torch.cat(feat, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return feat, labels, probs



def DNN2ProbObs(feat_entrada):
    salida_DNN = []
    for traza in feat_entrada:

        set_conjunto = TensorDataset(torch.from_numpy(traza),-1*torch.ones(len(traza))) #The target is set to -1, because it is not used
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


#The features are reloaded and in this case the temporal order of the features is important.
X_train = np.load(path_feat_train, allow_pickle = True)
X_val = np.load(path_feat_val, allow_pickle = True)

Probs_Observations_train = DNN2ProbObs(X_train)
Probs_Observations_val = DNN2ProbObs(X_val)



##################################### Viterbi Algorithm  #######################################3


file_viterbi_train = 'results/Viterbi_DNN_train'
file_viterbi_val = 'results/Viterbi_DNN_val'
#file_viterbi_train = 'results/Viterbi_DNN_train' #Este es el path que me gustaria que quedaran los ctm, pero no he podido

phones="../../models/phones_3estados.txt"
transitions_file="../../models/final_16_layers3_s1_lr001_NorthChile.mdl"

#train
Algoritmo_Viterbi(ref_file_train, file_viterbi_train, sac_train, phones, transitions_file, Probs_Observations_train, 'train')
#val
Algoritmo_Viterbi(ref_file_val, file_viterbi_val, sac_val, phones, transitions_file, Probs_Observations_val, 'val')
