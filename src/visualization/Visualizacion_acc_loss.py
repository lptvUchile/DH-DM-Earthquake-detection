#coding: utf-8 -*-
"""
Created on Sat Oct 22 11:37:57 2022

@author: Marc
"""

import matplotlib.pyplot as plt
import pandas as pd


path_acc = '/../reports/Acc_DNN.csv'
path_loss = '/../reports/Loss_DNN.csv'

dfAcc = pd.read_csv(path_acc)
dfLoss = pd.read_csv(path_loss)



def plot_loss_acc(Train_loss, Val_loss, Train_acc, Val_acc):
    """
    Function to create a plot of training and validation loss and accuracy.

    Args:
        Train_loss (list): List of training loss values over epochs.
        Val_loss (list): List of validation loss values over epochs.
        Train_acc (list): List of training accuracy values over epochs.
        Val_acc (list): List of validation accuracy values over epochs.

    This function creates a figure with two subplots. The first subplot shows the training and validation loss over epochs,
    and the second subplot shows the training and validation accuracy over epochs.

    Parameters:
        Train_loss (list): List of training loss values over epochs.
        Val_loss (list): List of validation loss values over epochs.
        Train_acc (list): List of training accuracy values over epochs.
        Val_acc (list): List of validation accuracy values over epochs.

    Returns:
        None
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    axes[0].plot(Train_loss, 'C0', label='Train',linewidth=0.8)
    axes[0].plot(Val_loss, 'C1', label='Val',linewidth=0.8)
    axes[0].set_title('Cross Entropy Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()
    
    axes[1].plot(Train_acc, 'C0', label='Train',linewidth=0.8)
    axes[1].plot(Val_acc, 'C1', label='Val',linewidth=0.8)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
        


#plot_loss_acc(dfLoss['Loss_Train'], dfLoss['Loss_Val'], dfAcc['Acc_Train'], dfAcc['Acc_Val'])

