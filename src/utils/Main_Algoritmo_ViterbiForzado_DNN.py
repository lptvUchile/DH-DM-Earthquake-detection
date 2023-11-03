import sys
sys.path.insert(1, '../utils/')
import numpy as np
import pandas as pd
import time
import kaldi_io as kio
from scipy.special import exp10
import matplotlib.pyplot as plt
from Probs_Transicion_ViterbiForzado import Probs_Transicion_Secuencia
from Viterbi_Forzado_restringido_DNN import Viterbi_Forzado_restringido
from Palabras_a_Estados import Palabras_a_Estados
from Sequence_MatrixProb import Sequence_MatrixProb
import IPython
import time



def Algoritmo_Viterbi_Forzado(phones, transitions_file, Probs_Observacion, database, name_database):
    """
    Implements the Viterbi algorithm for forced alignment and saves the results.

    Args:
    - phones: Phoneme information.
    - transitions_file: File containing transition probabilities.
    - Probs_Observacion: Observation probabilities.
    - database: Name of the database.
    - name_database: Name of the current dataset.

    Returns: The sequence of states
    """
    # Define paths to save results
    probs_path_results = '../../data/' + name_database + '/features/Probs_' + name_database + '_DNN'
    Alineaciones_path_results = '../../data/' + name_database + '/features/Alineaciones_ViterbiForzado_' + database + '_DNN.npy'
    Transcripcion = '../../data/' + name_database + '/sac/Transcripcion_' + name_database + '_' + database
    File_raw = '../../data/' + name_database + '/features/Features_' + name_database + '_' + database + '.npy'

    Vocabulario = {
        'Palabra': ['UNK', 'SIL', 'EVENTO'],
        'N_Estados': [3, 3, 9],
        'N_Fonemas': [[1, 3], [1, 3], [3, 3]]
    }

    # Load the transcription file
    Transcripcion = open(Transcripcion, "r")
    text = Transcripcion.readlines()

    # Read the feature matrix
    Matriz_Utt = np.load(File_raw, allow_pickle=True)
    
    Alineaciones = []

    for Indice in range(len(Probs_Observacion)):
        Utt = Matriz_Utt[Indice]
        Palabras = text[Indice].replace('\n', '')
        Transcripcion = Palabras.split(' ')[1:]

        # Represent the transcription in states
        Secuencia = Palabras_a_Estados(Vocabulario, Transcripcion)

        # Extract transition probabilities from the sequence
        P_Transicion = Probs_Transicion_Secuencia(Secuencia)

        # Define initial probabilities (pi)
        Prob_Inicial = np.zeros(len(Secuencia))
        if Secuencia[0] != [0, 0]:
            Prob_Inicial[0] = 1
        P_Inicial = np.log(Prob_Inicial)

        # Extract observation probabilities
        P_Observacion = Probs_Observacion[Indice]

        # Calculate the Viterbi algorithm for forced alignment
        Delta, Psi, S_opt_Indices, S_opt = Viterbi_Forzado_restringido(P_Transicion, P_Inicial, P_Observacion, Utt, Secuencia)
        Alineaciones.append(S_opt)

    # Save the state-frame assignments in a binary matrix
    Sequence_MatrixProb(Alineaciones, probs_path_results)

    # Save the alignments
    np.save(Alineaciones_path_results, Alineaciones)








