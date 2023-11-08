import numpy as np
import pandas as pd
import time
from scipy.special import exp10
import sys
from .Viterbi_Log_9estados import Viterbi_Log_SIL3
from .Decoding_Palabras_9estados import Decoding_Palabras_SIL3
from .Escribir_ctm import Escribir_ctm
from .matrix_transitions import Prob_Transicion_automatico
from .main_new_metrics import metricas_viterbi
import warnings
warnings.filterwarnings("ignore")



def Algoritmo_Viterbi(ref_file_p, file_viterbi, sac, phones, transitions_file, Probs_Observacion, nombre_conjunto):
    """
    Implements the Viterbi algorithm for speech recognition.

    Args:
    - ref_file_p: Reference file.
    - file_viterbi: Viterbi file.
    - sac: Information from the database.
    - phones: Phoneme information.
    - transitions_file: File containing transition probabilities.
    - Probs_Observacion: Observation probabilities.
    - nombre_conjunto: Name of the dataset.

    Returns: The sequence of states
    """
    print('Analysis of the', nombre_conjunto, 'dataset')
    
    # Start the timer
    start = time.time()

    # Read information from the database file
    Lineas_Nombres_Archivos = list(pd.read_excel(sac)['name'])

    # Read phoneme numbering and names
    Phones = open(phones, "r")
    Phones_lineas = Phones.readlines()[1:-2]
    Phones.close()

    # Define the vocabulary and its properties
    Vocabulario = {
        'Palabra': ['!SIL', 'EVENTO'],
        'N_Estados': [3, 9],
        'N_Fonemas': [[1, 3], [3, 3]]
    }

    # Define initial probabilities (pi)
    Prob_Inicial = [[exp10(-0.0017), 0, 0], [exp10(-2.7185), 0, 0, 0, 0, 0, 0, 0, 0]]
    P_Inicial = [list(np.log(i)) for i in Prob_Inicial]

    # Extract transition probabilities
    topology = open(transitions_file, "r")
    lineas = topology.readlines()
    topology.close()
    P_Transicion, Vocab = Prob_Transicion_automatico(Vocabulario, lineas, Phones_lineas, 'mono')

    # Create and open a file for Viterbi results
    fs = open(file_viterbi + '.ctm', 'w')

    for Indice in range(len(Probs_Observacion)):
        # Extract observation probabilities for the current utterance
        P_Observacion = Probs_Observacion[Indice]

        # Calculate the Viterbi algorithm
        Delta, Psi, S_opt = Viterbi_Log_SIL3(P_Transicion, P_Inicial, P_Observacion)

        # Associate states and words
        Ruido_diff, Evento_diff = Decoding_Palabras_SIL3(S_opt)

        # Write the results to the output file
        Escribir_ctm(Ruido_diff, Evento_diff, Lineas_Nombres_Archivos, Indice, fs)

    fs.close()

    # End the timer and print the time taken
    end = time.time()
    print('Time:', str(end - start))

    # Call the metric evaluation function
    metricas_viterbi(file_viterbi + '.ctm', ref_file_p, nombre_conjunto)
