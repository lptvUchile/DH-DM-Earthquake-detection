import numpy as np
import pandas as pd
import time
from scipy.special import exp10
from Probs_Observacion_GMM import Probs_Observacion
from Probs_Transicion_ViterbiForzado import Probs_Transicion_Secuencia
from Viterbi_Forzado_restringido import Viterbi_Forzado_restringido
from Palabras_a_Estados import Palabras_a_Estados
from Sequence_MatrixProb import Sequence_MatrixProb

# Define database and file paths
database = 'Val'
name_database = 'NorthChile'
probs_path_results = '../../data/' + name_database + '/features/Probs_' + name_database + '_' + database
Alineaciones_path_results = '../../data/' + name_database + '/features/Alineaciones_ViterbiForzado_' + name_database + '_' + database + '.npy' 
File_raw = '../../data/' + name_database + '/features/raw_mfcc_' + database + '.1.ark' # Matriz de caracteristicas. Tiene que ser el archivo .ark ya que no tiene caracteristicas de contexto.

# Define model file path
modelo = "../../models/final_" + name_database + ".txt"

# Transcription file path
Transcripcion = '../../data/' + name_database + '/sac/Transcripcion_' + name_database + '_' + database

# Define vocabulary with states
Vocabulario = {
    'Palabra': ['UNK', 'SIL', 'EVENTO'],
    'N_Estados': [3, 3, 9],
    'N_Fonemas': [[1, 3], [1, 3], [3, 3]]
}
Vocabulario = pd.DataFrame(Vocabulario)

# Read the transcription file
Transcripcion = open(Transcripcion, "r")
text = Transcripcion.readlines()

# Read an utterance
import kaldi_io as kio
utt_base_raw, Matriz_Utt = [], []
for key, mat in kio.read_mat_ark(File_raw):
    utt_base_raw.append(key)
    Matriz_Utt.append(mat)

# Read the model file
Archivo = open(modelo, "r")
Lineas_archivo = Archivo.readlines()
Archivo.close()

Indice = 0
Alineaciones = []

# Iterate through utterances
for Utt in Matriz_Utt:
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
    P_Observacion = Probs_Observacion(Utt, Lineas_archivo, Vocabulario)

    # Calculate the Viterbi algorithm for forced alignment
    Delta, Psi, S_opt_Indices, S_opt = Viterbi_Forzado_restringido(P_Transicion, P_Inicial, P_Observacion, Utt, Secuencia)
    Alineaciones.append(S_opt)
   
    Indice = Indice + 1

# Save the state-frame assignments in a binary matrix
Sequence_MatrixProb(Alineaciones, probs_path_results)

# Save the alignments
np.save(Alineaciones_path_results, Alineaciones)

