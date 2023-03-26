import numpy as np
import pandas as pd
import time
import kaldi_io as kio
from scipy.special import exp10
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../utils/')
from Probs_Transicion_ViterbiForzado import Probs_Transicion_Secuencia
from Viterbi_Forzado_restringido_DNN import Viterbi_Forzado_restringido
from Palabras_a_Estados import Palabras_a_Estados
from Sequence_MatrixProb import Sequence_MatrixProb
import IPython

import time
inicio = time.time()

start = time.time()



def Algoritmo_Viterbi_Forzado(phones,transitions_file,Probs_Observacion,database,name_database):

    probs_path_results = '../../data/'+name_database+'/features/Probs_'+name_database+'_DNN'
    Alineaciones_path_results = '../../data/'+name_database+'/features/Alineaciones_ViterbiForzado_'+database+'_DNN.npy'
    Transcripcion = '../../data/'+name_database+'/sac/Transcripcion_'+name_database+'_'+database
    File_raw = '../../data/'+name_database+'/features/Features_'+name_database+'_'+database+'.1.ark'
    Vocabulario = {
        'Palabra' : ['UNK','SIL','EVENTO'],
        'N_Estados': [3,3,9],
        'N_Fonemas': [[1,3],[1,3],[3,3]]
                }
    Vocabulario = pd.DataFrame(Vocabulario)

    #Lectura del archivo text con la transcripcion de palabras
    Transcripcion= open(Transcripcion, "r")
    text = Transcripcion.readlines()

    # Lectura de la matriz de features
    Matriz_Utt = np.load(File_raw, allow_pickle=True)
    

    Alineaciones = []
    for Indice in range(len(Probs_Observacion)):
        Utt= Matriz_Utt[Indice]
        Palabras = text[Indice].replace('\n','')
        Transcripcion = Palabras.split(' ')[1:]

        # Se representa en estados la transcripción de palabras
        Secuencia = Palabras_a_Estados(Vocabulario,Transcripcion)

        # 1. Funcion que extraiga las probabilidades de transición
        P_Transicion = Probs_Transicion_Secuencia(Secuencia)

        # 2. Se define pi
        Prob_Inicial = np.zeros(len(Secuencia))
        if Secuencia[0] != [0,0]:
            Prob_Inicial[0] = 1
        P_Inicial = np.log(Prob_Inicial)

        # 3. Funcion que extraiga las probabilidades de obervacion
        P_Observacion = Probs_Observacion[Indice]

        # 4.Función que calcule el Algoritmo de Viterbi (está listo y es la función Viterbi_modelo_estado).
        Delta,Psi,S_opt_Indices,S_opt = Viterbi_Forzado_restringido(P_Transicion, P_Inicial, P_Observacion, Utt, Secuencia)
        Alineaciones.append(S_opt)
    
    # Se guarda la asinación estado-frame en una matriz binaria
    Sequence_MatrixProb(Alineaciones,probs_path_results)

    # Se guardan las alineaciones
    np.save(Alineaciones_path_results,Alineaciones)


end = time.time()
print('Tiempo: '+ str(end - start))






