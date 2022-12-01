import numpy as np
import pandas as pd
import time
from scipy.special import exp10

import sys
sys.path.insert(1, '../utils/')

from Viterbi_Log_9estados import Viterbi_Log_SIL3
from Decoding_Palabras_9estados import Decoding_Palabras_SIL3
from Escribir_ctm import Escribir_ctm
from matrix_transitions import Prob_Transicion_automatico
from main_new_metrics import metricas_viterbi
import warnings
warnings.filterwarnings("ignore")


def Algoritmo_Viterbi(ref_file_p,file_viterbi,sac,phones,transitions_file,Probs_Observacion, nombre_conjunto):
        print('Analisis del conjunto de ', nombre_conjunto)
        start = time.time()


        #Path Resultados
        path_results = 'results'
 

        #Lectura del archivo sac.scp.txt
        Nombres_Archivos = open(sac, "r")
        Lineas_Nombres_Archivos = Nombres_Archivos.readlines()
        Nombres_Archivos.close()


        #Leemos la numeracion y el nombre de los fonemas
        Phones = open(phones, "r")
        Phones_lineas = Phones.readlines()[1:-2]
        Phones.close()

        Vocabulario = {
        'Palabra' : ['!SIL','EVENTO'],
        'N_Estados': [3,9],
        'N_Fonemas': [[1,3],[3,3]]
                }


        # 3. Se define pi
        Prob_Inicial = [[exp10(-0.0017),0,0],[exp10(-2.7185),0,0,0,0,0,0,0,0]]
        P_Inicial = [list(np.log(i)) for i in Prob_Inicial]


        # 4. Se extraen las probabilidades de transición
        topology = open(transitions_file, "r")
        lineas = topology.readlines()
        topology.close()
        P_Transicion,Vocab = Prob_Transicion_automatico(Vocabulario,lineas,Phones_lineas,'mono')
        fs = open(file_viterbi + '.ctm', 'w')

        for Indice in range(len(Probs_Observacion)):

                # 2. Probabilidades de obervacion por utterance
                P_Observacion = Probs_Observacion[Indice]

                # 4.Función que calcula el Algoritmo de Viterbi.
                Delta,Psi,S_opt = Viterbi_Log_SIL3(P_Transicion, P_Inicial, P_Observacion)

                # 5.Funcion que asocie estados y palabras
                Ruido_diff, Evento_diff = Decoding_Palabras_SIL3(S_opt)
                Escribir_ctm(Ruido_diff, Evento_diff,Lineas_Nombres_Archivos, Indice, fs)
        fs.close()


        end = time.time()
        print('Tiempo: '+ str(end - start))

        metricas_viterbi(file_viterbi+ '.ctm', ref_file_p, nombre_conjunto)





