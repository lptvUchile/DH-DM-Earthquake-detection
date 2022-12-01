import numpy as np
import pandas as pd
import time
from scipy.special import exp10
import matplotlib.pyplot as plt

from Probs_Observacion_GMM import Probs_Observacion
from Probs_Transicion_ViterbiForzado import Probs_Transicion_Secuencia
from Viterbi_Forzado_restringido import Viterbi_Forzado_restringido
from Palabras_a_Estados import Palabras_a_Estados
from Escribir_ctm import Escribir_ctm
from Decoding_Palabras_ViterbiForzado import Decoding_Palabras_1fonema
from Sequence_MatrixProb import Sequence_MatrixProb



start = time.time()

probs_path_results = '../../data/NorthChile/features/Probs_NorthChile_Val'
Alineaciones_path_results = '../../data/NorthChile/features/Alineaciones_ViterbiForzado_NorthChile_Val.npy'
lista_archivos = '../../data/NorthChile/sac/Sac_NorthChile_Val.scp'
#File_raw = '../../data/NorthChile/features/raw_mfcc_train_NorthChile_DEV.1.ark'
File_raw= "" # Features Matrix. Ouput of file Extraction_Features.py
modelo = "../../models/final_NorthChile.txt"
Transcripcion = '../../data/NorthChile/sac/Transcripcion_NorthChile_Val'


Vocabulario = {
    'Palabra' : ['UNK','SIL','EVENTO'],
    'N_Estados': [3,3,9],
    'N_Fonemas': [[1,3],[1,3],[3,3]]
            }
Vocabulario = pd.DataFrame(Vocabulario)

#Lectura del archivo text con la transcripcion de palabras
Transcripcion= open(Transcripcion, "r")
text = Transcripcion.readlines()



#Lectura del archivo sac.scp.txt
Nombres_Archivos = open(lista_archivos , "r")
Lineas_Nombres_Archivos = Nombres_Archivos.readlines()
Nombres_Archivos.close()

path_results = ''

#Lectura de una utterance.
import kaldi_io as kio
utt_base_raw,Matriz_Utt = [],[]
for key,mat in kio.read_mat_ark(File_raw):
        utt_base_raw.append(key)
        Matriz_Utt.append(mat)


#Lectura del archivo final.txt
Archivo = open(modelo, "r")
Lineas_archivo = Archivo.readlines()
Archivo.close()


Indice = 0
#Utt= Matriz_Utt[Indice]

Alineaciones = []
for Utt in Matriz_Utt:
    Palabras = text[Indice].replace('\n','')
    Transcripcion = Palabras.split(' ')[1:]

        #Se representa en estados la transcripción de palabras
    Secuencia = Palabras_a_Estados(Vocabulario,Transcripcion)

        # 1. Funcion que extraiga las probabilidades de transición
    P_Transicion = Probs_Transicion_Secuencia(Secuencia)
   
        # 3. Se define pi
    Prob_Inicial = np.zeros(len(Secuencia))
    if Secuencia[0] != [0,0]:
        Prob_Inicial[0] = 1
    P_Inicial = np.log(Prob_Inicial)


            # 2. Funcion que extraiga las probabilidades de obervacion
    P_Observacion = Probs_Observacion(Utt, Lineas_archivo,Vocabulario)

            # 4.Función que calcule el Algoritmo de Viterbi (está listo y es la función Viterbi_modelo_estado).
    Delta,Psi,S_opt_Indices,S_opt = Viterbi_Forzado_restringido(P_Transicion, P_Inicial, P_Observacion, Utt, Secuencia)
    Alineaciones.append(S_opt)

   
    #5.Funcion que asocie estados y palabras
    Ruido_diff, Evento_diff = Decoding_Palabras_1fonema(S_opt)

    #fn = open('Viterbi_Forzado.ctm', 'a')
    #Escribir_ctm(Ruido_diff, Evento_diff,Lineas_Nombres_Archivos, Indice, fn, path_results)
    Indice = Indice + 1


Sequence_MatrixProb(Alineaciones,probs_path_results)

np.save(Alineaciones_path_results,Alineaciones)





end = time.time()
print('Tiempo: '+ str(end - start))





