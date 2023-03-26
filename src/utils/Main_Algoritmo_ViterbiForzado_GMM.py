import numpy as np
import pandas as pd
import time
from scipy.special import exp10
from Probs_Observacion_GMM import Probs_Observacion
from Probs_Transicion_ViterbiForzado import Probs_Transicion_Secuencia
from Viterbi_Forzado_restringido import Viterbi_Forzado_restringido
from Palabras_a_Estados import Palabras_a_Estados
from Sequence_MatrixProb import Sequence_MatrixProb



start = time.time()

database = 'Val'
name_database = 'NorthChile'
probs_path_results = '../../data/'+name_database+'/features/Probs_'+name_database+'_'+database
Alineaciones_path_results = '../../data/'+name_database+'/features/Alineaciones_ViterbiForzado_'+name_database+'_'+database+'.npy'
File_raw= '../../data/'+name_database+'/features/raw_mfcc_'+database+'.1.ark' # Matriz de caracteristicas. Tiene que ser el archivo .ark ya que no tiene caracteristicas de contexto.

modelo = "../../models/final_"+name_database+".txt"
Transcripcion = '../../data/'+name_database+'/sac/Transcripcion_'+name_database+'_'+database


Vocabulario = {
    'Palabra' : ['UNK','SIL','EVENTO'],
    'N_Estados': [3,3,9],
    'N_Fonemas': [[1,3],[1,3],[3,3]]
            }
Vocabulario = pd.DataFrame(Vocabulario)

#Lectura del archivo text con la transcripcion de palabras
Transcripcion= open(Transcripcion, "r")
text = Transcripcion.readlines()


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

Alineaciones = []
for Utt in Matriz_Utt:
    Palabras = text[Indice].replace('\n','')
    Transcripcion = Palabras.split(' ')[1:]

    #Se representa en estados la transcripción de palabras
    Secuencia = Palabras_a_Estados(Vocabulario,Transcripcion)

    # 1. Funcion que extraiga las probabilidades de transición
    P_Transicion = Probs_Transicion_Secuencia(Secuencia)
   
    # 2. Se define pi
    Prob_Inicial = np.zeros(len(Secuencia))
    if Secuencia[0] != [0,0]:
        Prob_Inicial[0] = 1
    P_Inicial = np.log(Prob_Inicial)

    # 3. Funcion que extraiga las probabilidades de obervacion
    P_Observacion = Probs_Observacion(Utt, Lineas_archivo,Vocabulario)

    # 4.Función que calcule el Algoritmo de Viterbi (está listo y es la función Viterbi_modelo_estado).
    Delta,Psi,S_opt_Indices,S_opt = Viterbi_Forzado_restringido(P_Transicion, P_Inicial, P_Observacion, Utt, Secuencia)
    Alineaciones.append(S_opt)
   
    Indice = Indice + 1

# Se guarda la asinación estado-frame en una matriz binaria
Sequence_MatrixProb(Alineaciones,probs_path_results)

# Se guarda la asinación estado-frame en una matriz binaria
np.save(Alineaciones_path_results,Alineaciones)

end = time.time()
print('Tiempo: '+ str(end - start))





