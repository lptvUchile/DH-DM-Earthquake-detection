import numpy as np
import itertools
 

from Coordenadas import Coordenadas
from Backtraking_9estados import Backtraking
from Restricciones_Duracion_Estado import Restricciones_Duracion_Estado
from Probabilidad_ML import Probabilidad_ML
from Guardar_Token import Guardar_Token
from Restricciones_Duracion_Evento_9estados import Restricciones_Duracion_Evento


def Viterbi_Log_SIL3(P_Transicion, P_Inicial, P_Observacion):
    
    N_modelos = np.shape(P_Inicial)[0]
    N_frames = np.shape(P_Observacion[0])[0]
    Delta = []  
    Psi = []
    Token = []
    Token_eventos = []

  
    for i in range(N_frames):
        Delta.append([])
        Psi.append([])
        Token.append([])
        Token_eventos.append([])

        for j in range(N_modelos):
            N_estados = np.shape(P_Inicial[j])[0]
            Delta[i].append([])
            Psi[i].append([])
            Token[i].append([]) 
            Token_eventos[i].append([])


            if i == 0:
                for k in range(N_estados):
                    Pi =P_Inicial[j][k] + P_Observacion[j][0][k]
                    Delta[i][j].append(Pi)
                    Token[i][j].append(1)
                    Token_eventos[i][j].append(1)
                    
            else:                          
                for k in range(N_estados):

                    Probs_Dur_Es = Restricciones_Duracion_Estado(Token[i-1],    
                                                                        12,
                                                                        [j,k])

                                          

                    Probs_Dur_Ev =Restricciones_Duracion_Evento(i,N_frames,k,j,Token_eventos[i-1],12)
                          
                    P_ML= Probabilidad_ML(i,j,k,N_frames,N_estados,12)
                    P_Transicion_final =P_Transicion[j][k] + P_ML  +  Probs_Dur_Ev +Probs_Dur_Es
                    max_arg = np.array(list(itertools.chain(*Delta[i-1][:])))+ P_Transicion_final


                    
                    Coordenada = Coordenadas(Delta[i-1],max_arg)
                    Psi[i-1][j].append(Coordenada)

                    Token,Token_eventos = Guardar_Token(i,j,k,Coordenada,Token,Token_eventos)
                    temp_product = np.max(max_arg) + P_Observacion[j][i][k]
                    Delta[i][j].append(temp_product)


        if i == N_frames-1:
            Delta.append([])
            for j in range(N_modelos):
                N_estados = np.shape(P_Inicial[j])[0]
                Delta[i+1].append([])
                for k in range(N_estados):
                    
                    Probs_Dur_Es = Restricciones_Duracion_Estado(Token[i],12,[j,k])
                    Probs_Dur_Ev = Restricciones_Duracion_Evento(i,N_frames,k,j,Token_eventos[i],12)
                    max_arg = np.array(list(itertools.chain(*Delta[i][:]))) + P_Transicion[j][k]+ P_ML  +  Probs_Dur_Ev +Probs_Dur_Es
      
                    temp_product = np.max(max_arg) 
                    Delta[i+1][j].append(temp_product)
                    Coordenada = Coordenadas(Delta[i],max_arg)
                    Psi[i][j].append(Coordenada)

        


    S_opt = Backtraking(N_frames,Delta,Psi,Delta[N_frames][0][2],Delta[N_frames][1][8])
    
    return Delta,Psi,S_opt

