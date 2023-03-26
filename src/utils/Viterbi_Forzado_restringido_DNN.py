import numpy as np
from Backtraking_ViterbiForzado import Backtraking
from Guardar_Token_ViterbiForzado import Guardar_Token



def Viterbi_Forzado_restringido(P_Transicion, P_Inicial, P_Observacion, Utt, Secuencia):
    # Función que implementa el algoritmo de viterbi
    
    N_frames = Utt.shape[0]
    N_estados = len(Secuencia)
    Delta = np.zeros((N_frames+1,N_estados))  
    Psi = []
    Token = []
    Token_eventos = np.zeros((N_frames,len(Secuencia)))
    Token = np.zeros((N_frames,len(Secuencia)))
  
    for i in range(N_frames):
        Psi.append([])
        
        # Condición inicial
        if i == 0:
            for j in range(N_estados):
                M_E = Secuencia[j]

                Delta[i][j] = P_Inicial[j] + P_Observacion[M_E[0]-1][i][M_E[1]]
                
                Token[i][j] = 1
                  
        elif i > 0:                
            for j in range(N_estados):
                M_E = Secuencia[j]
                P_Transicion_final =P_Transicion[j]                 
                max_arg =Delta[i-1][:]  + P_Transicion_final
                Indice = np.argmax(max_arg)
                Psi[i-1].append(Indice)

                # Se guarda la información de la transición de estados
                Token,Token_eventos = Guardar_Token(i,j,Indice,Token,Token_eventos,M_E,Secuencia)                   
                temp_product = np.max(max_arg) + P_Observacion[M_E[0]-1][i][M_E[1]]
                Delta[i][j] = temp_product

        # Condición final último frame
        if i == N_frames-1:  
            for j in range(N_estados):
                M_E = Secuencia[j]                
                P_Transicion_final =P_Transicion[j] 
                max_arg =Delta[i][:]  + P_Transicion_final   
                temp_product = np.max(max_arg) 
                Delta[i][j] = temp_product
                Indice = np.argmax(max_arg)
                Psi[i].append(Indice)
         
    # Backtracking
    S_opt_Indices,S_opt = Backtraking(N_frames,Secuencia,Psi,Delta,N_estados-1)

    return Delta,Psi,S_opt_Indices,S_opt

