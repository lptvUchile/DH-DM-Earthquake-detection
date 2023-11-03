import numpy as np
from Backtraking_ViterbiForzado import Backtracking
from Guardar_Token_ViterbiForzado import Guardar_Token



def Viterbi_Forzado_restringido(P_Transicion, P_Inicial, P_Observacion, Utt, Secuencia):
    """
    Function that implements the Viterbi algorithm for forced restricted alignment.

    Args:
        P_Transition (list of lists): Transition probabilities between states.
        P_Initial (list): Initial state probabilities.
        P_Observation (list of lists): Observation probabilities.
        Utt (numpy.ndarray): Input observation sequence.
        Sequence (list): Sequence of states.

    Returns:
        Delta (numpy.ndarray): Matrix of accumulated probabilities.
        Psi (list of lists): List of backtracking pointers.
        S_opt_Indices (list): List of state indices in the optimal path.
        S_opt (list): List of state sequences in the optimal path.
    """
    
    N_frames = Utt.shape[0]
    N_estados = len(Secuencia)
    Delta = np.zeros((N_frames+1,N_estados))  
    Psi = []
    Token = []
    Token_eventos = np.zeros((N_frames,len(Secuencia)))
    Token = np.zeros((N_frames,len(Secuencia)))

    for i in range(N_frames):
        Psi.append([])

        # Initial condition
        if i == 0:
            for j in range(N_estados):
                M_E = Secuencia[j]

                Delta[i][j] = P_Inicial[j] + P_Observacion[i][M_E[0]][M_E[1]]
                
                Token[i][j] = 1
                  
        elif i > 0:                  
            for j in range(N_estados):
        
                M_E = Secuencia[j]
                P_Transicion_final =P_Transicion[j]                 
                max_arg =Delta[i-1][:]  + P_Transicion_final
                Indice = np.argmax(max_arg)
                Psi[i-1].append(Indice)

                # Save the state transition information
                Token,Token_eventos = Guardar_Token(i,j,Indice,Token,Token_eventos,M_E,Secuencia)                   
                temp_product = np.max(max_arg) + P_Observacion[i][M_E[0]][M_E[1]]
                Delta[i][j] = temp_product

        # Final condition for the last frame
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
    S_opt_Indices,S_opt = Backtracking(N_frames,Secuencia,Psi,Delta,N_estados-1)

    return Delta,Psi,S_opt_Indices,S_opt

