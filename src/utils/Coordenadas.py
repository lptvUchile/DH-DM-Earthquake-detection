import numpy as np
#Me falta comentar la función

def Coordenadas(D,max_arg):
    N_modelos = np.shape(D)[0]
    Indice = np.argmax(max_arg)
 
    for i in range(N_modelos):
        N_estados = np.shape(D[i])[0]
        for j in range(N_estados):
            if Indice == j:
                Posicion = [i,j]
        Indice = Indice - N_estados 

    return Posicion