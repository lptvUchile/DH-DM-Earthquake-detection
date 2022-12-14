import numpy as np

from Inicio_Final_Palabra import Inicio_Final_Palabra
def Decoding_Palabras_1fonema(S_opt):

    Ruido = [[],[]]
    Evento = [[],[]]

    for i in range(len(S_opt)):
        if S_opt[i] ==[1,0] :       
            Ruido[0].append(i)
        elif S_opt[i] ==[2,0] :       
            Evento[0].append(i)
        elif S_opt[i] ==[1,2] :       
            Ruido[1].append(i)
        elif S_opt[i] ==[2,2]:       
            Evento[1].append(i)  
    

    Ruido_diff = [list(np.diff(Ruido[0])),list(np.diff(Ruido[1]))]
    Evento_diff = [list(np.diff(Evento[0])),list(np.diff(Evento[1]))]
  
    Palabra_Ruido = Inicio_Final_Palabra(Ruido,Ruido_diff)
    Palabra_Evento = Inicio_Final_Palabra(Evento,Evento_diff)

    return Palabra_Ruido, Palabra_Evento
    


