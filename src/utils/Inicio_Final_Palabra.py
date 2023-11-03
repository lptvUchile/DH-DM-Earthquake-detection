
def Inicio_Final_Palabra(Indices,Inicio_Fin):
    """
    Determines the start and end frames of a specific state in a sequence.

    Parameters:
    Indices (list): A list of indices representing the state's location in the sequence.
    Inicio_Fin (list): A list of start and end frame information for the state.

    Returns:
    Inicio (list): A list of start frames for the state.
    Fin (list): A list of end frames for the state.
    """
     
     # Code for determining start frames.
    Inicio = []

    if len(Indices[0]) == 1:
        Inicio.append(Indices[0][0])
    elif len(Indices[0]) == 2:
        Inicio.append(Indices[0][0])
        if Inicio_Fin[0][0] > 1:
            Inicio.append(Indices[0][1])

    else:       
        for i in range(len(Inicio_Fin[0])):
            if i== 0:
                Inicio.append(Indices[0][i])
                if  Inicio_Fin[0][i] > 1:     
                    Inicio.append(Indices[0][i + 1])    
            else:
                if Inicio_Fin[0][i] > 1:
                    Inicio.append(Indices[0][i+1])  

    # Code for determining end frames..
    Fin = []

    if len(Indices[1]) == 1:
        Fin.append(Indices[1][0])
    elif len(Indices[1]) == 2:
        Fin.append(Indices[1][0])
        if Inicio_Fin[1][0] > 1:
            Fin.append(Indices[1][1])
    else:       
        for i in range(len(Inicio_Fin[1])):
            if i == len(Inicio_Fin[1]) - 1:
                if  Inicio_Fin[1][i] > 1:
                    Fin.append(Indices[1][i])
                Fin.append(Indices[1][i + 1])
                

            else:
                if Inicio_Fin[1][i] > 1:
                    Fin.append(Indices[1][i]) 
   
    return Inicio,Fin