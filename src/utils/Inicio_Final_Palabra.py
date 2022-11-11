
def Inicio_Final_Palabra(Indices,Inicio_Fin):
    
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