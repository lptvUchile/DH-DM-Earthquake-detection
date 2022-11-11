
def Palabras_a_Estados(Vocabulario,Palabras):
    
    Secuencia = []
    for i in Palabras:
        if  i == 'UNK':
            N_Estados =  Vocabulario['N_Estados'][0]
            for j in range(N_Estados):
                Secuencia.append([0,j])
      
        elif i =='!SIL':
            N_Estados = Vocabulario['N_Estados'][1]
            for j in range(N_Estados):
                Secuencia.append([1,j])

        elif i == 'EVENTO':
            N_Estados = Vocabulario['N_Estados'][2]
            for j in range(N_Estados):
                Secuencia.append([2,j])
    
    return Secuencia

            
