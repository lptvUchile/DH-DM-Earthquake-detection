import numpy as np
from copy import deepcopy


def eliminar_duplicados(lista):
    nueva_lista = []
    for elemento in lista:
        if not elemento in nueva_lista:
            nueva_lista.append(elemento)
    return nueva_lista



def Prob_Transicion_automatico(Vocabulario,lineas,Phones_lineas,type_exp):

    LogProbs =[]
    for j in range(len(lineas)):
        if lineas[j] == '<LogProbs> \n':
            vector = lineas[j+1].split(' ')[3:-1]
            LogProbs.append(vector)

    #Sacamos los triples con pdf
    Triples = 'No'
    Triples_vector = []
    for j in range(len(lineas)):
        if lineas[j].split(' ')[0] == '<Triples>':
            Triples = 'Si'
        if lineas[j].split(' ')[0] == '</Triples>':
            Triples = 'No'
        if Triples == 'Si':
            vector = lineas[j].split(' ')[0:-1]
            if vector[0]!= '<Triples>':
                Triples_vector.append(vector)


                                   


    Phones_words = [[],[]]
    Phones_number = [[],[]]


    types = ['_B','_E','_I','_S']
    for j in range(len(Phones_lineas)):
        list_linea = Phones_lineas[j].split('\n')[0].split(' ')[0]
        number_lineas = Phones_lineas[j].split('\n')[0].split(' ')[1]
        if list_linea.split('_B')[0] == '_sil' or list_linea.split('_E')[0] == '_sil' or list_linea.split('_I')[0] == '_sil' \
           or list_linea.split('_S')[0] == '_sil':
            Phones_words[0].append([list_linea])
            Phones_number[0].append([number_lineas])
        elif list_linea.split('_B')[0] == '_evento' or list_linea.split('_E')[0] == '_evento' or list_linea.split('_I')[0] == '_evento' \
           or list_linea.split('_S')[0] == '_evento' or list_linea.split('_ini')[0] == '_evento' or list_linea.split('_mid')[0] == '_evento'\
           or list_linea.split('_end')[0] == '_evento':
            Phones_words[1].append([list_linea])
            Phones_number[1].append([number_lineas])

    #Sacamos los fonemas innecesarios. Nos quedamos solo con los _s.
    if type_exp == 'mono':
        pdf = []
        for i in range(len(Phones_number)):
            pdf.append([])
            for j in range(len(Phones_number[i])):
                #pdf[i].append([])
                #for l in range(3):
                #    pdf[i][j].append([])
                for k in Triples_vector:
                    if Phones_number[i][j][0]==k[0]:
                        #print(int(k[1]))
                        pdf[i].append([k[2]]) 

    else:

        pdf = []
        for i in range(len(Phones_number)):
            pdf.append([])
            for j in range(len(Phones_number[i])):
                pdf[i].append([])
                for l in range(3):
                    pdf[i][j].append([])
                for k in Triples_vector:
                    if Phones_number[i][j][0]==k[0]:
                        pdf[i][j][int(k[1])].append(k[2]) 

        for i in range(len(pdf)):
            pdf[i] = eliminar_duplicados(pdf[i])[0]
            Phones_words[i] = Phones_words[i][-1]
            Phones_number[i] = Phones_number[i][-1]



    #Buscamos las probabilidades de transicion asociado a cada estado de cada fonema.
     
    Posiciones = deepcopy(pdf)
    Cantidad_estados = 0
    for i in range(len(pdf)):#Fonema
        for j in range(len(pdf[i])):
            for k in range(len(pdf[i][j])):
                for l in range(len(Triples_vector)):                        
                    if pdf[i][j][k]==Triples_vector[l][2]:
                        Posiciones[i][j][k] = l
                Cantidad_estados = Cantidad_estados+1
                  

                    
                  

    #Creamos matriz de probabilidades de transicion
    n=1
    Posiciones_inicio_palabras = []
    for i in range(len(Posiciones)): #fonema
        Posiciones_inicio_palabras.append([])
        for j in range(len(Posiciones[i])):
            for k in Posiciones[i][j]:            
                if j == 0:
                    Posiciones_inicio_palabras[i].append(n-1)
                n=n+1
                    
            
        
        
      
                
    Matriz_transicion = np.full((Cantidad_estados,Cantidad_estados), -np.inf)
    n=0
    for i in range(len(Posiciones)): #fonema
        for j in range(len(Posiciones[i])): #estado
            for k in range(len(Posiciones[i][j])): #pdf
                Matriz_transicion[n,n] = float(LogProbs[0][Posiciones[i][j][k]*2])            
                if n +1 == len(Posiciones[0])+ len(Posiciones[1]):
                     Matriz_transicion[0,n] = float(LogProbs[0][Posiciones[i][j][k]*2+1])
                else:
                    Matriz_transicion[n+1,n] = float(LogProbs[0][Posiciones[i][j][k]*2+1])
            
                        
                n=n+1

            

    #Completar diccionario

    Vocabulario['Phones'] = Phones_words
    Vocabulario['Number_Phones'] = Phones_number


   
    Estados_vocab = list(Vocabulario['N_Estados']) 
    Estados_Estados_cumsum = np.cumsum(list(Vocabulario['N_Estados']))
    if type_exp:  
        #Se divide la matriz por la cantidad de modelos total que se tiene.
        Separar_Matriz_Prob = np.vsplit(Matriz_transicion,Estados_Estados_cumsum)[0:len(Estados_vocab)]
        
        #Se transforma a lista para ser consistente en el código
        Matriz_Prob_Transicion = [i.tolist() for i in Separar_Matriz_Prob]
       
    return Matriz_Prob_Transicion,Vocabulario




   









