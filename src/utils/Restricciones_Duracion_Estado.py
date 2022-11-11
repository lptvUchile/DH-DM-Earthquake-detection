"""--------------------------------Importaciones Librerias--------------------------------"""
import numpy as np



def Restricciones_Duracion_Estado(Token_Estado,N_Estados,Estado):
    
    
    """
    Función que aplica las restricciones de duración de cualquier tipo de evento.

    Args:
        Token_Estado (Lista de Lista): Lista de Lista con dimensiones (Modelo, Estado), cuya
                    información corresponde a la duración de los estados del frame anterior..
        Estado (int): Estado que se está evaluando.   
        Probs_Transicion (Lista de Lista): Lista de Lista con dimensiones (Modelo, Estado) 
                    cuyo contenido corresponde a las probabilidades de transición al estado que 
                    se está evaluando.


    Returns:
        Nuevas_Probs_Transicion (Lista de Lista): Lista de Lista con dimensiones (Modelo, Estado) 
                    cuyo contenido corresponde a las probabilidades de transición penalizadas
                    por la duración de los estados.
    """
    #Se realiza una copia para no alterar la matriz base de probabilidad de transición.
    #Penalizacion = np.copy(Probs_Transicion)
    Penalizacion = np.zeros(N_Estados)
    Silencio = False

     #Se definen las cotas para las restricciones.
    cte_min = 0.6
    cte_max = 1

    tau_min_S0 = 1
    tau_max_S0 = 750
    tau_min_S1 = 1
    tau_max_S1 = 417
    tau_min_S2 = 1
    tau_max_S2 = 416

    tau_min_E0 = 3
    tau_max_E0 = 45
    tau_min_E1 = 3
    tau_max_E1 = 29
    tau_min_E2 = 1
    tau_max_E2 = 31
    tau_min_E3 = 1
    tau_max_E3 = 33
    tau_min_E4 = 1 
    tau_max_E4 = 40
    tau_min_E5 = 1
    tau_max_E5 = 184
    tau_min_E6 = 1
    tau_max_E6 = 49
    tau_min_E7 = 1
    tau_max_E7 = 65
    tau_min_E8 = 1
    tau_max_E8 = 252

    
    #Estado actual SIL=1   
    if Estado == [0,0]:
        
        if Token_Estado[0][0] < tau_min_S0*cte_min:
            if Silencio == True:
                Penalizacion[0] = np.log(1) 
        if Token_Estado[0][0] >= tau_max_S0*cte_max:
            if Silencio == True:
                Penalizacion[0] = np.log(0) 


        if Token_Estado[1][8] < tau_min_E8*cte_min:
            Penalizacion[11] = np.log(0)
        if Token_Estado[1][8] >= tau_max_E8*cte_max:
            Penalizacion[11] = np.log(1) 

    #Estado actual SIL=2   
    elif Estado == [0,1]:
        if Silencio == True:
            if Token_Estado[0][0] < tau_min_S0*cte_min:
                Penalizacion[0] = np.log(0)
            if Token_Estado[0][0] >= tau_max_S0*cte_max:
                Penalizacion[0] = np.log(1)

        
            if Token_Estado[0][1] < tau_min_S1+cte_min:
                Penalizacion[1] = np.log(1) 
            if Token_Estado[0][1] >= tau_max_S1*cte_max:
                Penalizacion[1] = np.log(0)


    #Estado actual SIL=3   
    elif Estado == [0,2]:
        if Silencio == True:
            if Token_Estado[0][1] < tau_min_S1*cte_min:
                Penalizacion[1] = np.log(0)
            if Token_Estado[0][1] >= tau_max_S1*cte_max:
                Penalizacion[1] = np.log(1)


            if Token_Estado[0][2] < tau_min_S2*cte_min:
                Penalizacion[2] = np.log(1) 
            if Token_Estado[0][2] >= tau_max_S2*cte_max:
                Penalizacion[2] = np.log(0)


    #Estado actual EVENTO=1   
    elif Estado == [1,0]:
   
        if Token_Estado[1][0] < tau_min_E0*cte_min:
            Penalizacion[3] = np.log(1) 
        if Token_Estado[1][0] >= tau_max_E0*cte_max:
            Penalizacion[3] = np.log(0)


        if Token_Estado[0][2] < tau_min_S2*cte_min:
            if Silencio == True:
                Penalizacion[2] = np.log(0)
        if Token_Estado[0][2] >= tau_max_S2*cte_max:
            if Silencio == True:
                Penalizacion[2] = np.log(1) 


    #Estado actual EVENTO=2   
    elif Estado == [1,1]:
    
        if Token_Estado[1][0] < tau_min_E0*cte_min:
            Penalizacion[3] = np.log(0)
        if Token_Estado[1][0] >= tau_max_E0*cte_max:
            Penalizacion[3] = np.log(1) 


        if Token_Estado[1][1] < tau_min_E1*cte_min:
            Penalizacion[4] = np.log(1)
        if Token_Estado[1][1] >= tau_max_E1*cte_max:
            Penalizacion[4] = np.log(0)

  
    #Estado actual EVENTO=3  
    elif Estado == [1,2]:

        if Token_Estado[1][1] < tau_min_E1*cte_min:
            Penalizacion[4] = np.log(0)
        if Token_Estado[1][1] >= tau_max_E1*cte_max:
            Penalizacion[4] = np.log(1) 


        if Token_Estado[1][2] < tau_min_E2*cte_min:
            Penalizacion[5] = np.log(1) 
        if Token_Estado[1][2] >= tau_max_E2*cte_max:
            Penalizacion[5] = np.log(0)


    #Estado actual EVENTO=4  
    elif Estado == [1,3]:

        if Token_Estado[1][2] < tau_min_E2*cte_min:
            Penalizacion[5] = np.log(0)
        if Token_Estado[1][2] >= tau_max_E2*cte_max:
            Penalizacion[5] = np.log(1) 


        if Token_Estado[1][3] < tau_min_E3*cte_min:
            Penalizacion[6] = np.log(1) 
        if Token_Estado[1][3] >= tau_max_E3*cte_max:
            Penalizacion[6] = np.log(0)


    #Estado actual EVENTO=5  
    elif Estado == [1,4]:
   
        if Token_Estado[1][3] < tau_min_E3*cte_min:
            Penalizacion[6] = np.log(0)
        if Token_Estado[1][3] >= tau_max_E3*cte_max:
            Penalizacion[6] = np.log(1) 


        if Token_Estado[1][4] < tau_min_E4*cte_min:
            Penalizacion[7] = np.log(1) 
        if Token_Estado[1][4] >= tau_max_E4*cte_max:
            Penalizacion[7] = np.log(0)


    #Estado actual EVENTO=6  
    elif Estado == [1,5]:
    
        if Token_Estado[1][4] < tau_min_E4*cte_min:
            Penalizacion[7] = np.log(0)
        if Token_Estado[1][4] >= tau_max_E4*cte_max:
            Penalizacion[7] = np.log(1) 


        if Token_Estado[1][5] < tau_min_E5*cte_min:
            Penalizacion[8] = np.log(1) 
        if Token_Estado[1][5] >= tau_max_E5*cte_max:
            Penalizacion[8] = np.log(0)


    #Estado actual EVENTO=7  
    elif Estado == [1,6]:

        if Token_Estado[1][5] < tau_min_E5*cte_min:
            Penalizacion[8] = np.log(0)
        if Token_Estado[1][5] >= tau_max_E5*cte_max:
            Penalizacion[8] = np.log(1) 


        if Token_Estado[1][6] < tau_min_E6*cte_min:
            Penalizacion[9] = np.log(1) 
        if Token_Estado[1][6] >= tau_max_E6*cte_max:
            Penalizacion[9] = np.log(0)


    #Estado actual EVENTO=8  
    elif Estado == [1,7]:
  
        if Token_Estado[1][6] < tau_min_E6*cte_min:
            Penalizacion[9] = np.log(0)
        if Token_Estado[1][6] >= tau_max_E6*cte_max:
            Penalizacion[9] = np.log(1) 


        if Token_Estado[1][7] < tau_min_E7*cte_min:
            Penalizacion[10] = np.log(1) 
        if Token_Estado[1][7] >= tau_max_E7*cte_max:
            Penalizacion[10] = np.log(0)    

    
    #Estado actual EVENTO=9  
    elif Estado == [1,8]:

        if Token_Estado[1][7] < tau_min_E7*cte_min:
            Penalizacion[10] = np.log(0)
        if Token_Estado[1][7] >= tau_max_E7*cte_max:
            Penalizacion[10] = np.log(1) 


        if Token_Estado[1][8] < tau_min_E8*cte_min:
            Penalizacion[11] = np.log(1) 
        if Token_Estado[1][8] >= tau_max_E8*cte_max:
            Penalizacion[11] = np.log(0)    

    return Penalizacion

   