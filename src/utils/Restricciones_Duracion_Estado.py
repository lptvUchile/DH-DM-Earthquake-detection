"""--------------------------------Importaciones Librerias--------------------------------"""
import numpy as np



def Restricciones_Duracion_Estado(Token_Estado,N_Estados,Estado):
    
    
    """
    Function that applies duration constraints to any type of event.

    Args:
        Token_State (List of Lists): List of Lists with dimensions (Model, State), where
                    the information corresponds to the duration of states from the previous frame.
        State (int): The state being evaluated.
        Transition_Probs (List of Lists): List of Lists with dimensions (Model, State) 
                    containing transition probabilities to the state being evaluated.

    Returns:
        New_Transition_Probs (List of Lists): List of Lists with dimensions (Model, State) 
                    containing transition probabilities penalized by state duration.
    """
    # A copy is made to avoid altering the base transition probability matrix.
    # Penalty = np.copy(Transition_Probs)
    Penalizacion = np.zeros(N_Estados)
    Silencio = False

    # The bounds for the constraints are defined.
  
    #NorthChile
    tau_min_S0 = 1
    tau_max_S0 = 635
    tau_min_S1 = 1
    tau_max_S1 = 426
    tau_min_S2 = 1
    tau_max_S2 = 182

    tau_min_E0 = 1
    tau_max_E0 = 98
    tau_min_E1 = 1
    tau_max_E1 = 24
    tau_min_E2 = 1
    tau_max_E2 = 24
    tau_min_E3 = 1
    tau_max_E3 = 25
    tau_min_E4 = 1 
    tau_max_E4 = 30
    tau_min_E5 = 1
    tau_max_E5 = 18
    tau_min_E6 = 1
    tau_max_E6 = 22
    tau_min_E7 = 1
    tau_max_E7 = 44
    tau_min_E8 = 1
    tau_max_E8 = 72
    

    cte_min_0 = 1
    cte_min_1 = 1
    cte_min_2 = 1
    cte_min_3 = 1
    cte_min_4 = 1
    cte_min_5 = 1
    cte_min_6 = 1
    cte_min_7 = 1
    cte_min_8 = 1
    cte_min_9 = 1
    cte_min_10 = 1
    cte_min_11 = 1

    lista_max = [0.07,0.80,0.10,0.70,0.90,0.90,0.90,1.60,1,0.80,0.70,1]
    cte_max_0 = lista_max[0]
    cte_max_1 = lista_max[1]
    cte_max_2 = lista_max[2]
    cte_max_3 = lista_max[3]
    cte_max_4 = lista_max[4]
    cte_max_5 = lista_max[5]
    cte_max_6 = lista_max[6]
    cte_max_7 = lista_max[7]
    cte_max_8 = lista_max[8]
    cte_max_9 = lista_max[9]
    cte_max_10 = lista_max[10]
    cte_max_11 = lista_max[11]
    """
    #Iquique
    
    tau_min_S0 = 1
    tau_max_S0 = 5
    tau_min_S1 = 1
    tau_max_S1 = 15
    tau_min_S2 = 1
    tau_max_S2 = 7

    tau_min_E0 = 1
    tau_max_E0 = 3
    tau_min_E1 = 1
    tau_max_E1 = 14
    tau_min_E2 = 1
    tau_max_E2 = 3
    tau_min_E3 = 1
    tau_max_E3 = 3
    tau_min_E4 = 1 
    tau_max_E4 = 7
    tau_min_E5 = 1
    tau_max_E5 = 10
    tau_min_E6 = 1
    tau_max_E6 = 13
    tau_min_E7 = 1
    tau_max_E7 = 20
    tau_min_E8 = 1
    tau_max_E8 = 25



    cte_min_0 = 1
    cte_min_1 = 1
    cte_min_2 = 1
    cte_min_3 = 1
    cte_min_4 = 1
    cte_min_5 = 1
    cte_min_6 = 1
    cte_min_7 = 1
    cte_min_8 = 1
    cte_min_9 = 1
    cte_min_10 = 1
    cte_min_11 = 1
    

    lista_max = [1.5,0.90,1.2,0.70,1,1,1,1,1,1,1,1]
    cte_max_0 = lista_max[3]
    cte_max_1 = lista_max[4]
    cte_max_2 = lista_max[5]
    cte_max_3 = lista_max[6]
    cte_max_4 = lista_max[7]
    cte_max_5 = lista_max[8]
    cte_max_6 = lista_max[9]
    cte_max_7 = lista_max[10]
    cte_max_8 = lista_max[11]
    cte_max_9 = lista_max[0]
    cte_max_10 = lista_max[1]
    cte_max_11 = lista_max[2]
    """

    # Current state SIL=1   
    if Estado == [0,0]:
        
        if Token_Estado[0][0] < tau_min_S0*cte_min_9:
            if Silencio == True:
                Penalizacion[0] = np.log(1) 
        if Token_Estado[0][0] >= tau_max_S0*cte_max_9:
            if Silencio == True:
                Penalizacion[0] = np.log(0) 


        if Token_Estado[1][8] < tau_min_E8*cte_min_8:
            Penalizacion[11] = np.log(0)
        if Token_Estado[1][8] >= tau_max_E8*cte_max_8:
            Penalizacion[11] = np.log(1) 

    # Current state SIL=2   
    elif Estado == [0,1]:
        if Silencio == True:
            if Token_Estado[0][0] < tau_min_S0*cte_min_9:
                Penalizacion[0] = np.log(0)
            if Token_Estado[0][0] >= tau_max_S0*cte_max_9:
                Penalizacion[0] = np.log(1)

        
            if Token_Estado[0][1] < tau_min_S1+cte_min_10:
                Penalizacion[1] = np.log(1) 
            if Token_Estado[0][1] >= tau_max_S1*cte_max_10:
                Penalizacion[1] = np.log(0)


    # Current state SIL=3   
    elif Estado == [0,2]:
        if Silencio == True:
            if Token_Estado[0][1] < tau_min_S1*cte_min_10:
                Penalizacion[1] = np.log(0)
            if Token_Estado[0][1] >= tau_max_S1*cte_max_10:
                Penalizacion[1] = np.log(1)


            if Token_Estado[0][2] < tau_min_S2*cte_min_11:
                Penalizacion[2] = np.log(1) 
            if Token_Estado[0][2] >= tau_max_S2*cte_max_11:
                Penalizacion[2] = np.log(0)


    # Current state EVENTO=1   
    elif Estado == [1,0]:
   
        if Token_Estado[1][0] < tau_min_E0*cte_min_0:
            Penalizacion[3] = np.log(1) 
        if Token_Estado[1][0] >= tau_max_E0*cte_max_0:
            Penalizacion[3] = np.log(0)


        if Token_Estado[0][2] < tau_min_S2*cte_min_11:
            if Silencio == True:
                Penalizacion[2] = np.log(0)
        if Token_Estado[0][2] >= tau_max_S2*cte_max_11:
            if Silencio == True:
                Penalizacion[2] = np.log(1) 


    # Current state EVENTO=2   
    elif Estado == [1,1]:
    
        if Token_Estado[1][0] < tau_min_E0*cte_min_0:
            Penalizacion[3] = np.log(0)
        if Token_Estado[1][0] >= tau_max_E0*cte_max_0:
            Penalizacion[3] = np.log(1) 


        if Token_Estado[1][1] < tau_min_E1*cte_min_1:
            Penalizacion[4] = np.log(1)
        if Token_Estado[1][1] >= tau_max_E1*cte_max_1:
            Penalizacion[4] = np.log(0)
  
    # Current state EVENTO=3  
    elif Estado == [1,2]:

        if Token_Estado[1][1] < tau_min_E1*cte_min_1:
            Penalizacion[4] = np.log(0)
        if Token_Estado[1][1] >= tau_max_E1*cte_max_1:
            Penalizacion[4] = np.log(1) 


        if Token_Estado[1][2] < tau_min_E2*cte_min_2:
            Penalizacion[5] = np.log(1) 
        if Token_Estado[1][2] >= tau_max_E2*cte_max_2:
            Penalizacion[5] = np.log(0)


    # Current state EVENTO=4  
    elif Estado == [1,3]:

        if Token_Estado[1][2] < tau_min_E2*cte_min_2:
            Penalizacion[5] = np.log(0)
        if Token_Estado[1][2] >= tau_max_E2*cte_max_2:
            Penalizacion[5] = np.log(1) 


        if Token_Estado[1][3] < tau_min_E3*cte_min_3:
            Penalizacion[6] = np.log(1) 
        if Token_Estado[1][3] >= tau_max_E3*cte_max_3:
            Penalizacion[6] = np.log(0)


    # Current state EVENTO=5  
    elif Estado == [1,4]:
   
        if Token_Estado[1][3] < tau_min_E3*cte_min_3:
            Penalizacion[6] = np.log(0)
        if Token_Estado[1][3] >= tau_max_E3*cte_max_3:
            Penalizacion[6] = np.log(1) 


        if Token_Estado[1][4] < tau_min_E4*cte_min_4:
            Penalizacion[7] = np.log(1) 
        if Token_Estado[1][4] >= tau_max_E4*cte_max_4:
            Penalizacion[7] = np.log(0)


    # Current state EVENTO=6  
    elif Estado == [1,5]:
    
        if Token_Estado[1][4] < tau_min_E4*cte_min_4:
            Penalizacion[7] = np.log(0)
        if Token_Estado[1][4] >= tau_max_E4*cte_max_4:
            Penalizacion[7] = np.log(1) 


        if Token_Estado[1][5] < tau_min_E5*cte_min_5:
            Penalizacion[8] = np.log(1) 
        if Token_Estado[1][5] >= tau_max_E5*cte_max_5:
            Penalizacion[8] = np.log(0)


    # Current state EVENTO=7  
    elif Estado == [1,6]:

        if Token_Estado[1][5] < tau_min_E5*cte_min_5:
            Penalizacion[8] = np.log(0)
        if Token_Estado[1][5] >= tau_max_E5*cte_max_5:
            Penalizacion[8] = np.log(1) 


        if Token_Estado[1][6] < tau_min_E6*cte_min_6:
            Penalizacion[9] = np.log(1) 
        if Token_Estado[1][6] >= tau_max_E6*cte_max_6:
            Penalizacion[9] = np.log(0)


    # Current state EVENTO=8  
    elif Estado == [1,7]:
  
        if Token_Estado[1][6] < tau_min_E6*cte_min_6:
            Penalizacion[9] = np.log(0)
        if Token_Estado[1][6] >= tau_max_E6*cte_max_6:
            Penalizacion[9] = np.log(1) 


        if Token_Estado[1][7] < tau_min_E7*cte_min_7:
            Penalizacion[10] = np.log(1) 
        if Token_Estado[1][7] >= tau_max_E7*cte_max_7:
            Penalizacion[10] = np.log(0)     

    
    # Current state EVENTO=9  
    elif Estado == [1,8]:

        if Token_Estado[1][7] < tau_min_E7*cte_min_7:
            Penalizacion[10] = np.log(0)
        if Token_Estado[1][7] >= tau_max_E7*cte_max_7:
            Penalizacion[10] = np.log(1) 


        if Token_Estado[1][8] < tau_min_E8*cte_min_8:
            Penalizacion[11] = np.log(1) 
        if Token_Estado[1][8] >= tau_max_E8*cte_max_8:
            Penalizacion[11] = np.log(0)  

    return Penalizacion

   