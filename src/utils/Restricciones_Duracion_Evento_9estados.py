"""--------------------------------Importaciones Librerias--------------------------------"""
import numpy as np



def Restricciones_Duracion_Evento(frame,N_frame,Estado, Modelo, Token_Evento, N_Estados):
    """
    Function that applies duration constraints to any type of event.

    Args:
        State (int): The state being evaluated.
        Model (int): The model being evaluated.
        Token_Event (List of Lists): List of Lists with dimensions (Model, State), where
                    the information corresponds to the duration of the word it originates from.
        Transition_Probs (List of Lists): List of Lists with dimensions (Model, State) 
                    containing transition probabilities to the state being evaluated.

    Returns:
        New_Transition_Probs (List of Lists): List of Lists with dimensions (Model, State) 
                    containing transition probabilities penalized by the duration of the word
                    at the moment of evaluation in a state.
    """
    # A copy is made to avoid altering the base transition probability matrix.
    Nuevas_Probs_Transicion_1 = np.zeros(N_Estados)
    Silencio = False

    cte_min_evento = 1
    cte_max_evento = 1
    cte_min_sil = 1
    cte_max_sil = 0.9

	
    # Bounds for the constraints are defined.
    tau_min_sil =  3
    tau_max_sil = 359

    tau_min_evento = 9
    tau_max_evento = 246

    # These lists correspond to the values of k, alpha, and rho for the gamma probability density function (PDF).
    const_sil = [0.05087268826527303, 0.004083307029459156, 0.4231122743925578]
    const_ev = [0.04106021394602003, 0.005426343202140914, 0.5203863130853138]
    
    """
    # Iquique
    tau_min_sil =  3
    tau_max_sil = 25

    tau_min_evento = 12
    tau_max_evento = 67

    # These lists correspond to the values of k, alpha, and rho for the gamma probability density function (PDF).
    const_sil = [0.01977531557548385, 0.20818115412710006, 2.373265157048941]
    const_ev = [8.342796895597173e-60, 0.8224271267102915, 45.48022010707912]
    """



    # Penalty constraints for the duration of words associated with the word !SIL.
    if frame < N_frame-1:
        if [Modelo,Estado] == [0,0]:
            if Token_Evento[1][8] < tau_min_evento*cte_min_evento:
                Nuevas_Probs_Transicion_1[11] = np.log(0) 

            elif Token_Evento[1][8] >= tau_max_evento*cte_max_evento:
                Nuevas_Probs_Transicion_1[11] = np.log(0) 
                
            
            else:
                tau = Token_Evento[1][8]
                pdf_gamma_ev = np.log(const_ev[0]*np.exp(-const_ev[1]*tau)*tau**(const_ev[2]-1))
                Nuevas_Probs_Transicion_1[11] = pdf_gamma_ev

                    

        elif [Modelo,Estado] == [1,0]:
            if Silencio == True:
                if Token_Evento[0][2] < tau_min_sil*cte_min_sil:
                        Nuevas_Probs_Transicion_1[2] = np.log(0) 

                elif Token_Evento[0][2] >= tau_max_sil*cte_max_sil:
                    Nuevas_Probs_Transicion_1[2] = np.log(0) 


                else:
                    tau = Token_Evento[0][2]
                    pdf_gamma_sil = np.log(const_sil[0]*np.exp(-const_sil[1]*tau)*tau**(const_sil[2]-1))
                    Nuevas_Probs_Transicion_1[2] = pdf_gamma_sil
    else:
        if [Modelo,Estado] == [1,8]:
            if Token_Evento[1][8] < tau_min_evento*cte_min_evento:
                Nuevas_Probs_Transicion_1[11] = np.log(0) 

            elif Token_Evento[1][8] >= tau_max_evento*cte_max_evento:
                Nuevas_Probs_Transicion_1[11] = np.log(0) 
                
            
            else:
                tau = Token_Evento[1][8]
                pdf_gamma_ev = np.log(const_ev[0]*np.exp(-const_ev[1]*tau)*tau**(const_ev[2]-1))
                Nuevas_Probs_Transicion_1[11] = pdf_gamma_ev

     
         

        elif [Modelo,Estado] == [0,2]:
            if Silencio == True:
                if Token_Evento[0][2] < tau_min_sil*cte_min_sil:
                        Nuevas_Probs_Transicion_1[2] = np.log(0) 

                elif Token_Evento[0][2] >= tau_max_sil*cte_max_sil:
                    Nuevas_Probs_Transicion_1[2] = np.log(0) 


                else:
                    tau = Token_Evento[0][2]
                    pdf_gamma_sil = np.log(const_sil[0]*np.exp(-const_sil[1]*tau)*tau**(const_sil[2]-1))
                    Nuevas_Probs_Transicion_1[2] = pdf_gamma_sil
        

    return Nuevas_Probs_Transicion_1