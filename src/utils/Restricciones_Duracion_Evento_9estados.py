"""--------------------------------Importaciones Librerias--------------------------------"""
import numpy as np



def Restricciones_Duracion_Evento(frame,N_frame,Estado, Modelo, Token_Evento, N_Estados):
    
    """
    Función que aplica las restricciones de duración de cualquier tipo de evento.

    Args:
        Estado (int): Estado que se está evaluando.
        Modelo (int): Modelo que se está evaluando.
        Token_Evento (Lista de Lista): Lista de Lista con dimensiones (Modelo, Estado), cuya
                    información corresponde a la duración de la palabra de la que proviene.
        Probs_Transicion (Lista de Lista): Lista de Lista con dimensiones (Modelo, Estado) 
                    cuyo contenido corresponde a las probabilidades de transición al estado que 
                    se está evaluando.


    Returns:
        Nuevas_Probs_Transicion (Lista de Lista): Lista de Lista con dimensiones (Modelo, Estado) 
                    cuyo contenido corresponde a las probabilidades de transición penalizadas
                    por la duración que tiene la palabra al momento de evaluarla en un estado.
    """
    #Se realiza una copia para no alterar la matriz base de probabilidad de transición.
    #Nuevas_Probs_Transicion_1 = np.copy(Probs_Transicion_1)
    Nuevas_Probs_Transicion_1 = np.zeros(N_Estados)
    Silencio =  False

    cte_min = 1
    cte_max = 1

	
    #Se definen las cotas para las restricciones.
    tau_min_sil =  3
    tau_max_sil = 1251

    tau_min_evento = 14
    tau_max_evento = 325

    #Estas listas corresponden a los valores de k,alpha,rho de la pdf gamma.
    const_sil = [0.09740596205154974,0.004206177606744674,0.2336910216531273]
    const_ev = [1.543801734751951e-05,0.058826820965994404,0.490254114733414]

    
    #Restricciones de penalización para la duración de palabras asociada a la palabra !SIL.
    if frame < N_frame-1:
        if [Modelo,Estado] == [0,0]:
            if Token_Evento[1][8] < tau_min_evento*cte_min:
                Nuevas_Probs_Transicion_1[11] = np.log(0) 

            elif Token_Evento[1][8] >= tau_max_evento*cte_max:
                Nuevas_Probs_Transicion_1[11] = np.log(0) 
                
            
            else:
                tau = Token_Evento[1][8]
                pdf_gamma_ev = np.log(const_ev[0]*np.exp(-const_ev[1]*tau)*tau**(const_ev[2]-1))
                Nuevas_Probs_Transicion_1[11] = pdf_gamma_ev

                    

        elif [Modelo,Estado] == [1,0]:
            if Silencio == True:
                if Token_Evento[0][2] < tau_min_sil*cte_min:
                        Nuevas_Probs_Transicion_1[2] = np.log(0) 

                elif Token_Evento[0][2] >= tau_max_sil*cte_max:
                    Nuevas_Probs_Transicion_1[2] = np.log(0) 


                else:
                    tau = Token_Evento[0][2]
                    pdf_gamma_sil = np.log(const_sil[0]*np.exp(-const_sil[1]*tau)*tau**(const_sil[2]-1))
                    Nuevas_Probs_Transicion_1[2] = pdf_gamma_sil
    else:
        if [Modelo,Estado] == [1,8]:
            if Token_Evento[1][8] < tau_min_evento*cte_min:
                Nuevas_Probs_Transicion_1[11] = np.log(0) 

            elif Token_Evento[1][8] >= tau_max_evento*cte_max:
                Nuevas_Probs_Transicion_1[11] = np.log(0) 
                
            
            else:
                tau = Token_Evento[1][8]
                pdf_gamma_ev = np.log(const_ev[0]*np.exp(-const_ev[1]*tau)*tau**(const_ev[2]-1))
                Nuevas_Probs_Transicion_1[11] = pdf_gamma_ev

     
         

        elif [Modelo,Estado] == [0,2]:
            if Silencio == True:
                if Token_Evento[0][2] < tau_min_sil*cte_min:
                        Nuevas_Probs_Transicion_1[2] = np.log(0) 

                elif Token_Evento[0][2] >= tau_max_sil*cte_max:
                    Nuevas_Probs_Transicion_1[2] = np.log(0) 


                else:
                    tau = Token_Evento[0][2]
                    pdf_gamma_sil = np.log(const_sil[0]*np.exp(-const_sil[1]*tau)*tau**(const_sil[2]-1))
                    Nuevas_Probs_Transicion_1[2] = pdf_gamma_sil
        
            

    return Nuevas_Probs_Transicion_1