import numpy as np
import pandas as pd


def Probs_Transicion_Secuencia(Secuencia):
    """
    Transition probability between states.

    Args:
        Secuencia (list): A list representing a sequence of states, where each state is a list of two elements.

    Returns:
        numpy.ndarray: A matrix of transition probabilities between states.
    """
    
    N_estados_Sec = len(Secuencia)
    Matriz_Prob = np.zeros((N_estados_Sec,N_estados_Sec))
  
    
    for i in range(N_estados_Sec):
        if i  == 0 :

            if Secuencia[i] == [1,0]:
                #Matriz_Prob[i,i] = np.exp(-0.02032683) #NorthChile
                Matriz_Prob[i,i] = np.exp(-0.2014971) #Iquique


            elif Secuencia[i] == [2,0]:
                #Matriz_Prob[i,i] = np.exp(-0.03659784) #NorthChile
                Matriz_Prob[i,i] = np.exp(-0.6557626 )  #Iquique



        else:
         

            if Secuencia[i] == [1,0]:
                
                Matriz_Prob[i,i] = np.exp( -1.701038) #Iquique
                #Matriz_Prob[i,i] = np.exp(-0.02032683 ) #NorthChile


                if  Secuencia[i-1] == [2,8]:
                    
                    #Matriz_Prob[i,i-1] = np.exp( -2.836514) #NorthChile
                    Matriz_Prob[i,i-1] = np.exp(-2.335614 ) #Iquique
               
                
            elif Secuencia[i] == [1,1]:
                #Matriz_Prob[i,i-1] = np.exp(-3.905959 ) #NorthChile
                Matriz_Prob[i,i-1] = np.exp( -1.701038) #Iquique

                #Matriz_Prob[i,i] =  np.exp( -0.02030086) #NorthChile
                Matriz_Prob[i,i] =  np.exp( -0.3656113) #Iquique
     

            elif Secuencia[i] == [1,2]:
                #Matriz_Prob[i,i-1] = np.exp( -3.907224) #NorthChile
                Matriz_Prob[i,i-1] = np.exp( -1.183427) #Iquique

                #Matriz_Prob[i,i] =  np.exp(-0.4598059) #Iquique
                Matriz_Prob[i,i] =  np.exp(-0.0744376 ) #NorthChile



            elif Secuencia[i] == [2,0]:
                #Matriz_Prob[i,i] =  np.exp(-0.03659784) #NorthChile
                Matriz_Prob[i,i] =  np.exp(-0.6557626 ) #Iquique
             

                if  Secuencia[i-1] == [1,2]:
                    #Matriz_Prob[i,i-1] = np.exp(-2.634782 ) #NorthChile
                    Matriz_Prob[i,i-1] = np.exp(-0.9980599) #Iquique



            elif Secuencia[i] == [2,1]:
                #Matriz_Prob[i,i] = np.exp(-0.2006333 )  #NorthChile
                Matriz_Prob[i,i] = np.exp(-0.2711503)   #Iquique

                
                #Matriz_Prob[i,i-1] = np.exp( -3.326009 ) #NorthChile
                Matriz_Prob[i,i-1] = np.exp(-0.731984) #Iquique



            elif Secuencia[i] == [2,2]:
                #Matriz_Prob[i,i] = np.exp(-0.1714975 ) #NorthChile
                Matriz_Prob[i,i] = np.exp(-0.551193) #Iquique

                #Matriz_Prob[i,i-1] = np.exp(-1.704916) #NorthChile
                Matriz_Prob[i,i-1] = np.exp( -1.437596 )  #Iquique


            elif Secuencia[i] == [2,3]:
                #Matriz_Prob[i,i] =  np.exp(-0.1412314) #NorthChile
                Matriz_Prob[i,i] =  np.exp(-0.6337412) #Iquique

                #Matriz_Prob[i,i-1] = np.exp( -1.84771 ) #NorthChile
                Matriz_Prob[i,i-1] = np.exp(-0.8586398) #Iquique


            elif Secuencia[i] == [2,4]:
                #Matriz_Prob[i,i] =  np.exp(-0.1688617 ) #NorthChile
                Matriz_Prob[i,i] =  np.exp(-0.2533131 ) #Iquique
 
                #Matriz_Prob[i,i-1] = np.exp(-2.02714 ) #NorthChile
                Matriz_Prob[i,i-1] = np.exp( -0.7563064 ) #Iquique


            elif Secuencia[i] == [2,5]:
                #Matriz_Prob[i,i] = np.exp(-0.1363701) #NorthChile
                Matriz_Prob[i,i] = np.exp(-0.1844936) #Iquique
 
                #Matriz_Prob[i,i-1] = np.exp(-1.861918) #NorthChile
                Matriz_Prob[i,i-1] = np.exp(-1.497113 ) #Iquique


            elif Secuencia[i] == [2,6]:
                #Matriz_Prob[i,i] = np.exp( -0.1298422) #NorthChile
                Matriz_Prob[i,i] = np.exp(-0.1337733) #Iquique

                #Matriz_Prob[i,i-1] =  np.exp(-2.059793) #NorthChile
                Matriz_Prob[i,i-1] =  np.exp( -1.78097) #Iquique


            elif Secuencia[i] == [2,7]:
                #Matriz_Prob[i,i] =  np.exp(-0.08278222 )  #NorthChile
                Matriz_Prob[i,i] =  np.exp(-0.08080526)  #Iquique 

                #Matriz_Prob[i,i-1] = np.exp(-2.105654) #NorthChile
                Matriz_Prob[i,i-1] = np.exp(-2.07775) #Iquique



            elif Secuencia[i] == [2,8]:
                #Matriz_Prob[i,i] = np.exp(-0.06041869) #NorthChile
                Matriz_Prob[i,i] = np.exp(-0.1017571 ) #Iquique

                #Matriz_Prob[i,i-1] = np.exp(-2.532647) #NorthChile
                Matriz_Prob[i,i-1] = np.exp(-2.555844) #Iquique


            
    Matriz_Prob = np.log(Matriz_Prob)
    return Matriz_Prob



            
            
