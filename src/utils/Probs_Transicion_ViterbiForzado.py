import numpy as np
import pandas as pd


def Probs_Transicion_Secuencia(Secuencia):

    N_estados_Sec = len(Secuencia)
    Matriz_Prob = np.zeros((N_estados_Sec,N_estados_Sec))
  
    
    for i in range(N_estados_Sec):
        if i  == 0 :

            if Secuencia[i] == [1,0]:
                #Matriz_Prob[i,i] = np.exp(-0.02013384)
                Matriz_Prob[i,i] = np.exp(-0.2014971)

            elif Secuencia[i] == [2,0]:
                #Matriz_Prob[i,i] = np.exp(-0.6145542)   
                Matriz_Prob[i,i] = np.exp(-0.6557626) 


        else:
         

            if Secuencia[i] == [1,0]:
                #Matriz_Prob[i,i] = np.exp(-0.02013384)
                Matriz_Prob[i,i] = np.exp(-0.2014971) 

                if  Secuencia[i-1] == [2,8]:
                    #Matriz_Prob[i,i-1] = np.exp(-3.569328)
                    Matriz_Prob[i,i-1] = np.exp(-2.335614)
               
                
            elif Secuencia[i] == [1,1]:
                #Matriz_Prob[i,i-1] = np.exp(-3.915403)
                Matriz_Prob[i,i-1] = np.exp(-1.701038)
                #Matriz_Prob[i,i] =  np.exp(-0.02111524)
                Matriz_Prob[i,i] = np.exp(-0.3656113) 

            elif Secuencia[i] == [1,2]:
                #Matriz_Prob[i,i-1] = np.exp(-3.868299)
                Matriz_Prob[i,i-1] = np.exp(-1.183427)
                #Matriz_Prob[i,i] =  np.exp(-0.2764842)
                Matriz_Prob[i,i] = np.exp(-0.4598059) 


            elif Secuencia[i] == [2,0]:
                #Matriz_Prob[i,i] =  np.exp(-0.6145542) 
                Matriz_Prob[i,i] = np.exp(-0.6557626)               

                if  Secuencia[i-1] == [1,2]:
                    Matriz_Prob[i,i-1] = np.exp(-0.9980599 )
                    #Matriz_Prob[i,i-1] = np.exp(-2.095153)



            elif Secuencia[i] == [2,1]:
                #Matriz_Prob[i,i] = np.exp(-0.239385)   
                Matriz_Prob[i,i] = np.exp(-0.2711503 )   
                #Matriz_Prob[i,i-1] = np.exp(-0.778448)
                Matriz_Prob[i,i-1] = np.exp(-0.731984 )


            elif Secuencia[i] == [2,2]:
                #Matriz_Prob[i,i] = np.exp(-0.1980211)
                Matriz_Prob[i,i] = np.exp(-0.551193) 
                #Matriz_Prob[i,i-1] = np.exp(-1.546988) 
                Matriz_Prob[i,i-1] = np.exp(-1.437596 )

            elif Secuencia[i] == [2,3]:
                #Matriz_Prob[i,i] =  np.exp(-0.2733005)
                Matriz_Prob[i,i] = np.exp(-0.6337412) 
                #Matriz_Prob[i,i-1] = np.exp(-1.716759)
                Matriz_Prob[i,i-1] = np.exp(-0.8586398)

            elif Secuencia[i] == [2,4]:
                #Matriz_Prob[i,i] =  np.exp(-0.1352024)
                Matriz_Prob[i,i] = np.exp(-0.2533131 ) 
                #Matriz_Prob[i,i-1] = np.exp(-1.430724)
                Matriz_Prob[i,i-1] = np.exp(-0.7563064)

            elif Secuencia[i] == [2,5]:
                #Matriz_Prob[i,i] = np.exp(-0.1062506)
                Matriz_Prob[i,i] = np.exp(-0.1844936) 
                #Matriz_Prob[i,i-1] = np.exp(-2.067822)
                Matriz_Prob[i,i-1] = np.exp(-1.497113)

            elif Secuencia[i] == [2,6]:
                #Matriz_Prob[i,i] = np.exp(-0.1313068)
                Matriz_Prob[i,i] = np.exp(-0.1337733) 
                #Matriz_Prob[i,i-1] =  np.exp(-2.29461)
                Matriz_Prob[i,i-1] = np.exp(-1.78097)

            elif Secuencia[i] == [2,7]:
                #Matriz_Prob[i,i] =  np.exp(-0.08076475)   
                Matriz_Prob[i,i] = np.exp(-0.08080526)   
                #Matriz_Prob[i,i-1] = np.exp(-2.095153)
                Matriz_Prob[i,i-1] = np.exp(-2.07775)


            elif Secuencia[i] == [2,8]:
                #Matriz_Prob[i,i] = np.exp(-0.02857928)
                Matriz_Prob[i,i] = np.exp(-0.1017571) 
                #Matriz_Prob[i,i-1] = np.exp(-2.556325)
                Matriz_Prob[i,i-1] = np.exp(-2.555844)


            
    Matriz_Prob = np.log(Matriz_Prob)
    return Matriz_Prob



            
            
