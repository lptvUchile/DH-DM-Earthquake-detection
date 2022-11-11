
import numpy as np
from scipy.special import exp10

def Probabilidad_ML(i,j,k,N_frames,N_estados,a):
    P_Transicion_Nuevas_2 = np.zeros(a)
    if k == 0 and i != N_frames-1 and i>1 :                    
      
        if j == 0:
           P_Transicion_Nuevas_2[11] = np.log(exp10(-0.0386)) 


        elif j == 1:
            P_Transicion_Nuevas_2[2] = np.log(exp10(-0.1160)) 
        
    elif k == N_estados-1 and i == N_frames -1:

        if j == 0:
            P_Transicion_Nuevas_2[1] = np.log(exp10(-0.6311))
            P_Transicion_Nuevas_2[2] = np.log(exp10(-0.6311))

        elif j == 1:
            P_Transicion_Nuevas_2[10] =  np.log(exp10(-1.0744 )) 
            P_Transicion_Nuevas_2[11] =  np.log(exp10(-1.0744 )) 

    return P_Transicion_Nuevas_2
  