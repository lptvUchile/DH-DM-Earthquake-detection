
import numpy as np

def Backtraking(N_frames,D,E, max_arg_2,max_arg_3):
    
    S_opt =[]
    max_value = 0
    maximo = np.max([max_arg_2,max_arg_3])

    if max_arg_2 == maximo:
        max_value  = [0,2]
    if max_arg_3 == maximo:
        max_value = [1,8]   
    S_opt.append(max_value)
    
    for n in range(N_frames-2, -1, -1):
        max_value = E[n][max_value[0]][max_value[1]]
        S_opt.insert(0,max_value)

    return S_opt