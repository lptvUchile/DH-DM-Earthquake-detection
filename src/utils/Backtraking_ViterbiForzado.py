
def Backtraking(N_frames,Secuencia,E,Delta,max_value):
    
    S_opt =[] 
    S_opt_Indices = []
    
    S_opt.append(Secuencia[max_value])
 

    if Delta[N_frames-1][max_value-1]> Delta[N_frames-1][max_value-1]:
        max_value  = max_value - 1
    
    
    for n in range(N_frames-2, -1, -1):
        max_value = E[n][max_value]
        S_opt_Indices.insert(0,max_value)
        S_opt.insert(0,Secuencia[max_value])

    return S_opt_Indices,S_opt