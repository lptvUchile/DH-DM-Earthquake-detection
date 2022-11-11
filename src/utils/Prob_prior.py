
import numpy as np

# Ingreso de probabilidades de observacion
#P = np.load('Target/Alineaciones_Viterbi_Forzados_Val_A1.npy',allow_pickle=True)
P = np.load('../../data/Iquique/features/Alineaciones_ViterbiForzado_Iquique_Val.npy',allow_pickle=True)
Ps = np.empty([1,12])
    
p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,T = 0,0,0,0,0,0,0,0,0,0,0,0,0  
for i in range (len(P)):
    z = P[i]
    T = T + len(z)
    p1 = p1 + z.count([1,0])
    p2 = p2 + z.count([1,1])
    p3 = p3 + z.count([1,2])
    
    p4 = p4 + z.count([2,0])
    p5 = p5 + z.count([2,1])
    p6 = p6 + z.count([2,2])
    p7 = p7 + z.count([2,3])
    p8 = p8 + z.count([2,4])
    p9 = p9 + z.count([2,5])
    p10 = p10 + z.count([2,6])
    p11 = p11 + z.count([2,7])
    p12 = p12 + z.count([2,8])

# Noise state    
p1 = p1/T
p2 = p2/T
p3 = p3/T
# Event state
p4 = p4/T
p5 = p5/T
p6 = p6/T
p7 = p7/T
p8 = p8/T
p9 = p9/T
p10 = p10/T
p11 = p11/T
p12 = p12/T

Ps=([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12])
np.save('../../data/Iquique/features/Probs_Prior_Iquique_Val', Ps)

