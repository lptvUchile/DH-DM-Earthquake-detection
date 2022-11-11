import numpy as np

def Ecuacion_Prob_Obs(G_const,Media_inv,In_Var,Frame):

    Probabilidad = G_const + np.dot(Media_inv,Frame) - 0.5*np.dot(In_Var,Frame**2)

    return Probabilidad
