import numpy as np

def Ecuacion_Prob_Obs(G_const,Media_inv,In_Var,Frame):
    """
    Función que calcula la probabilidad de observación a partir de la inversa de la 
    media y varianzas, y la constante gaussiana de la forma:
    G_const + (1/mu)*x - ((1/var)*x^2)/2
    """


    Probabilidad = G_const + np.dot(Media_inv,Frame) - 0.5*np.dot(In_Var,Frame**2)

    return Probabilidad
