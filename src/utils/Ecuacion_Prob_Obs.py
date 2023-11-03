import numpy as np

def Ecuacion_Prob_Obs(G_const, Media_inv, In_Var, Frame):
    """
    Function that calculates the observation probability based on the inverse of the mean and variances,
    and the Gaussian constant using the formula: G_const + (1/mu)*x - ((1/var)*x^2)/2
    Args:
        G_const (float): Gaussian constant.
        Media_inv (array): Inverse of the mean.
        In_Var (array): Inverse of the variance.
        Frame (array): Input frame data.

    Returns:
        float: The calculated observation probability.
    """

    # Calculate the observation probability using the provided formula
    Probabilidad = G_const + np.dot(Media_inv, Frame) - 0.5 * np.dot(In_Var, Frame**2)

    return Probabilidad

