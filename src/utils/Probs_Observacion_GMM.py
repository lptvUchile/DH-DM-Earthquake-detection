import numpy as np
from scipy.special import logsumexp
from Parametros_Kaldi import Parametros_Kaldi
from Ecuacion_Prob_Obs import Ecuacion_Prob_Obs


def Probs_Observacion(Utt, Lineas_archivo, Vocabulario):
    """
    Function that creates an observation probability matrix from estimated data using GMM.
    
    Args:
        Utt (numpy.ndarray): An input matrix of frames for which observation probabilities are calculated.
        Lineas_archivo (list): A list of lines from a Kaldi file containing parameter information.
        Vocabulario (dict): A dictionary with vocabulary and state information.

    Returns:
        list: A list of observation probability matrices for each frame, separated by state.
    """
    Estados_vocab = np.cumsum(list(Vocabulario['N_Estados']))
    [GCONSTS, MEANS_INVVARS, INV_VARS, N_Estados] = Parametros_Kaldi(Lineas_archivo)
    N_frames = Utt.shape[0]
    Matriz_Prob = np.zeros((N_frames, N_Estados))
    Matriz_Prob_model = []

    for i in range(N_frames):
        num = 0
        for j in range(N_Estados):
            # This is for one state.
            numgauss = len(GCONSTS[j])
            Probabilidad = np.zeros((numgauss))
            for k in range(numgauss):
                # Calculate component of a Gaussian.
                Probabilidad[k] = Ecuacion_Prob_Obs(GCONSTS[j][k], MEANS_INVVARS[k + num], INV_VARS[k + num], Utt[i])
            num = numgauss + num
            Matriz_Prob[i][j] = logsumexp(Probabilidad)

        Separar_Matriz_Prob = np.split(Matriz_Prob[i], Estados_vocab)[0:len(Estados_vocab)]
        Matriz_Prob_model.append([i for i in Separar_Matriz_Prob])

    return Matriz_Prob_model
