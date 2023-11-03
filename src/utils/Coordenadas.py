import numpy as np


def Coordenadas(D, max_arg):
    """
    Returns the [model, state] position with the highest cumulative probability.

    Args:
        D (array): A matrix containing probabilities.
        max_arg (array): Array of maximum probabilities.

    Returns:
        list: A list containing the [model, state] position with the highest cumulative probability.
    """

    N_models = np.shape(D)[0]
    Index = np.argmax(max_arg)

    for i in range(N_models):
        N_states = np.shape(D[i])[0]
        for j in range(N_states):
            if Index == j:
                Position = [i, j]
        Index = Index - N_states

    return Position
