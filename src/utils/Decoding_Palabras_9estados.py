import numpy as np
from .Inicio_Final_Palabra import Inicio_Final_Palabra

def Decoding_Palabras_SIL3(S_opt):
    """
    Function that provides the start and end frames of silence and seismic events.

    Args:
        S_opt (list): List containing state sequences.

    Returns:
        tuple: A tuple containing the start and end frames of noise and event segments.
    """

    # Initialize lists to store the start and end frames of noise and event segments
    Ruido = [[], []]  # Noise
    Evento = [[], []]  # Event

    # Iterate through the state sequence
    for i in range(len(S_opt)):
        if S_opt[i] == [0, 0]:
            Ruido[0].append(i)
        elif S_opt[i] == [1, 0]:
            Evento[0].append(i)
        elif S_opt[i] == [0, 2]:
            Ruido[1].append(i)
        elif S_opt[i] == [1, 8]:
            Evento[1].append(i)

    # Calculate the differences between consecutive frames in noise and event segments
    Ruido_diff = [list(np.diff(Ruido[0])), list(np.diff(Ruido[1]))]
    Evento_diff = [list(np.diff(Evento[0])), list(np.diff(Evento[1]))]

    # Use the Inicio_Final_Palabra function to determine the start and end frames of segments
    Palabra_Ruido = Inicio_Final_Palabra(Ruido, Ruido_diff)
    Palabra_Evento = Inicio_Final_Palabra(Evento, Evento_diff)

    return Palabra_Ruido, Palabra_Evento



