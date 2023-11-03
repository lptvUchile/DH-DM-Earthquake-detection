import numpy as np

def Parametros_Kaldi(Lineas_archivo):
    """
    Extracts information about means, variances, and Gaussian constants for each state.

    Args:
        Lineas_archivo (list): A list of lines from a Kaldi file containing parameter information.

    Returns:
        tuple: A tuple containing lists of Gaussian constants, means and inverse variances, and the number of states.
    """
    GCONSTS = []          # List to store Gaussian constants.
    MEANS_INVVARS = []    # List to store means and inverse variances.
    INV_VARS = []         # List to store inverse variances.
    N_Estados = None      # Number of states (will be initialized).

    for i in range(len(Lineas_archivo)):
        Parametro = Lineas_archivo[i].split(' ')
        if len(Parametro) > 2 and Parametro[2] == '<NUMPDFS>':
            # If the line indicates the number of states, extract and store it.
            N_Estados = int(Parametro[3])

        elif Parametro[0] == '<GCONSTS>':
            # If the line indicates Gaussian constants, extract and store them.
            G = Lineas_archivo[i].split('  ')[1].split(' ')
            G = [x for x in G if x not in ['[', ']\n']]
            G = np.asarray(list(map(float, G)))
            GCONSTS.append(G)
            numgauss = len(G)

        elif Parametro[0] == '<MEANS_INVVARS>':
            # If the line indicates means and inverse variances, extract and store them.
            for k in range(numgauss):
                Media_inv = Lineas_archivo[i + 1 + k].split(' ')
                Media_inv = [x for x in Media_inv if x not in ['', '\n', ']\n']]
                Media_inv = np.asarray(list(map(float, Media_inv)))
                MEANS_INVVARS.append(Media_inv)

        elif Parametro[0] == '<INV_VARS>':
            # If the line indicates inverse variances, extract and store them.
            for k in range(numgauss):
                In_Var = Lineas_archivo[i + 1 + k].split(' ')
                In_Var = [x for x in In_Var if x not in ['', '\n', ']\n']]
                In_Var = np.asarray(list(map(float, In_Var)))
                INV_VARS.append(In_Var)

    return GCONSTS, MEANS_INVVARS, INV_VARS, N_Estados
