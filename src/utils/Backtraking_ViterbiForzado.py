def Backtracking(N_frames, Secuencia, E, Delta, max_value):
    """
    Function that performs backtracking to find the optimal state sequence.

    Args:
        N_frames (int): Number of frames.
        Secuencia (list): List of sequences (not provided in the code).
        E (array): Matrix E (not provided in the code).
        Delta (array): Matrix Delta (not provided in the code).
        max_value (int): Initial maximum value.

    Returns:
        tuple: A tuple containing two lists - one with the optimal state indices and the other with the optimal state values.
    """
    S_opt_Indices = []
    S_opt = []
    
    # Add the initial state to the optimal state lists
    S_opt_Indices.append(max_value)
    S_opt.append(Secuencia[max_value])

    # Check if the current state has a greater Delta value than the previous state
    if Delta[N_frames - 1][max_value - 1] > Delta[N_frames - 1][max_value - 1]:
        max_value = max_value - 1

    # Perform backtracking to determine the optimal state sequence by iterating through frames.
    for n in range(N_frames - 2, -1, -1):
        max_value = E[n][max_value]
        S_opt_Indices.insert(0, max_value)
        S_opt.insert(0, Secuencia[max_value])

    # Return the optimal state indices and state values as a tuple.
    return S_opt_Indices, S_opt
