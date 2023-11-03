import numpy as np

# Define a function named Backtracking that takes five input arguments
def Backtracking(N_frames, D, E, max_arg_2, max_arg_3):
    """
    Function that performs backtracking to find the optimal sequence of states.

    Args:
        N_frames (int): Number of frames.
        D (array): Matrix D (not provided in the code).
        E (array): Matrix E (not provided in the code).
        max_arg_2 (int): Maximum argument 2.
        max_arg_3 (int): Maximum argument 3.

    Returns:
        list: A list containing the optimal state sequence.
    """

    # Initialize an empty list to store the optimal state sequence and variables to track maximum values.
    S_opt = []
    max_value = 0

    # Determine the maximum value between max_arg_2 and max_arg_3
    maximo = np.max([max_arg_2, max_arg_3])

    # Identify which argument corresponds to the maximum value and store the state transition information in max_value.
    if max_arg_2 == maximo:
        max_value = [0, 2]
    if max_arg_3 == maximo:
        max_value = [1, 8]

    # Add the initial maximum state to the list.
    S_opt.append(max_value)

    # Perform backtracking to determine the optimal state sequence by iterating through frames.
    for n in range(N_frames - 2, -1, -1):
        max_value = E[n][max_value[0]][max_value[1]]  # Update max_value based on matrix E
        S_opt.insert(0, max_value)  # Insert the new maximum state at the beginning of the list.

    # Return the optimal state sequence as a list.
    return S_opt
