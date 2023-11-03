def Palabras_a_Estados(Vocabulario, Palabras):
    """
    Maps words to a sequence of states.

    Args:
        Vocabulary (dict): A dictionary containing information about the number of states for different word types.
        Words (list): A list of words to be mapped to states.

    Returns:
        list: A list of state sequences corresponding to the input words.
    """
    
    Secuencia = []  # Initialize an empty list to store the state sequences.

    for i in Palabras:
        if i == 'UNK':
            # If the word is 'UNK', retrieve the number of states for the 'UNK' word type from the vocabulary.
            N_Estados = Vocabulario['N_Estados'][0]
            # Create state sequences for the 'UNK' word type.
            for j in range(N_Estados):
                Secuencia.append([0, j])  # Append state information to the sequence.

        elif i == '!SIL':
            # If the word is '!SIL', retrieve the number of states for the '!SIL' word type from the vocabulary.
            N_Estados = Vocabulario['N_Estados'][1]
            # Create state sequences for the '!SIL' word type.
            for j in range(N_Estados):
                Secuencia.append([1, j])  # Append state information to the sequence.

        elif i == 'EVENTO':
            # If the word is 'EVENTO', retrieve the number of states for the 'EVENTO' word type from the vocabulary.
            N_Estados = Vocabulario['N_Estados'][2]
            # Create state sequences for the 'EVENTO' word type.
            for j in range(N_Estados):
                Secuencia.append([2, j])  # Append state information to the sequence.

    return Secuencia  # Return the list of state sequences corresponding to the input words.

            
