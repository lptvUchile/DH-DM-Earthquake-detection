def Guardar_Token(i,j,Coordenada,Token,Token_eventos,M_E,Secuencia):
    """
    Function that saves the information of the previous state that leads to the current state.

    Parameters:
    i (int): Current time step.
    j (int): Current state.
    Coordenada (list): Coordinates representing the current state [model, state].
    Token (list): A 2D list to store token information.
    Token_eventos (list): A 2D list to store token events information.
    M_E (list): Model and state of the current event.
    Secuencia (list): Sequence of events.

    Returns:
    Token (list): Updated 2D list of token information.
    Token_eventos (list): Updated 2D list of token events information.
    """
    if j == Coordenada:  
        Token[i][j] = Token[i-1][j]+1
    else: 
        Token[i][j] = 1
        
    if M_E ==[1,0]:
        if Secuencia[Coordenada] == [2,2]:
            Token_eventos[i][j] = 1
        elif Secuencia[Coordenada] == [1,0]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1
    elif M_E  == [1,1]:
        if Secuencia[Coordenada] == [1,0]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [1,1]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1

    elif M_E  == [1,2]:
        if Secuencia[Coordenada] == [1,1]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [1,2]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1

    elif M_E  == [2,0]:
        if Secuencia[Coordenada] == [1,2]:
            Token_eventos[i][j] = 1
        elif Secuencia[Coordenada] == [2,0]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1

    elif M_E  == [2,1]:
        if Secuencia[Coordenada] == [2,0]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [2,1]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1
        
    elif M_E  == [2,2]:
        if Secuencia[Coordenada] == [2,1]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [2,2]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1

    elif M_E  == [2,3]:
        if Secuencia[Coordenada] == [2,2]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [2,3]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1

    elif M_E  == [2,4]:
        if Secuencia[Coordenada] == [2,3]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [2,4]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1

    elif M_E  == [2,5]:
        if Secuencia[Coordenada] == [2,4]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [2,5]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1
    
    elif M_E  == [2,6]:
        if Secuencia[Coordenada] == [2,5]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [2,6]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1

    elif M_E  == [2,7]:
        if Secuencia[Coordenada] == [2,6]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [2,7]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1

    elif M_E  == [2,8]:
        if Secuencia[Coordenada] == [2,7]:
            Token_eventos[i][j] = Token_eventos[i-1][j-1]+1
        elif Secuencia[Coordenada] == [2,8]:
            Token_eventos[i][j] = Token_eventos[i-1][j]+1
        else:
            Token_eventos[i][j] = 1

    return Token,Token_eventos