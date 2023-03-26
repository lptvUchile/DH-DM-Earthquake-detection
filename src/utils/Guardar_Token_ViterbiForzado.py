def Guardar_Token(i,j,Coordenada,Token,Token_eventos,M_E,Secuencia):
    # Funcion que guarda la información del estado anterior del que proviene el estado actual

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