def Guardar_Token(i,j,Coordenada,Token,Token_eventos,M_E,Secuencia):

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



     
        

    

    return Token,Token_eventos