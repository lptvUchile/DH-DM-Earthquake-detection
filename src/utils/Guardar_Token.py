def Guardar_Token(i,j,k,Coordenada,Token,Token_eventos):
    
    if Coordenada == [j,k]:
        Token[i][j].append(Token[i-1][j][k]+1)
    else: 
        Token[i][j].append(1)

    if [j,k]==[0,0]:
        if Coordenada == [1,8]:
            Token_eventos[i][j].append(1)
        elif Coordenada == [0,0]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)
    elif [j,k] == [0,1]:
        if Coordenada == [0,0]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)
        elif Coordenada == [0,1]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)
    elif [j,k] == [0,2]:
        if Coordenada == [0,1]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)
        elif Coordenada == [0,2]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)

    elif [j,k] == [1,0]:
        if Coordenada == [0,2]:
            Token_eventos[i][j].append(1)
        elif Coordenada == [1,0]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)
    elif [j,k] == [1,1]:
        if Coordenada == [1,0]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)
        elif Coordenada == [1,1]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)
        
    elif [j,k] == [1,2]:
        if Coordenada == [1,1]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)
        elif Coordenada == [1,2]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)

    elif [j,k] == [1,3]:
        if Coordenada == [1,2]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)
        elif Coordenada == [1,3]:
                Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)

    elif [j,k] == [1,4]:
        if Coordenada == [1,3]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)

        elif Coordenada == [1,4]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)

    elif [j,k] == [1,5]:
        if Coordenada == [1,4]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)
        elif Coordenada == [1,5]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)

    elif [j,k] == [1,6]:
        if Coordenada == [1,5]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)
        elif Coordenada == [1,6]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
            
        else:
            Token_eventos[i][j].append(1)

    elif [j,k] == [1,7]:
        if Coordenada == [1,6]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)
        elif Coordenada == [1,7]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)

        else:
            Token_eventos[i][j].append(1)
    
    elif [j,k] == [1,8]:
        if Coordenada == [1,7]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k-1]+1)
        elif Coordenada == [1,8]:
            Token_eventos[i][j].append(Token_eventos[i-1][j][k]+1)
        else:
            Token_eventos[i][j].append(1)
    else:
        Token_eventos[i][j].append(1)

    return Token, Token_eventos