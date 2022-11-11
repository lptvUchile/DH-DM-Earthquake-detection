import numpy as np

def Parametros_Kaldi(Lineas_archivo):
    GCONSTS = []
    MEANS_INVVARS = []
    INV_VARS = []

    for i in range(len(Lineas_archivo)):
        Parametro = Lineas_archivo[i].split(' ')
        if len(Parametro) > 2 and Parametro[2] == '<NUMPDFS>':
           N_Estados = int(Parametro[3])

        elif Parametro[0] == '<GCONSTS>':
            G = Lineas_archivo[i].split('  ')[1].split(' ')
            G = [ x for x in G if x not in  ['[',']\n'] ]
            #G = list(map(float, G))[0]
            
            G = np.asarray(list(map(float, G)))
            GCONSTS.append(G)
            numgauss = len(G)
        elif Parametro[0] == '<MEANS_INVVARS>':
            
            for k in range(numgauss):                
                Media_inv = Lineas_archivo[i+1+k].split(' ')
                Media_inv = [ x for x in Media_inv if x not in  ['','\n',']\n'] ]

                Media_inv =  np.asarray(list(map(float, Media_inv)))
                MEANS_INVVARS.append(Media_inv)
                
            
        elif Parametro[0] == '<INV_VARS>':
            
            for k in range(numgauss):
                In_Var = Lineas_archivo[i+1+k].split(' ')
                In_Var = [ x for x in In_Var if x not in  ['','\n',']\n'] ]
                In_Var =  np.asarray(list(map(float, In_Var)))
                INV_VARS.append(In_Var)


    return GCONSTS, MEANS_INVVARS, INV_VARS, N_Estados