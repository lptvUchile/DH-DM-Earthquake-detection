import numpy as np

def Sequence_MatrixProb(Alineaciones,Filename):

    Matrix_prob = []
    a=0
    for i in Alineaciones:  
        Matrix_prob.append([])
        for j in range(len(i)):
            Matrix_prob[a].append([])
            if i[j][0] == 2:
                Indice = i[j][1]+3
            else:
                Indice = i[j][1]
            for k in range(12):
                if k == Indice:
                    Matrix_prob[a][j].append(1)
                else:
                    Matrix_prob[a][j].append(0)
        a=a+1
        print(a)



    Matrix_prob_state = np.array([np.array(x) for x in Matrix_prob])
    np.save(Filename+'+.npy', Matrix_prob_state, allow_pickle=True)

                        
