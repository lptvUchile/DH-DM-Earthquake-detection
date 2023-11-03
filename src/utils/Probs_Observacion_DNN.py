import numpy as np

def Probs_Observacion_DNN(File, pdf):
    """
    Function that creates an observation probability matrix from data obtained using DNN.

    Args:
        File (file): The file containing the data obtained from the DNN.
        pdf (list): A list of lists specifying the indices of silence and event probabilities.

    Returns:
        list: A list of observation probability matrices based on the provided data and pdf indices.
    """
    Contenido_Archivo = File.readlines()
    Probs = []
    c = -1

    for i in Contenido_Archivo:
        e = i.split(' ')
        if len(e) == 3:
            Probs.append([])
            c = c + 1
        else:
            filas = i.split(' ')
            columns = filas[2:-1]
            evento = []
            silencio = []
            for j in pdf[0]:
                for i in range(len(columns)):
                    if i == j:
                        silencio.append(float(columns[i]))
            for k in pdf[1]:
                for i in range(len(columns)):
                    if i == k:
                        evento.append(float(columns[i]))
            Probs[c].append([silencio, evento])

    return Probs
