import numpy as np
from scipy import signal

def nfft_function(Y):
    # Funcion que calcula la fft
    dimensiones = np.shape(Y)
    ptos=dimensiones[1]
    promedios=np.zeros((dimensiones[0],int(dimensiones[1]/2+1)))
    a=0
    for i in range(0,ptos,2):
        if i == ptos-1:
            promedios[:,a]=Y[:,-1]
        else:
            promedios[:,a]=np.mean([Y[:,i],Y[:,i+1]],axis=0)
        a=a+1
    promedios = np.array(promedios)
    return promedios

def get_frames(signal, frame_length, frame_shift, window=None):
    # Funcion que eventana una señal

    if window is None:
        window=np.hamming(frame_length) 

    L = len(signal)
    N = int(np.fix((L-frame_length)/frame_shift + 1)) #number of frames

    Index = (np.matlib.repmat(np.arange(frame_length),N,1)+np.matlib.repmat(np.expand_dims(np.arange(N)*frame_shift,1),1,frame_length)).T
    hw=np.matlib.repmat(np.expand_dims(window,1),1,N)
    Seg=signal[Index]*hw

    return Seg.T

def parametrizador(senial, frame_length, frame_shift, nfft, escala, window=None):
    # Funcion que parametriza una señal eventanada.

    y = get_frames(senial,frame_length,frame_shift)
    Y = np.abs(np.fft.fft(y, 256, axis=1))
    if escala == 'logaritmica':
        Y = np.log10(Y[:, :int(np.fix(Y.shape[1]/2))+1] )
        Y = nfft_function(Y)
        Y = nfft_function(Y)
    elif escala == 'lineal':
        Y = Y[:, :int(np.fix(Y.shape[1]/2))+1] 
    return Y

def E2(senial, frame_length, frame_shift, escala):
    # Cálculo de energia

    y = get_frames(senial,frame_length,frame_shift)        
    if escala =='logaritmica':
        Y = np.log10(np.sum(y**2,1))
    elif escala == 'lineal':
        Y = np.sum(y**2,1)
    return Y

def E3(senial, frame_length,frame_shift, escala):
    # Cálculo de energia normalizada
    Edos = E2(senial, frame_length,frame_shift, escala)
    if escala == 'logaritmica':
       Salida = Edos-np.max(Edos)
    elif escala == 'lineal':
       Salida = Edos/np.max(Edos)
    return Salida

#Filtrar la señal
def butter_highpass(cutoff, fs, order=3):
    #Filtro pasa banda
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

#filtro causal
def butter_highpass_lfilter(data, cutoff, fs, order=6):
    b, a = butter_highpass(cutoff, fs, order=order)
    #hago padding hacia la izquierda
    len_padding = 20 * order    #numero de muestras que se agregan
    data_ = np.pad(data, (len_padding, 0), 'symmetric', reflect_type='odd') 
    y = signal.lfilter(b, a, data_)
    y = y[len_padding:]   #elimino el padding
    return y

def Delta_selector(feat_mvn, delta_feat, delta_type):
    # Funcion que selecciona el tipo de delta
    if delta_type == 'Delta1':
        delta_feat = Delta1(feat_mvn, 2, "edge")
        delta_delta_feat = Delta1(delta_feat, 2, 'linear_ramp')
    elif delta_type == 'Delta2':
        delta_feat = Delta2(feat_mvn)
        delta_delta_feat = Delta2(delta_feat)
    else:
        raise ValueError('Delta type must be Delta1 or Delta2')
    return delta_feat, delta_delta_feat
     
def Delta1(feat, N, type):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = np.shape(feat)[0]   
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode=type)   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    
    return delta_feat

def Delta2(espectro):
    # Funcion que calcula otra version de delta.
    n_bines = np.shape(espectro)[0]
    n_frames = np.shape(espectro)[1]
    Deltas = np.zeros([n_bines, n_frames])
    for f in range(n_bines):
        for t in range(n_frames):
            if t > 0 and t < (n_frames - 1):
                Deltas[f,t] = (espectro[f, t+1] - espectro[f, t-1])/2
            elif t == 0:
                Deltas[f,t] = (espectro[f, t+1] - espectro[f, t])
            else:
                Deltas[f,t] = (espectro[f, t] - espectro[f, t-1])
    return Deltas

def Contexto(traza):
    # Funcion que genera una matriz con contexto.
    shape = np.shape(traza)
    tam_contexto = 1
    feat_salida_traza = []
    for tiempo in range(shape[0]):
        if tiempo == 0:
            feat_inicio = np.zeros(shape[1]*tam_contexto)  #feat antes del primer tiempo son zeros, definido por MM, hay que corroborarlo
            feat = np.hstack((feat_inicio, np.hstack(traza[tiempo:tiempo+tam_contexto*2]))) #tiempo actual y contexto al futuro         
            feat_salida_traza.append(feat)

        elif tiempo+1 == shape[0]: #ultimo tiempo
             feat_final = np.zeros(shape[1]*tam_contexto) #Tiempo futuro son ceros
             feat = np.hstack((np.hstack(traza[tiempo+1-tam_contexto*2:tiempo+1]), feat_final))        
             feat_salida_traza.append(feat)
             
        else:        
            feat = np.hstack(traza[tiempo-tam_contexto:tiempo+tam_contexto+1])
            feat_salida_traza.append(feat)

    return feat_salida_traza