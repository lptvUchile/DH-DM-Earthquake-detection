# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 18:30:49 2020

@author: Jorge
"""

import numpy as np
import os
from kaldiio import WriteHelper #instalar via "pip install kaldiio"
from obspy import read, read_inventory
from scipy import signal
from scipy import signal
from obspy.clients.fdsn import Client
import IPython
import kaldi_io as kio
import pandas as pd
from obspy import read, UTCDateTime
import numpy.matlib
from os import remove

# Configuracion
nfft = 64
output_units = 'VEL'
Energy = 'E3'
escala = 'logaritmica'
fs = 40
frame_length=int(4000*fs/1000)
frame_shift=int(2000*fs/1000)
delta = 'Delta1'

database = "NorthChile" # Nombre de la base de datos que se quiere analizar
data = "Val" #Particion de la base de datos que se quiere extrae las caracteristicas. Por ejemplo 'Train', 'Val' o 'Test'.

data_path = "./data"

path_s5 = os.path.join(data_path, database, 'sac') # Se define la ruta de la base de datos que se quiere analizar. Por 'ejemplo ../../data/NorthChile/sac/'.
path_results = os.path.join(data_path, database, 'features') #nombre de la ruta de salida. Por ejemplo '../../data/NorthChile/features/'.
sac_scp = os.path.join(path_s5, data + '.xlsx')


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

def parametrizador(senial, frame_length, frame_shift, nfft, escala,window=None):
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


def E2(senial, frame_length,frame_shift,escala):
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
     
    
def Delta1(feat, N,type):
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



ark_out = os.path.join(path_results + 'raw_mfcc_'+data+'.1.ark')
scp_out = os.path.join(path_results + 'raw_mfcc_'+data+'.1.scp')


#leemos listado de sismogramas
content = pd.read_excel(sac_scp)
keys = content['name'].dropna()

    
# Se calculan los features estaticos
with WriteHelper('ark,scp:{:s},{:s}'.format(ark_out, scp_out), compression_method=2) as writer:
    for i in range(len(content)):
        print(i)
        network = content['network'].iloc[i]
        if  network == 'C' or  network == 'C1':
            service = 'IRIS'
        else:
            service = 'GFZ'
        locations = '*'   #todos los locaciones disponibles para esa estacion
        formato = 'SAC'
        station = content['station'].iloc[i]
        client = Client(service)
        channels = content['channel'].iloc[i]
        ini_time = UTCDateTime(content['starttime'].iloc[i])
        end_time = UTCDateTime(content['endtime'].iloc[i])
        read_sac1 = client.get_waveforms(network, station, locations, channels,ini_time,end_time )     

        canal_N = channels[0:-1]+ 'N'
        canal_Z = channels[0:-1]+ 'Z'
        read_sac2 = client.get_waveforms(network, station, locations, canal_N ,ini_time,end_time )     
        read_sac3 = client.get_waveforms(network, station, locations, canal_Z ,ini_time,end_time )     
 


        if np.shape(read_sac1)[1] != np.shape(read_sac2)[1] or np.shape(read_sac1)[1] != np.shape(read_sac3)[1] or np.shape(read_sac2)[1] != np.shape(read_sac3)[1]:
            minimo = np.min([np.shape(read_sac1)[1],np.shape(read_sac2)[1],np.shape(read_sac3)[1]])
            read_sac1[0].data = read_sac1[0].data[0:minimo]
            read_sac2[0].data = read_sac2[0].data[0:minimo]
            read_sac3[0].data = read_sac3[0].data[0:minimo]



        if read_sac1[0].stats.network == 'C1' or read_sac1[0].stats.network == 'C':
            rs = Client('IRIS')
        else:
            rs = Client('GFZ')
        inv1 = rs.get_stations(network=read_sac1[0].stats.network, station=station,level='response')
        inv2 = inv1
        inv3 = inv1

        
        if output_units != 'CUENTAS':
              read_sac1[0].remove_response(inventory=inv1, output=output_units)
              read_sac2[0].remove_response(inventory=inv2, output=output_units)
              read_sac3[0].remove_response(inventory=inv3, output=output_units)

        Factor_Reduccion = 5
        if channels == 'HHE':
           Cantidad_segundos = len(read_sac1)/100
           data1_upsample = signal.resample_poly(read_sac1[0].data, Cantidad_segundos*200,len(read_sac1))
           data1_resampleado = signal.decimate(data1_upsample,Factor_Reduccion)
           data1 = butter_highpass_lfilter(data1_resampleado, cutoff=1, fs=fs, order=3)


           Cantidad_segundos = len(read_sac1)/100
           data2_upsample = signal.resample_poly(read_sac2[0].data, Cantidad_segundos*200,len(read_sac1))
           data2_resampleado = signal.decimate(data2_upsample,Factor_Reduccion)
           data2 = butter_highpass_lfilter(data2_resampleado, cutoff=1, fs=fs, order=3)


           Cantidad_segundos = len(read_sac1)/100
           data3_upsample = signal.resample_poly(read_sac3[0].data, Cantidad_segundos*200,len(read_sac1))
           data3_resampleado = signal.decimate(data3_upsample.data,Factor_Reduccion)
           data3 = butter_highpass_lfilter(data3_resampleado, cutoff=1, fs=fs, order=3)

        else:
           data1 = butter_highpass_lfilter(read_sac1[0].data, cutoff=1, fs=fs, order=3)
           data2 = butter_highpass_lfilter(read_sac2[0].data, cutoff=1, fs=fs, order=3)
           data3 = butter_highpass_lfilter(read_sac3[0].data, cutoff=1, fs=fs, order=3)


        feat1 = parametrizador(data1, frame_length, frame_shift, nfft,escala)    
        feat2 = parametrizador(data2, frame_length, frame_shift, nfft,escala)
        feat3 = parametrizador(data3, frame_length, frame_shift, nfft,escala) 
        
        feat = np.hstack([feat1, feat2, feat3]) 
        feat_Energy = np.vstack((E3(data1,frame_length,frame_shift,escala),
                                    E3(data2, frame_length,frame_shift,escala),
                                    E3(data3, frame_length,frame_shift,escala)))         
        feat = np.hstack((feat, feat_Energy.T ))	
        writer(keys[i],feat)
        




ark_out = os.path.join(path_results + 'raw_mfcc_'+data+'.1.ark')
scp_out = os.path.join(path_results + 'raw_mfcc_'+data+'.1.scp')




ark_in = path_results + 'raw_mfcc_'+data+'.1.ark'
utt_base_raw,raw_features = [], []
for key,mat in kio.read_mat_ark(ark_in):
    utt_base_raw.append(key)
    raw_features.append(mat)
print(np.shape(raw_features[0]))


with WriteHelper('ark,scp:{:s},{:s}'.format(ark_out, scp_out), compression_method=2) as writer:
    #  Normalización
    for i in range(np.shape(raw_features)[0]):
        feat_mvn = np.zeros((raw_features[i].shape[0],np.shape(raw_features[i])[1])) 
        for j in range(np.shape(raw_features[i])[1]):
            mean = np.mean(raw_features[i][:,j])  
            std = np.std(raw_features[i][:,j])
            feat_mvn[:,j] = (raw_features[i][:,j]-mean)/std

    

        # Delta y Delta+Delta
        if delta == 'Delta1':
            delta_feat = Delta1(feat_mvn,2,'edge')
            delta_delta_feat = Delta1(delta_feat,2,'linear_ramp')
        if delta == 'Delta2':
            delta_feat = Delta2(feat_mvn)
            delta_delta_feat = Delta2(delta_feat)
            
            

        feat_mvn_delta = np.concatenate((feat_mvn,delta_feat),axis=1)
        feat_mvn_delta2 = np.concatenate((feat_mvn_delta,delta_delta_feat),axis=1)
        writer(keys[i],feat_mvn_delta2)


# Se calcula crea una matriz con contexto
base_contexto = []
for key,mat in kio.read_mat_ark(ark_in):
    utt_base_raw.append(key)
    raw_features.append(mat)
    base_contexto.append(Contexto(mat))#Aqui queda los features+contexto
    
    # Se guarda en un ark los features estaticos + dinamicos
print(np.shape(raw_features[0]))        

                
      
     
feat_context = np.array([np.array(x) for x in base_contexto])
np.save(path_results + 'Features_'+database+'_'+data + '.npy',feat_context)
remove(scp_out)



