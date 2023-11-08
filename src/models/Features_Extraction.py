# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 18:30:49 2020

@author: Jorge
"""

import os
import argparse

import numpy as np
from kaldiio import WriteHelper #instalar via "pip install kaldiio"
from obspy import read, read_inventory
from scipy import signal
from obspy.clients.fdsn import Client
import IPython
import kaldi_io as kio
import pandas as pd
from obspy import read, UTCDateTime
import numpy.matlib
from os import remove

from src.utils.ft_extraction_utils import parametrizador, Contexto, Delta_selector, butter_highpass_lfilter
from src.utils.ft_extraction_utils import E2, E3

energy_mapper = {
    'E2': E2,
    'E3': E3
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nfft', type=int, default=64)
    parser.add_argument('--output_units', type=str, default='VEL')
    parser.add_argument('--Energy', type=str, default='E3')
    parser.add_argument('--escala', type=str, default='logaritmica')
    parser.add_argument('--fs', type=int, default=40)
    parser.add_argument('--delta', type=str, default='Delta1')
    parser.add_argument('--database', type=str, default='NorthChile')
    parser.add_argument('--data', type=str, default='Val')
    parser.add_argument('--data_path', type=str, default='./data')
    # # Configuracion
    # nfft = 64
    # output_units = 'VEL'
    # Energy = 'E3' #? Energy is not used?
    # escala = 'logaritmica'
    # fs = 40
    # delta = 'Delta1'

    # database = "NorthChile" # Nombre de la base de datos que se quiere analizar
    # data = "Val" #Particion de la base de datos que se quiere extrae las caracteristicas. Por ejemplo 'Train', 'Val' o 'Test'.

    # data_path = "./data"

    args = parser.parse_args()

    Energy_func = energy_mapper[args.Energy]


    frame_length = int(4000*args.fs/1000)
    frame_shift = int(2000*args.fs/1000)

    # Paths
    path_s5 = os.path.join(args.data_path, args.database, 'sac') # Se define la ruta de la base de datos que se quiere analizar. Por 'ejemplo ../../data/NorthChile/sac/'.
    path_results = os.path.join(args.data_path, args.database, 'features') #nombre de la ruta de salida. Por ejemplo '../../data/NorthChile/features/'.
    sac_scp = os.path.join(path_s5, args.data + '.xlsx')

    ark_out = os.path.join(path_results + f'raw_mfcc_{args.data}.1.ark')
    scp_out = os.path.join(path_results + f'raw_mfcc_{args.data}.1.scp')


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

            
            if args.output_units != 'CUENTAS':
                read_sac1[0].remove_response(inventory=inv1, output=args.output_units)
                read_sac2[0].remove_response(inventory=inv2, output=args.output_units)
                read_sac3[0].remove_response(inventory=inv3, output=args.output_units)

            Factor_Reduccion = 5
            if channels == 'HHE':
                Cantidad_segundos = len(read_sac1)/100
                data1_upsample = signal.resample_poly(read_sac1[0].data, Cantidad_segundos*200,len(read_sac1))
                data1_resampleado = signal.decimate(data1_upsample,Factor_Reduccion)
                data1 = butter_highpass_lfilter(data1_resampleado, cutoff=1, fs=args.fs, order=3)


                Cantidad_segundos = len(read_sac1)/100
                data2_upsample = signal.resample_poly(read_sac2[0].data, Cantidad_segundos*200,len(read_sac1))
                data2_resampleado = signal.decimate(data2_upsample,Factor_Reduccion)
                data2 = butter_highpass_lfilter(data2_resampleado, cutoff=1, fs=args.fs, order=3)


                Cantidad_segundos = len(read_sac1)/100
                data3_upsample = signal.resample_poly(read_sac3[0].data, Cantidad_segundos*200,len(read_sac1))
                data3_resampleado = signal.decimate(data3_upsample.data,Factor_Reduccion)
                data3 = butter_highpass_lfilter(data3_resampleado, cutoff=1, fs=args.fs, order=3)

            else:
                data1 = butter_highpass_lfilter(read_sac1[0].data, cutoff=1, fs=args.fs, order=3)
                data2 = butter_highpass_lfilter(read_sac2[0].data, cutoff=1, fs=args.fs, order=3)
                data3 = butter_highpass_lfilter(read_sac3[0].data, cutoff=1, fs=args.fs, order=3)


            feat1 = parametrizador(data1, frame_length, frame_shift, args.nfft, args.escala)    
            feat2 = parametrizador(data2, frame_length, frame_shift, args.nfft, args.escala)
            feat3 = parametrizador(data3, frame_length, frame_shift, args.nfft, args.escala) 
            
            feat = np.hstack([feat1, feat2, feat3]) 
            feat_Energy = np.vstack((Energy_func(data1,frame_length,frame_shift, args.escala),
                                        Energy_func(data2, frame_length,frame_shift, args.escala),
                                        Energy_func(data3, frame_length,frame_shift, args.escala)))         
            feat = np.hstack((feat, feat_Energy.T ))	
            writer(keys[i],feat)
        

    ark_out = os.path.join(path_results + f'raw_mfcc_{args.data}.1.ark')
    scp_out = os.path.join(path_results + f'raw_mfcc_{args.data}.1.scp')
    ark_in = os.path.join(path_results + f'raw_mfcc_{args.data}.1.ark')

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
            delta_feat, delta_delta_feat = Delta_selector(feat_mvn, args.delta)
                
            feat_mvn_delta = np.concatenate((feat_mvn, delta_feat),axis=1)
            feat_mvn_delta2 = np.concatenate((feat_mvn_delta, delta_delta_feat),axis=1)
            writer(keys[i], feat_mvn_delta2)



    # Se calcula crea una matriz con contexto
    base_contexto = []
    for key,mat in kio.read_mat_ark(ark_in):
        utt_base_raw.append(key)
        raw_features.append(mat)
        base_contexto.append(Contexto(mat))#Aqui queda los features+contexto
        
        # Se guarda en un ark los features estaticos + dinamicos
    print(np.shape(raw_features[0]))    

    
    feat_context = np.array([np.array(x) for x in base_contexto])
    np.save(os.path.join(path_results, f'Features_{args.database}_{args.data}.npy'), feat_context)
    remove(scp_out)
