# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 18:30:49 2020

@author: Jorge
"""

import os
import argparse

import numpy as np
from kaldiio import WriteHelper #instalar via "pip install kaldiio"
from obspy.core.stream import Stream
from obspy import read, read_inventory
from obspy.clients.fdsn import Client
import IPython
import kaldi_io as kio
import pandas as pd
from obspy import read, UTCDateTime
import numpy.matlib
from os import remove

from src.utils.ft_extraction_utils import parametrizador, Contexto, Delta_selector
from src.utils.ft_extraction_utils import energy_mapper


def process_waveform(
    waveform: Stream, 
    channels: str = 'HHE', 
    reduction_factor: int = 5, 
    fs: int = 40
    ):
    """
    Process waveform to extract features
    waveform: obspy.core.stream.Stream
    """
    from scipy import signal
    from src.utils.ft_extraction_utils import butter_highpass_lfilter
    # Process data
    if channels == 'HHE':
        n_secs = len(waveform[0].data) / 100
        wave_upsampled = signal.resample_poly(waveform[0].data, n_secs * 200, len(waveform[0].data))
        wave_resampled = signal.decimate(wave_upsampled, reduction_factor)
        wave_processed = butter_highpass_lfilter(wave_resampled, cutoff=1, fs=fs, order=3)
    else:
        wave_processed = butter_highpass_lfilter(waveform[0].data, cutoff=1, fs=fs, order=3)
    return wave_processed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='NorthChile')
    parser.add_argument('--data', type=str, default='Val')
    parser.add_argument('--data_path', type=str, default='./data')

    parser.add_argument('--nfft', type=int, default=64)
    parser.add_argument('--output_units', type=str, default='VEL')
    parser.add_argument('--Energy', type=str, default='E3')
    parser.add_argument('--escala', type=str, default='logaritmica')
    parser.add_argument('--fs', type=int, default=40)
    parser.add_argument('--delta', type=str, default='Delta1')
    parser.add_argument('--reduction_factor', type=int, default=5)

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

    # Frame params
    frame_length = int(4000*args.fs/1000)
    frame_shift = int(2000*args.fs/1000)

    # Relative Paths
    path_s5 = os.path.join(args.data_path, args.database, 'sac') # Se define la ruta de la base de datos que se quiere analizar. Por 'ejemplo ../../data/NorthChile/sac/'.
    path_results = os.path.join(args.data_path, args.database, 'features') #nombre de la ruta de salida. Por ejemplo '../../data/NorthChile/features/'.
    sac_scp = os.path.join(path_s5, args.data + '.xlsx')
    
    ark_out = os.path.join(path_results + f'raw_mfcc_{args.data}.1.ark')
    scp_out = os.path.join(path_results + f'raw_mfcc_{args.data}.1.scp')


    # Read seismogram database from content in sac_scp
    content = pd.read_excel(sac_scp)
    keys = content['name'].dropna()
        
    # Static feature calculations
    with WriteHelper('ark,scp:{:s},{:s}'.format(ark_out, scp_out), compression_method=2) as writer:
        for i in range(len(content)):
            print(i)
        
            channels = [
                content['channel'].iloc[i], 
                content['channel'].iloc[i][0:-1] + 'N', 
                content['channel'].iloc[i][0:-1] + 'Z'
            ]

            waveform_kwargs = {
                'network': content['network'].iloc[i],
                'station': content['station'].iloc[i],
                'location': "*",
                'starttime': UTCDateTime(content['starttime'].iloc[i]),
                'endtime': UTCDateTime(content['endtime'].iloc[i])
            }

            # Select service
            service = 'IRIS' if waveform_kwargs['network'] in ('C', 'C1') else 'GFZ'    
            client = Client(service)

            # Fetch waveforms for all channels
            waveforms = [
                client.get_waveforms(channel=channel, **waveform_kwargs)
                for channel in channels
            ]

            # Ensure all waveforms have the same length
            min_length = min(np.shape(waveform)[1] for waveform in waveforms)
            for waveform in waveforms:
                waveform[0].data = waveform[0].data[:min_length]

            # Fetch station inventory once, note that it's the same for all waveforms
            inv = client.get_stations(network=waveform_kwargs['network'], station=waveform_kwargs['station'], level='response')
            inv = client.get_stations()

            # Remove response if required
            if args.output_units != 'CUENTAS':
                for waveform in waveforms:
                    waveform[0].remove_response(inventory=inv, output=args.output_units)

            # Process waveforms
            parametrized_feats = []
            energy_feats = []
            for waveform in waveforms:
                wave_processed = process_waveform(waveform)
                # Parametrized features 
                parametrized_feat = parametrizador(wave_processed, frame_length, frame_shift, args.nfft, args.escala)
                # Energy features
                energy_feat = energy_mapper(args.Energy)(wave_processed, frame_length, frame_shift, args.escala)
                # Store features
                parametrized_feats.append(parametrized_feat)
                energy_feats.append(energy_feat)

            # Concatenate features
            parametrized_feats = np.hstack(parametrized_feats) 
            energy_feats = np.vstack(energy_feats)
            all_feats = np.hstack((parametrized_feats, energy_feats.T))	

            # Write features
            writer(keys[i], all_feats)

    ark_out = os.path.join(path_results, f'raw_mfcc_{args.data}.1.ark')
    scp_out = os.path.join(path_results, f'raw_mfcc_{args.data}.1.scp')
    ark_in = os.path.join(path_results, f'raw_mfcc_{args.data}.1.ark')

    utt_base_raw, raw_features = [], []
    for key, mat in kio.read_mat_ark(ark_in):
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
