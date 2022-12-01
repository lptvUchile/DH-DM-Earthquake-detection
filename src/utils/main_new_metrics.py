#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:10:17 2021

@author: marc
"""


import pandas as pd
import numpy as np
import os
import shutil
from metrics_results import metrics_results
from prediction2ctm import prediction2ctm
from seisbench2ctm import seisbench2ctm
import IPython


def metricas_viterbi(file_viterbi,ref_file_p, nombre_conjunto):
    thresholds = np.array([20]) # Threshold vector
    results = 'results' # output folder path
    # We create the folder where the results will be saved
    path_resultados = os.getcwd() +'/'+results
    """
    if not os.path.exists(path_resultados):    
        os.makedirs(path_resultados) 
    """


    #####################################################################################
    # File paths

    # we read the information from the reference labels
    ref_p = pd.read_excel(ref_file_p)
    ref_p = ref_p[['Name','StartSecond', 'EndSecond','Type']] 
    #ref_p = ref_p[['Name','StartSecond', 'EndSecond','Type','distancia','magnitud']]     
    ref_p_event = ref_p[ref_p['Type'] =='EVENTO']


    #####################################################################################

    viterbi_p = pd.read_csv(file_viterbi, sep=' ', names=['utt', 'unos', 'ini', 'fin', 'label'])


    # We calculate the total of detections by seismogram
    details_ref = []
    for i in viterbi_p.utt.unique():#ref_p_event.Nombre.unique():
        ubic = ref_p_event.loc[ref_p_event['Name'] == i]
        n_events_utt = ubic.shape[0] 
        long_seg_utt = ubic.iloc[-1]['StartSecond'] + ubic.iloc[-1]['EndSecond']
        details_ref.append([n_events_utt,long_seg_utt]) 

#    IPython.embed()
    resume_ref = np.sum(np.array(details_ref),0)
    
        

    # We get results for the different detection thresholds
    for i in range(len(thresholds)):

        #metrics_per_evento = pd.DataFrame(columns = ['score', 'FN','FP','Error_P','distancia','magnitud'])
        metrics_per_evento = pd.DataFrame(columns = ['score', 'FN','FP','Error_P'])

        # We create a results folder for each threshold
        path_umbral = path_resultados+'/umbral_'+str(thresholds[i])
        #if os.path.exists(path_umbral):
        #    shutil.rmtree(path_umbral)
        #os.makedirs(path_umbral)

        # We calculate the detection metrics for each seismogram          
        results1,results2 = metrics_results(viterbi_p, ref_p_event, thresholds[i])
        VP = len(results2['FP_FN'][results2['FP_FN']=='VP'])
        FN = len(results2['FP_FN'][results2['FP_FN']=='FN'])
        FP = len(results2['FP_FN'][results2['FP_FN']=='FP'])      
        prom_result = [FN, FP, results1.Error_P.mean()]
        metrics_per_evento = metrics_per_evento.append({'score':'1', 
                                                        'FN': prom_result[0] / (resume_ref[0]),
                                                        'FP':prom_result[1] / (resume_ref[0]), 
                                                        'Error_P':prom_result[2]}, 
                                                        ignore_index=True)
    

        # We save the results in excel format
        Precision = round(VP/(VP+FP),2)
        Recall = round(VP/(VP+FN),2)
        F1score = round((2*Precision*Recall)/(Recall+Precision),2)
        Metricas = pd.Series({'Precision':Precision, 'Recall':Recall,'F1score':F1score})
        Metricas.to_csv(path_umbral+'/'+ 'score_metricas_'+ nombre_conjunto +'.csv', sep='\t', index=[0,1,2])
        results1.to_csv(path_umbral+'/' + 'score_'+ nombre_conjunto  + '.csv', sep='\t', index = False)
        tfile = open(path_umbral+'/' + 'score_'+'FP_FN_'+nombre_conjunto+'.txt', 'a')
        tfile.write(results2 .to_string(index = False,header = False))
        tfile.close()

        # Contains the percentages of false positives, true positives and error of P from the test database
        metrics_per_evento.to_csv(path_resultados + '/resume_metrics_events_'+ nombre_conjunto +'.csv', sep ='\t', index = False)
        
        



