from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

def metrics_calculator(viterbi_seismogram, refer_seismogram, umbral):

    """
    Function that defines if the detection is false positive, false negative or true positive.

    Args: 
        viterbi_seismogram (dataframe): Detections of a seismogram made by the algorithm being 
        evaluated
        refer_seismogram (dataframe): Labels of the same seismogram containing viterbi_seismogram
        threshold (int): Thresholds that define what can be considered as false positive, false 
        negative and true positive

    Return:
        n_metric (dataframe): Contains the number of false positives, false negatives, and P-wave 
        error of seismogram detections
        results_FP_FN (dataframe): Contains the detection type classification
     
    """

    #umbral: segundos de holgura para considerar un evento como detectado
    Seismogram = viterbi_seismogram.utt.unique()[0]
    v = viterbi_seismogram.copy()

    r = refer_seismogram.copy()
    v['common_idx'] = None
    r['common_idx'] = None
    v['tipo'] = None
    r['tipo'] = None
    
    #DM tiene las distancias de todos con todos
    DM = pairwise_distances(v.ini.to_numpy().reshape(-1,1), r.StartSecond.to_numpy().reshape(-1,1))
    #verdaderos positivos
    common = 0
    while True:
        #buscamos el minimo
        minimo = np.where(DM == DM.min().min())
        min_coors = list(zip(minimo[0], minimo[1]))[0]
        
        #vemos si el minimo esta dentro del umbral de deteccion
        if DM[min_coors[0], min_coors[1]] < umbral:            
            indice_viterbi = min_coors[0]
            indice_refer = min_coors[1]
            v.loc[v.index[indice_viterbi], 'common_idx'] = common
            r.loc[r.index[indice_refer], 'common_idx'] = common
            
            v.loc[v.index[indice_viterbi], 'tipo'] = 'VP'        
            r.loc[r.index[indice_refer], 'tipo'] = 'VP'
            
            common += 1
            DM[min_coors[0],:] = np.inf
            DM[:,min_coors[1]] = np.inf
        else:
            break
    
    v.tipo = v.tipo.fillna('FP')
    r.tipo = r.tipo.fillna('FN')
    FP = (v.tipo == 'FP').sum()
    FN = (r.tipo == 'FN').sum()


    #calculamos error de P entre los VP. Hacemos merge
    vps = pd.merge(v[v.tipo == 'VP'], r[r.tipo == 'VP'], on='common_idx', how='inner')
    vps['fin_x'] = vps['StartSecond'] + vps['EndSecond'] 
    error_p = abs(vps['ini'] - vps['StartSecond']).mean()
    #mag= r.magnitud.iloc[0]
    #Dist = r.distancia.iloc[0]

    FP_FN = list(v.tipo) + list(r.tipo[r.tipo == 'FN'])
    Ini = list(v['ini']) + list(r['StartSecond'][r.tipo == 'FN'])
    Fin = list(v['fin']) + list(r['EndSecond'][r.tipo == 'FN'])


    #results_FP_FN = pd.DataFrame({'Name':Seismogram , 'StartSecond':Ini ,'EndSecond':Fin, 'FP_FN':FP_FN,'distancia':Dist,'magnitud':mag})
    results_FP_FN = pd.DataFrame({'Name':Seismogram , 'StartSecond':Ini ,'EndSecond':Fin, 'FP_FN':FP_FN})
    n_metric = pd.Series({'Name': Seismogram , 'FN':FN,'FP':FP,'Error_P':error_p,})

    return n_metric, results_FP_FN

	 	

