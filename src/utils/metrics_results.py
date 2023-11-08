import pandas as pd
import numpy as np
from .metrics_calculator import metrics_calculator

    

def metrics_results(viterbi_labels,reference_labels, threshold):

    """
    Function that returns the false positives, false negatives and true positives of all the
    detections of each seismogram

    Args:
        viterbi_labels (dataframe): contains the detections of the algorithm being evaluated.
        reference_labels (dataframe): contains the labels of the reference earthquakes.
        threshold (int): thresholds that define what can be considered as false positive, false 
        negative and true positive

    Return:
        seismogram_results (dataframe): contains the information about the quantities of the type of detection 
        for each seismogram evaluated by the algorithm
        seismogram_results_FP_FN (dataframe): contains the information about the type of detection of 
        all the seismograms evaluated by the algorithm
     
    """    
    seismogram_results = pd.DataFrame(columns = ['Name', 'FN','FP','Error_P'])
    seismogram_results_FP_FN = pd.DataFrame(columns = ['Name', 'StartSecond', 'EndSecond','FP_FN'])
    #seismogram_results_FP_FN = pd.DataFrame(columns = ['Name', 'StartSecond', 'EndSecond','FP_FN','distancia','magnitud'])
 
    #processing for each signal
    for event in viterbi_labels.utt.unique():
        reference_labels['Name'] = reference_labels['Name'].str.strip()

        vi = viterbi_labels[viterbi_labels.utt == event]
      
        re = reference_labels[reference_labels.Name == event]
       

        if re.empty:
            #metr = pd.Series({'Name': event, 'FN':np.NaN, 'FP':np.NaN,'Error_P':np.NaN,'magnitud':np.NaN,'distancia':np.Nan})
            metr = pd.Series({'Name': event, 'FN':np.NaN, 'FP':np.NaN,'Error_P':np.NaN})

        elif vi.empty:
            #metr = pd.Series({'Name': event, 'FN':np.NaN, 'FP':np.NaN,'Error_P':np.NaN, 'magnitud':np.NaN,'distancia':np.Nan})
            metr = pd.Series({'Name': event, 'FN':np.NaN, 'FP':np.NaN,'Error_P':np.NaN})
  
        else:    
            metr = metrics_calculator(vi,re, threshold)
            
        seismogram_results = seismogram_results.append(metr[0], ignore_index=True)
        seismogram_results_FP_FN = seismogram_results_FP_FN.append(metr[1], ignore_index=True)

    if viterbi_labels.utt.unique().size == 0:
        seismogram_results = pd.DataFrame(pd.Series({'Name': np.NaN, 
                                        'FN':np.NaN,
                                        'FP':np.NaN,
                                        'Error_P':np.NaN})).T

 
    index = seismogram_results_FP_FN[seismogram_results_FP_FN["EndSecond"] == 0].index
    seismogram_results_FP_FN = seismogram_results_FP_FN.drop(index) 

    return seismogram_results,seismogram_results_FP_FN
