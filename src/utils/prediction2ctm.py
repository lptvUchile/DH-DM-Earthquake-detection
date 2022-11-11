
import pandas as pd
import numpy as np
import datetime

def prediction2ctm(path_prediction,start_time):
    
    """
    Function used to convert the format of the results obtained with EqTransformer 
    to the format that we use in our calculation of metrics.

    Args:
        path_prediction (str): path where the "X_prediction_results.csv" file of the predictions obtained by EQTransformer is located
        start_time (str): Initial date with format "yyyy-mm-dd hh:mm:ss.ss" of the trace to be converted

    Return:
        salida (dataframe): contains the information of the events indicating the initial time ("StarSecond_V") 
        and the duration in seconds ("EndSecond_V")
     
    """ 
    
    df = pd.read_csv(path_prediction)

    ind_corte = len(df)
    
    t_start_trace = pd.to_datetime(start_time, infer_datetime_format = True)

    df_segmento = df.iloc[0:ind_corte]
    cols = ['event_start_time', 'event_end_time']
    tiempos_eventos =df_segmento[cols]
    tiempos_eventos = tiempos_eventos[cols].apply(pd.to_datetime, infer_datetime_format = True)

    tiempos_eventos['diff'] = np.array((tiempos_eventos[cols[1]]-tiempos_eventos[cols[0]]).dt.total_seconds(),
                             dtype = int)
    tiempos_eventos['start_set0'] = np.array((tiempos_eventos[cols[0]] - t_start_trace).dt.total_seconds(), 
                                    dtype = int)

    salida = [['2021_CO10', 1, tiempos_eventos['start_set0'][i],tiempos_eventos['diff'][i],str('EVENTO')] 
            for i in range(len(tiempos_eventos))]
    salida = pd.DataFrame(salida)
    salida.columns = ['Name', 'ones', 'StartSecond_V', 'EndSecond_V', 'Label']
    # salida.to_csv(r'EQT_ctm_2021_CO10.ctm', header=None, index=None, sep=' ', mode='a')

    return salida 



