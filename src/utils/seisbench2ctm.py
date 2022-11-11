
import pandas as pd
import numpy as np
from obspy.core import UTCDateTime


def seisbench2ctm(path_prediction,start_time,detection,dif_sta,inicios):
    
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
    
# path_prediction = 'detections/Resultados_CRED_original_LPTV.csv'
# start_time = "2021-07-04 00:00:00.00"
# detection = "event"
# dif_sta = True 
# inicios = 'detections/inicios_CRED_original_lptv.npy'

    df= pd.read_excel(path_prediction)
    ind_corte = len(df)
    
    
    inic = np.load(inicios, allow_pickle=True)    
    ini = pd.DataFrame(inic,columns=['start_time']) # OJO CON EL NUMERO DE COLUMNAS
    # ini = ini['start_time'].apply(pd.to_datetime, infer_datetime_format = True)
    ini = pd.DataFrame(ini)
    
       
    t_start_trace = pd.to_datetime(start_time, infer_datetime_format = True)
    
    
    if detection == "event":
         
        df_segmento = df.iloc[0:ind_corte]
    
        cols = ['event_start_time', 'event_end_time']
        tiempos_eventos = df_segmento[cols]
        # tiempos_eventos = tiempos_eventos[cols].apply(pd.to_datetime, infer_datetime_format = True)
    
        start = []
        end = []
        for i in range(len(df_segmento)):
            star = UTCDateTime(tiempos_eventos['event_start_time'][i])
            en = UTCDateTime(tiempos_eventos['event_end_time'][i])
            
            start.append(star)
            end.append(en)
    
        tiempos_eventos['event_start_time'] = start
        tiempos_eventos['event_end_time'] = end   
        tiempos_eventos['diff'] = np.array((tiempos_eventos[cols[1]]-tiempos_eventos[cols[0]]))
        
        tiempos_eventos['diff'] = np.array((tiempos_eventos[cols[1]]-tiempos_eventos[cols[0]]))
    
        if dif_sta == False:
            tiempos_eventos['start_set0'] = np.array((tiempos_eventos[cols[0]] - t_start_trace).dt.total_seconds(), dtype = int)
            salida = [['2021_CO10', 1, tiempos_eventos['start_set0'][i],tiempos_eventos['diff'][i],str('EVENTO')] 
                      for i in range(len(tiempos_eventos))]
        else:
            tiempos_eventos['start_set0'] = np.array((tiempos_eventos[cols[0]] - ini['start_time']))
            salida = [[df['station'][i], 1, tiempos_eventos['start_set0'][i],tiempos_eventos['diff'][i],str('EVENTO')] 
                    for i in range(len(tiempos_eventos))]
    
        salida = pd.DataFrame(salida)
        salida.columns = ['Name', 'ones', 'StartSecond_V', 'EndSecond_V', 'Label']
        # salida.to_csv(r'EQT_ctm_2021_CO10.ctm', header=None, index=None, sep=' ', mode='a')
    else:
        
        df.columns = ['station','event_start_time','pick']
        tiempos_eventos = df['event_start_time'].apply(pd.to_datetime, infer_datetime_format = True)
        
        salida = []
        j = 1
        for i in range(len(df)):
            if df["pick"][i] == 'P':
                diff=30
                ini = (tiempos_eventos[i] - t_start_trace).total_seconds()
                sal = ['2021_CO10', 1, ini ,diff,str('EVENTO')] 
                
                salida.append(sal)
                
            elif df["pick"][i] == 'S' and i>1:
                  diff = (tiempos_eventos[i] - tiempos_eventos[i-1]).total_seconds()
                  ini = (tiempos_eventos[i] - t_start_trace).total_seconds()
                  sal = ['2021_CO10', 1, ini ,diff,str('EVENTO')]
                  salida[i-(1+j)]=sal 
                  j = j + 1
                
        salida = pd.DataFrame(salida)
        salida.columns = ['Name', 'ones', 'StartSecond_V', 'EndSecond_V', 'Label']
          
    return salida 




