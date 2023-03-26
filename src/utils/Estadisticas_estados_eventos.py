import numpy as np
import itertools
import pandas as pd
from scipy import stats
"""
Se calcula los minimos, máximos, medias y varianzas de las duraciones de estados y eventos.
"""


database ='Val'
name_database = 'NorthChile'

Alineaciones = np.load("../../data/"+name_database+"/features/Alineaciones_ViterbiForzado_"+database+"_DNN.npy",allow_pickle=True)

n_E_1 = []
n_E_2 = []
n_E_3 = []
n_E_4 = []
n_E_5 = []
n_E_6 = []
n_E_7 = []
n_E_8 = []
n_E_9 = []
n_Sil_1 = []
n_Sil_2 = []
n_Sil_3 = []

df_sil = pd.DataFrame(columns=['utt', 'ini', 'fin','Tipo'])
a=0
b=0
for j in Alineaciones:
    x=j
    indexes = [index for index, _ in enumerate(x) if x[index] != x[index-1]]
    indexes.append(len(x))
    final = [x[indexes[i]:indexes[i+1]] for i, _ in enumerate(indexes) if i != len(indexes)-1]
    
    for k in final:
        if k[0] == [1,0]:
            n_Sil_1.append(len(k))
        elif k[0] == [1,1]:
            n_Sil_2.append(len(k))
        elif k[0] == [1,2]:
            n_Sil_3.append(len(k))
        elif k[0] == [2,0]:
            n_E_1.append(len(k))
        elif k[0] == [2,1]:
            n_E_2.append(len(k))
        elif k[0] == [2,2]:
            n_E_3.append(len(k))
        elif k[0] == [2,3]:
            n_E_4.append(len(k))
        elif k[0] == [2,4]:
            n_E_5.append(len(k))
        elif k[0] == [2,5]:
            n_E_6.append(len(k))
        elif k[0] == [2,6]:
            n_E_7.append(len(k))
        elif k[0] == [2,7]:
            n_E_8.append(len(k))
        elif k[0] == [2,8]:
           n_E_9.append(len(k))

    a=a+1


print('Duraciones estados silencio')
print('SIL_1: Minimmo: '+str(np.nanmin(n_Sil_1))+', Maximo: '+str(stats.scoreatpercentile(n_Sil_1,95)))
print('SIL_2: Minimmo: '+str(np.nanmin(n_Sil_2))+', Maximo: '+str(stats.scoreatpercentile(n_Sil_2,95)))
print('SIL_3: Minimmo: '+str(np.nanmin(n_Sil_3))+', Maximo: '+str(stats.scoreatpercentile(n_Sil_3,95)))
print('Duraciones estados sismos')
print('Ev_1: Minimmo: '+str(np.nanmin(n_E_1))+', Maximo: '+str(stats.scoreatpercentile(n_E_1,95)))
print('Ev_2: Minimmo: '+str(np.nanmin(n_E_2))+', Maximo: '+str(stats.scoreatpercentile(n_E_2,95)))
print('Ev_3: Minimmo: '+str(np.nanmin(n_E_3))+', Maximo: '+str(stats.scoreatpercentile(n_E_3,95)))
print('Ev_4: Minimmo: '+str(np.nanmin(n_E_4))+', Maximo: '+str(stats.scoreatpercentile(n_E_4,95)))
print('Ev_5: Minimmo: '+str(np.nanmin(n_E_5))+', Maximo: '+str(stats.scoreatpercentile(n_E_5,95)))
print('Ev_6: Minimmo: '+str(np.nanmin(n_E_6))+', Maximo: '+str(stats.scoreatpercentile(n_E_6,95)))
print('Ev_7: Minimmo: '+str(np.nanmin(n_E_7))+', Maximo: '+str(stats.scoreatpercentile(n_E_7,95)))
print('Ev_8: Minimmo: '+str(np.nanmin(n_E_8))+', Maximo: '+str(stats.scoreatpercentile(n_E_8,95)))
print('Ev_9: Minimmo: '+str(np.nanmin(n_E_9))+', Maximo: '+str(stats.scoreatpercentile(n_E_9,95)))
print('')

#EVENTOS

n_E = []
n_Sil =[]

for j in Alineaciones:
    j = ['s' if x==[1,0] else x for x in j]
    j = ['s' if x==[1,1] else x for x in j]
    j = ['s' if x==[1,2] else x for x in j]
    j = ['e' if x==[2,0] else x for x in j]
    j = ['e' if x==[2,1] else x for x in j]
    j = ['e' if x==[2,2] else x for x in j]
    j = ['e' if x==[2,3] else x for x in j]
    j = ['e' if x==[2,4] else x for x in j]
    j = ['e' if x==[2,5] else x for x in j]
    j = ['e' if x==[2,6] else x for x in j]
    j = ['e' if x==[2,7] else x for x in j]
    j = ['e' if x==[2,8] else x for x in j]
    x=j
    indexes = [index for index, _ in enumerate(x) if x[index] != x[index-1]]
    indexes.append(len(x))
    final = [x[indexes[i]:indexes[i+1]] for i, _ in enumerate(indexes) if i != len(indexes)-1]
    for k in final:
        if k[0] == 'e':
            n_E.append(len(k))
        elif k[0] == 's':
            n_Sil.append(len(k))


print('Eventos')
print('Ev_sismo')
print('Minimmo: '+str(np.nanmin(n_E))+', Maximo: '+str(stats.scoreatpercentile(n_E,95)))
print('Promedio: '+str(np.mean(n_E))+ ' +- ' +str(np.std(n_E)))
print('')
print('Ev_silencio')
print('Minimo: '+str(np.nanmin(n_Sil))+', Maximo: '+str(stats.scoreatpercentile(n_Sil,95)))
print('Promedio: ' +str(np.mean(n_Sil))+ ' +- ' +str(np.std(n_Sil)))
