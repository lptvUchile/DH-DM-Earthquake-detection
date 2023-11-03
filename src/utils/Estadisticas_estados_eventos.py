import numpy as np
import itertools
import pandas as pd
from scipy import stats

"""
Calculate the minimum, maximum, means, and variances of state and event durations.
"""

# Define the database and its name
database = 'Train'
name_database = 'NorthChile'

# Load Alineaciones data from a specific file
Alineaciones = np.load('../../data/'+name_database+'/features/Alineaciones_ViterbiForzado_'+name_database+'_'+database+"_DNN.npy", allow_pickle=True)

# Initialize lists for state durations (Silence states)
n_E_1 = []
n_E_2 = []
n_E_3 = []
n_E_4 = []
n_E_5 = []
n_E_6 = []
n_E_7 = []
n_E_8 = []
n_E_9 = []

# Initialize lists for silence durations
n_Sil_1 = []
n_Sil_2 = []
n_Sil_3 = []

# Initialize a DataFrame to store results
df_sil = pd.DataFrame(columns=['utt', 'ini', 'fin', 'Tipo'])

a = 0

# Iterate through Alineaciones
for j in Alineaciones:
    x = j
    indexes = [index for index, _ in enumerate(x) if x[index] != x[index-1]]
    indexes.append(len(x))
    final = [x[indexes[i]:indexes[i+1]] for i, _ in enumerate(indexes) if i != len(indexes)-1]
    
    for k in final:
        if k[0] == [1, 0]:
            n_Sil_1.append(len(k))
        elif k[0] == [1, 1]:
            n_Sil_2.append(len(k))
        elif k[0] == [1, 2]:
            n_Sil_3.append(len(k))
        elif k[0] == [2, 0]:
            n_E_1.append(len(k))
        elif k[0] == [2, 1]:
            n_E_2.append(len(k))
        elif k[0] == [2, 2]:
            n_E_3.append(len(k))
        elif k[0] == [2, 3]:
            n_E_4.append(len(k))
        elif k[0] == [2, 4]:
            n_E_5.append(len(k))
        elif k[0] == [2, 5]:
            n_E_6.append(len(k))
        elif k[0] == [2, 6]:
            n_E_7.append(len(k))
        elif k[0] == [2, 7]:
            n_E_8.append(len(k))
        elif k[0] == [2, 8]:
           n_E_9.append(len(k))

    a += 1

# Print state durations - Silence
print('State Durations - Silence')
print('SIL_1: Minimum: ' + str(np.nanmin(n_Sil_1)) + ', Maximum: ' + str(stats.scoreatpercentile(n_Sil_1, 95)))
print('SIL_2: Minimum: ' + str(np.nanmin(n_Sil_2)) + ', Maximum: ' + str(stats.scoreatpercentile(n_Sil_2, 95)))
print('SIL_3: Minimum: ' + str(np.nanmin(n_Sil_3)) + ', Maximum: ' + str(stats.scoreatpercentile(n_Sil_3, 95)))

# Print state durations - Events
print('State Durations - Events')
print('Ev_1: Minimum: ' + str(np.nanmin(n_E_1)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E_1, 95)))
print('Ev_2: Minimum: ' + str(np.nanmin(n_E_2)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E_2, 95)))
print('Ev_3: Minimum: ' + str(np.nanmin(n_E_3)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E_3, 95)))
print('Ev_4: Minimum: ' + str(np.nanmin(n_E_4)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E_4, 95)))
print('Ev_5: Minimum: ' + str(np.nanmin(n_E_5)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E_5, 95)))
print('Ev_6: Minimum: ' + str(np.nanmin(n_E_6)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E_6, 95)))
print('Ev_7: Minimum: ' + str(np.nanmin(n_E_7)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E_7, 95)))
print('Ev_8: Minimum: ' + str(np.nanmin(n_E_8)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E_8, 95)))
print('Ev_9: Minimum: ' + str(np.nanmin(n_E_9)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E_9, 95)))
print('')


#EVENTOS

# Initialize lists for event and silence durations
n_E = []
n_Sil = []

# Process Alineaciones to create event and silence sequences
for j in Alineaciones:
    # Replace state labels with 's' for silence and 'e' for event
    j = ['s' if x == [1, 0] else x for x in j]
    j = ['s' if x == [1, 1] else x for x in j]
    j = ['s' if x == [1, 2] else x for x in j]
    j = ['e' if x == [2, 0] else x for x in j]
    j = ['e' if x == [2, 1] else x for x in j]
    j = ['e' if x == [2, 2] else x for x in j]
    j = ['e' if x == [2, 3] else x for x in j]
    j = ['e' if x == [2, 4] else x for x in j]
    j = ['e' if x == [2, 5] else x for x in j]
    j = ['e' if x == [2, 6] else x for x in j]
    j = ['e' if x == [2, 7] else x for x in j]
    j = ['e' if x == [2, 8] else x for x in j]

    x = j
    indexes = [index for index, _ in enumerate(x) if x[index] != x[index-1]]
    indexes.append(len(x))
    final = [x[indexes[i]:indexes[i+1]] for i, _ in enumerate(indexes) if i != len(indexes)-1]
    
    for k in final:
        if k[0] == 'e':
            n_E.append(len(k))
        elif k[0] == 's':
            n_Sil.append(len(k))

# Print statistics for event durations
print('Events')
print('Event - Seismic (Ev_sismo)')
print('Minimum: ' + str(np.nanmin(n_E)) + ', Maximum: ' + str(stats.scoreatpercentile(n_E, 95)))
print('Average: ' + str(np.mean(n_E)) + ' +/- ' + str(np.std(n_E)))

# Print statistics for silence durations
print('')
print('Silence Durations (Ev_silencio)')
print('Minimum: ' + str(np.nanmin(n_Sil)) + ', Maximum: ' + str(stats.scoreatpercentile(n_Sil, 95)))
print('Average: ' + str(np.mean(n_Sil)) + ' +/- ' + str(np.std(n_Sil)))
