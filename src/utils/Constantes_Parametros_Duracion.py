import numpy as np

"""
Se calculan los parámetros de una función gamma a partir del promedio y desviación estandar
"""

mu = 11.4 # Promedio
dv =  7.4 # Desviación estandar

alpha = mu/(dv**2)
p = (mu**2)/(dv**2)
def integral_discreta(alpha,p):
    t = 1
    suma = 0
    while True:
        termino = t ** (p - 1) * np.exp(-alpha * t)
        if termino < 1e-100:
            break
        else:
            suma += termino
            t += 1
    return 1/suma

print('')
print('Parametros')
print('alpha: '+str(alpha))
print('p: '+str(p))
print('k: '+str(integral_discreta(alpha,p)))
print('')
print('Insertar en Restricciones_Duracion_Evento_9estados')
print([integral_discreta(alpha,p),alpha,p])
