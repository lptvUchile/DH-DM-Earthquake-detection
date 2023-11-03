import numpy as np

# Mean and standard deviation of the distribution
mu = 11.4
dv = 7.4

# Calculate the parameters alpha and p of the gamma distribution
alpha = mu / (dv ** 2)
p = (mu ** 2) / (dv ** 2)

# Function to calculate the discrete integral of the gamma distribution
def discrete_integral(alpha, p):
    t = 1
    suma = 0
    while True:
        term = t ** (p - 1) * np.exp(-alpha * t)
        if term < 1e-100:
            break
        else:
            suma += term
            t += 1
    return 1 / suma

# Print the calculated parameters
print('')
print('Parameters')
print('alpha: ' + str(alpha))
print('p: ' + str(p))
print('k: ' + str(discrete_integral(alpha, p)))
print('')

# Print the parameters in a specific format for use in Restricciones_Duracion_Evento_9estados
print('Insert into Restricciones_Duracion_Evento_9estados')
print([discrete_integral(alpha, p), alpha, p])
