def Escribir_ctm(Ruido_diff, Evento_diff,Nombres_Archivos,Indice,fs):
    # Función que escribe en un archivo de texto los inicios y final de una detección sismica
    
    Nombre_sac = Nombres_Archivos[Indice].split(' ')[0]
    
    if len(Evento_diff[0]) == len(Evento_diff[1]) and len(Ruido_diff[0]) == len(Ruido_diff[1]):
        if len(Evento_diff[0])==0:
            print(Nombre_sac + ' 1 ' + str(0) + ' ' + str(0) + ' EVENTO')
            fs.write(Nombre_sac + ' 1 ' + str(0) + ' ' + str(0) + ' EVENTO\n')
    
    
        for i in range(len(Evento_diff[0])):
            Inicio = 2*Evento_diff[0][i]
            Duracion = (Evento_diff[1][i]-Evento_diff[0][i]+1)*2
            print(Nombre_sac + ' 1 ' + str(Inicio) + ' ' + str(Duracion) + ' EVENTO')
            fs.write(Nombre_sac + ' 1 ' + str(Inicio) + ' ' + str(Duracion) + ' EVENTO\n')

