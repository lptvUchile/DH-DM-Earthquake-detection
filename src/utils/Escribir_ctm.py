def Escribir_ctm(Ruido_diff, Evento_diff, Nombres_Archivos, Indice, fs):
    """
    Function that writes the start and end times of seismic detection into a text file.

    Args:
        Ruido_diff (list): List of differences in noise segments.
        Evento_diff (list): List of differences in event segments.
        Nombres_Archivos (list): List of file names.
        Indice (int): Index for the file name.
        fs (file): File stream for writing the results.

    Returns:
        None
    """

    # Extract the SAC file name from Nombres_Archivos
    Nombre_sac = Nombres_Archivos[Indice].split(' ')[0]

    # Check if the lengths of Evento_diff and Ruido_diff lists are equal and if Evento_diff is empty
    if len(Evento_diff[0]) == len(Evento_diff[1]) and len(Ruido_diff[0]) == len(Ruido_diff[1]):
        if len(Evento_diff[0]) == 0:
            # Print and write information for an event with zero duration
            print(Nombre_sac + ' 1 ' + str(0) + ' ' + str(0) + ' EVENTO')
            fs.write(Nombre_sac + ' 1 ' + str(0) + ' ' + str(0) + ' EVENTO\n')

        # Iterate through event differences to extract start and duration information
        for i in range(len(Evento_diff[0])):
            Inicio = 2 * Evento_diff[0][i]
            Duracion = (Evento_diff[1][i] - Evento_diff[0][i] + 1) * 2
            # Print and write the start and duration of the event
            print(Nombre_sac + ' 1 ' + str(Inicio) + ' ' + str(Duracion) + ' EVENTO')
            fs.write(Nombre_sac + ' 1 ' + str(Inicio) + ' ' + str(Duracion) + ' EVENTO\n')


