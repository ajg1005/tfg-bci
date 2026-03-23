import numpy as np
import pandas as pd


def extract_epoch_features(X_epochs, channel_names):

    #Comprobar dimensiones
    if X_epochs.ndim != 3:
        raise ValueError("X_epochs debe tener forma (n_epochs, epoch_size, n_channels)")

    n_epochs, _, n_channels = X_epochs.shape

    if len(channel_names) != n_channels:
        raise ValueError("El número de canales no coincide con channel_names")

    #Diccionario por ventana 
    rows = []

    # Iterar por epoch guardar las features
    for epoch in X_epochs:
        
        row = {}

        # Iterar por canal
        for i, ch in enumerate(channel_names):
            
            signal = epoch[:, i]

            # Caracteristicas basicas
            mean_val = np.mean(signal)                
            std_val = np.std(signal)                  
            min_val = np.min(signal)                  
            max_val = np.max(signal)                  
            median_val = np.median(signal)            
            var_val = np.var(signal)                  

            # Percentiles
            q25_val = np.percentile(signal, 25)       
            q75_val = np.percentile(signal, 75)       
            iqr_val = q75_val - q25_val               

            # Rango
            range_val = max_val - min_val             

            energy_val = np.sum(signal ** 2)          
            rms_val = np.sqrt(np.mean(signal ** 2))   

            # Guardar features en diccionario con nombres
            row[f"{ch}_mean"] = mean_val
            row[f"{ch}_median"] = median_val
            row[f"{ch}_std"] = std_val
            row[f"{ch}_var"] = var_val
            row[f"{ch}_min"] = min_val
            row[f"{ch}_max"] = max_val
            row[f"{ch}_range"] = range_val
            row[f"{ch}_q25"] = q25_val
            row[f"{ch}_q75"] = q75_val
            row[f"{ch}_iqr"] = iqr_val
            row[f"{ch}_energy"] = energy_val
            row[f"{ch}_rms"] = rms_val

        # Añadir
        rows.append(row)

    # Convertir en dataframe
    return pd.DataFrame(rows)