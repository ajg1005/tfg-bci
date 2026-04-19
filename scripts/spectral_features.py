import numpy as np
import pandas as pd
from scipy.signal import welch


EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def bandpower(freqs, psd, fmin, fmax):
    """
    Potencia de una banda integrando la PSD entre fmin y fmax.
    """
    idx = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(idx):
        return 0.0

    return np.trapezoid(psd[idx], freqs[idx])


def spectral_entropy(psd):
    """
    Entropía espectral a partir de la PSD normalizada.
    """
    psd = np.asarray(psd, dtype=float)

    psd_sum = np.sum(psd)

    if not np.isfinite(psd_sum) or psd_sum <= 0:
        return 0.0
    
    psd_norm = psd / psd_sum
    psd_norm = psd_norm[psd_norm > 0]
    spectral_ent=-np.sum(psd_norm * np.log2(psd_norm))

    return spectral_ent


def mean_frequency(freqs, psd, fmin=None, fmax=None):
    """
    Calcula la frecuencia media ponderada por la PSD en un rango dado.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    if fmin is not None and fmax is not None:
        idx = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[idx]
        psd = psd[idx]

    if len(freqs) == 0 or np.sum(psd) <= 0:
        return 0.0
    
    mean_freq=np.sum(freqs * psd) / np.sum(psd)

    return mean_freq


def extract_spectral_features(
    X_epochs,
    channel_names,
    sfreq=128,
    bands=EEG_BANDS,
    nperseg=128,
):
    """
    Extraer features espectrales por epoch y canal:
    - Potencia absoluta por banda
    - Potencia relativa por banda
    - Entropía espectral global
    - Frecuencia beta media en O1 y O2
    - Ratio theta/beta
    """

    rows = []

    for epoch in X_epochs:
        row = {}

        for ch_idx, ch_name in enumerate(channel_names):
            signal = epoch[:, ch_idx]

            freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)

            # Rango útil total
            total_idx = (freqs >= 0.5) & (freqs <= 45)
            total_psd = psd[total_idx]

            total_power = bandpower(freqs, psd, 0.5, 45)

            # Entropía espectral global del canal
            row[f"{ch_name}_spectral_entropy"] = spectral_entropy(total_psd)

            band_powers = {}

            for band_name, (fmin, fmax) in bands.items():
                power = bandpower(freqs, psd, fmin, fmax)
                band_powers[band_name] = power

                # Potencia absoluta
                row[f"{ch_name}_{band_name}_abs_power"] = power

                # Potencia relativa
                if total_power > 0:
                    row[f"{ch_name}_{band_name}_rel_power"] = power / total_power
                else:
                    row[f"{ch_name}_{band_name}_rel_power"] = 0.0

            # Ratios
            theta = band_powers["theta"]
            beta = band_powers["beta"]

            row[f"{ch_name}_theta_beta_ratio"] = theta / beta if beta > 0 else 0.0
            

            # Frecuencia beta media O1 y O2
            if ch_name in {"O1", "O2"}:
                beta_fmin, beta_fmax = bands["beta"]
                row[f"{ch_name}_beta_mean_freq"] = mean_frequency(
                    freqs, psd, beta_fmin, beta_fmax
                )

        rows.append(row)

    return pd.DataFrame(rows)