"""
Generateurs de donnees synthetiques, reserves aux tests.

Ce module ne doit jamais etre importe par du code de production : son role est
de fournir des entrees controlees, pas de boucher les trous d'un jeu de
donnees reel. Les fonctions portent toutes `synthetic` dans leur nom pour que
la nature de leur sortie soit lisible sur la ligne d'appel.
"""

from __future__ import annotations

import numpy as np

def synthetic_rr_from_hr(hr_array: np.ndarray, rmssd_ms: float = 35.0,
                     seed: int = 42) -> np.ndarray:
    """
    Génère des intervalles RR synthétiques physiologiquement plausibles
    à partir d'une série de FC (bpm).

    Modèle :
      RR_base(t) = 60000 / HR(t)                    [ms]
      RR(t)      = RR_base(t) + bruit_HRV(t)

    Le bruit HRV est modélisé comme un bruit coloré (1/f) à court terme,
    caractéristique du système nerveux autonome au repos relatif.
    L'amplitude est paramétrisée par rmssd_ms (typiquement 20-60 ms chez
    un coureur entraîné).

    Parameters
    ----------
    hr_array : FC en bpm (array 1D, résolution 1Hz)
    rmssd_ms : RMSSD cible en ms (variabilité inter-battements)
    seed : graine aléatoire pour reproductibilité

    Returns
    -------
    np.ndarray : intervalles RR en ms (longueur ≈ len(hr_array))

    Note
    ----
    Sortie SIMULEE. Un processus autoregressif calibre sur une frequence
    cardiaque ne porte aucune information autonome : son exposant DFA
    decrit le generateur, pas un athlete. Reserve aux tests.
    """
    rng = np.random.default_rng(seed)
    n = len(hr_array)

    # RR de base en ms
    hr_safe = np.where(hr_array > 0, hr_array, 60.0)
    rr_base = 60000.0 / hr_safe

    # Bruit coloré (HRV ~ autocorrélé à court terme)
    # Simulation simplifiée : AR(1) avec coefficient ~ 0.6
    noise = np.zeros(n)
    ar_coeff = 0.6
    scale = rmssd_ms * np.sqrt(1 - ar_coeff**2)
    for i in range(1, n):
        noise[i] = ar_coeff * noise[i-1] + rng.normal(0, scale)

    rr_mock = rr_base + noise

    # Contraintes physiologiques : RR entre 300ms (200 bpm) et 1500ms (40 bpm)
    rr_mock = np.clip(rr_mock, 300, 1500)

    return rr_mock


