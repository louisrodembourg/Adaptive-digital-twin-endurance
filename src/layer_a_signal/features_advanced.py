"""
Layer A — Signal Lab : Advanced Signal Cleaning v2 + HRV Tools
Algorithmes robustes pour données 1Hz bruitées (Savitzky-Golay, Z-score, etc.)
et calcul de la variabilité cardiaque (HRV / DFA α₁).

Features:
  - detect_anomalies()         : Z-score glissant pour détecter spikes
  - smooth_savgol()            : Savitzky-Golay (meilleur que median pour séries temps)
  - smart_interpolation()      : Interpolation avec limites physiologiques
  - flag_artifacts()           : Marquer les segments nettoyés pour traçabilité
  - clean_signal_v2()          : Pipeline complet de nettoyage
  - dfa_alpha1_from_rr()       : Calcule DFA α₁ (proxy seuil aérobie)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from ..config import load_params as _shared_load_params


def load_params(config_path: str | Path | None = None) -> dict:
    """Parametres bruts — delegue au chargeur unique et mis en cache."""
    return _shared_load_params(config_path)


# ═══════════════════════════════════════════════════════════════
#  1. DÉTECTION D'ANOMALIES (Z-SCORE GLISSANT)
# ═══════════════════════════════════════════════════════════════

def detect_anomalies(
    series: pd.Series,
    window: int = 5,
    threshold: float = 3.0,
    min_gap: int = 2
) -> pd.Series:
    """
    Détecte les anomalies par z-score glissant.

    Parameters
    ----------
    series : données univariées
    window : fenêtre de calcul de la std (points)
    threshold : z-score threshold (3.0 = très strict)
    min_gap : ignorer gaps nettement < à une valeur (2s = peu probable)

    Returns
    -------
    pd.Series(bool) : True si anomalie détectée

    Example
    -------
    >>> hr = df['hr'].copy()
    >>> anomalies = detect_anomalies(hr, window=5, threshold=3.0)
    >>> print(f"Found {anomalies.sum()} anomalies")
    """
    series = series.fillna(series.mean())  # Remplir NaNs temporairement

    # Rolling z-score
    mean = series.rolling(window=window, center=True, min_periods=1).mean()
    std = series.rolling(window=window, center=True, min_periods=1).std()
    std = std.replace(0, 1)  # Eviter division par 0

    z_score = np.abs((series - mean) / std)
    anomalies = z_score > threshold

    return anomalies


# ═══════════════════════════════════════════════════════════════
#  2. LISSAGE SAVITZKY-GOLAY
# ═══════════════════════════════════════════════════════════════

def smooth_savgol(
    series: pd.Series,
    window: int = 11,
    polyorder: int = 3,
    preserve_nan: bool = True
) -> pd.Series:
    """
    Lisse une série temporelle avec filtre Savitzky-Golay.

    Avantages vs median filter:
      - Pas de déphasage (contrairement à médian)
      - Préserve les gradients
      - Très efficace pour vitesse/cadence

    Parameters
    ----------
    series : données univariées
    window : fenêtre (doit être impair, recommandé 11-21)
    polyorder : ordre polynôme (3 = bonne balance)
    preserve_nan : garder NaNs inchangés après traitement

    Returns
    -------
    pd.Series lissée
    """
    if len(series) < window:
        return series.copy()

    # Tracker les NaNs originaux
    nan_mask = series.isna()

    # Remplir NaNs avec interpolation linéaire pour SavGol
    s_filled = series.bfill().ffill()

    # Appliquer SavGol
    try:
        smoothed = scipy_signal.savgol_filter(
            s_filled,
            window_length=window,
            polyorder=polyorder,
            mode='nearest'  # Gère les bords correctement
        )
        result = pd.Series(smoothed, index=series.index)

        # Restaurer NaNs originaux si demandé
        if preserve_nan:
            result[nan_mask] = np.nan

        return result
    except ValueError:
        # Si window > len(series), fallback
        return series.copy()


# ═══════════════════════════════════════════════════════════════
#  3. INTERPOLATION INTELLIGENTE
# ═══════════════════════════════════════════════════════════════

def smart_interpolation(
    df: pd.DataFrame,
    col: str,
    max_gap: int = 30,
    method: str = 'linear',
    params: dict | None = None
) -> pd.Series:
    """
    Interpole les NaNs avec contraintes physiologiques.

    Parameters
    ----------
    df : DataFrame complet (pour contexte)
    col : colonne à interpoler
    max_gap : gap max acceptable (30s = ~4 HR beats)
    method : 'linear' ou 'cubic'
    params : dict pour limites (hr_min, hr_max, etc.)

    Returns
    -------
    Series interpolée avec limites respectées
    """
    if params is None:
        params = load_params()

    s = df[col].copy()

    # Identifier les gaps
    nan_mask = s.isna()
    if not nan_mask.any():
        return s

    # Interpoler
    s_interp = s.interpolate(method=method, limit=max_gap, fill_value='extrapolate')

    # Appliquer contraintes physiologiques si colonne HR
    if col == 'hr':
        hr_min = params.get('signal', {}).get('hr_min_valid', 40)
        hr_max = params.get('signal', {}).get('hr_max_valid', 210)
        s_interp = s_interp.clip(lower=hr_min, upper=hr_max)

    # Vitesse >= 0
    elif col == 'speed_ms':
        s_interp = s_interp.clip(lower=0)

    return s_interp


# ═══════════════════════════════════════════════════════════════
#  4. FLAGGING DES ARTEFACTS
# ═══════════════════════════════════════════════════════════════

def flag_artifacts(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Ajoute des colonnes de flagging pour traçabilité.

    Retourne df avec colonnes supplémentaires:
      - _flag_hr_jump : True si Δ|HR| > seuil
      - _flag_hr_anomaly : True si z-score détecte anomalie
      - _flag_speed_anomaly : True si speed aberrante
      - _flag_cleaned : Any flag triggeré
    """
    if params is None:
        params = load_params()

    df = df.copy()
    sig = params.get('signal', {})

    # ─── HR Jump Detection ────────────────────────────────────────
    if 'hr' in df.columns:
        hr_jump_threshold = sig.get('hr_jump_threshold_bpm', 5)
        hr_diff = df['hr'].diff().abs()
        df['_flag_hr_jump'] = hr_diff > hr_jump_threshold
    else:
        df['_flag_hr_jump'] = False

    # ─── HR Anomaly (z-score) ───────────────────────────────────
    if 'hr' in df.columns:
        anomalies = detect_anomalies(
            df['hr'],
            window=sig.get('anomaly_window', 5),
            threshold=sig.get('anomaly_threshold', 3.0)
        )
        df['_flag_hr_anomaly'] = anomalies
    else:
        df['_flag_hr_anomaly'] = False

    # ─── Speed Anomaly ───────────────────────────────────────────
    if 'speed_ms' in df.columns:
        # Speed spikes (GPS glitch): > 35 km/h en course = impossible
        speed_max_running = sig.get('speed_max_running_ms', 10)  # ~36 km/h
        df['_flag_speed_anomaly'] = df['speed_ms'] > speed_max_running
    else:
        df['_flag_speed_anomaly'] = False

    # ─── Combined Flag ───────────────────────────────────────────
    df['_flag_cleaned'] = (
        df['_flag_hr_jump'] |
        df['_flag_hr_anomaly'] |
        df['_flag_speed_anomaly']
    )

    return df


# ═══════════════════════════════════════════════════════════════
#  5. PIPELINE COMPLET: CLEAN_SIGNAL_V2
# ═══════════════════════════════════════════════════════════════

def clean_signal_v2(df: pd.DataFrame, params: dict | None = None, verbose: bool = False) -> pd.DataFrame:
    """
    Pipeline complet de nettoyage avancé.

    Steps:
      1. Flagging & détection d'anomalies
      2. Suppression bounds (HR, speed)
      3. Smart interpolation
      4. Savitzky-Goyal smoothing
      5. Vérification contraintes finales

    Parameters
    ----------
    df : DataFrame brut
    params : config (si None, charge depuis params.yaml)
    verbose : affiche étapes

    Returns
    -------
    DataFrame nettoyé avec colonnes de flagging
    """
    if params is None:
        params = load_params()

    df = df.copy()
    sig = params.get('signal', {})

    if verbose:
        print("[clean_signal_v2] Démarrage du nettoyage...")

    # ─── STEP 1: Flagging initial ────────────────────────────────
    df = flag_artifacts(df, params)

    if verbose:
        n_flags = df['_flag_cleaned'].sum()
        pct = 100 * n_flags / len(df) if len(df) > 0 else 0
        print(f"  ✓ Flagging: {n_flags} points marqués ({pct:.1f}%)")

    # ─── STEP 2: Bounds (bounds dur, remplacer par NaN) ──────────
    hr_min = sig.get('hr_min_valid', 40)
    hr_max = sig.get('hr_max_valid', 210)

    if 'hr' in df.columns:
        out_of_bounds = (df['hr'] < hr_min) | (df['hr'] > hr_max)
        df.loc[out_of_bounds, 'hr'] = np.nan

    if 'speed_ms' in df.columns:
        df.loc[df['speed_ms'] < 0, 'speed_ms'] = np.nan

    if verbose:
        print(f"  ✓ Bounds: HR [{hr_min}, {hr_max}] bpm, Speed [0, ∞] m/s")

    # ─── STEP 3: Smart Interpolation ─────────────────────────────
    cols_to_interp = ['hr', 'speed_ms', 'power', 'cadence', 'altitude']
    for col in cols_to_interp:
        if col in df.columns and df[col].isna().any():
            max_gap = sig.get('interp_max_gap_s', 30)
            df[col] = smart_interpolation(df, col, max_gap=max_gap, params=params)

    if verbose:
        print(f"  ✓ Interpolation: {cols_to_interp}")

    # ─── STEP 4: Savitzky-Golay Smoothing ────────────────────────
    cols_to_smooth = {
        'hr': {'window': sig.get('savgol_window_hr', 11), 'polyorder': 3},
        'speed_ms': {'window': sig.get('savgol_window_speed', 11), 'polyorder': 3},
        'cadence': {'window': sig.get('savgol_window_cadence', 15), 'polyorder': 2},
    }

    for col, params_sg in cols_to_smooth.items():
        if col in df.columns and not df[col].isna().all():
            df[col] = smooth_savgol(
                df[col],
                window=params_sg['window'],
                polyorder=params_sg['polyorder']
            )

    if verbose:
        print("  ✓ Savitzky-Golay: HR, speed, cadence")

    # ─── STEP 5: Recalcul dérivées ──────────────────────────────
    if 'speed_ms' in df.columns:
        df['speed_kmh'] = df['speed_ms'] * 3.6
        df['pace_min_km'] = np.where(
            df['speed_ms'] > 0.5,
            1000 / (df['speed_ms'] * 60),
            np.nan
        )

    if verbose:
        print("  ✓ Dérivées recalculées (speed_kmh, pace_min_km)")
        print("  ✅ Nettoyage v2 complet!")

    return df


# ═══════════════════════════════════════════════════════════════
#  6. UTILITY: Report Quality ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

def signal_quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """
    Compare raw vs cleaned pour générer un rapport de qualité.

    Returns
    -------
    dict avec statistiques avant/après
    """
    report = {}

    # HR Stats
    if 'hr' in df_raw.columns:
        report['hr'] = {
            'raw_nans': df_raw['hr'].isna().sum(),
            'clean_nans': df_clean['hr'].isna().sum(),
            'raw_std': df_raw['hr'].std(),
            'clean_std': df_clean['hr'].std(),
            'raw_jumps': (df_raw['hr'].diff().abs() > 5).sum(),
            'clean_jumps': (df_clean['hr'].diff().abs() > 5).sum() if '_flag_hr_jump' in df_clean.columns else 0,
        }

    # Speed Stats
    if 'speed_ms' in df_raw.columns:
        report['speed'] = {
            'raw_nans': df_raw['speed_ms'].isna().sum(),
            'clean_nans': df_clean['speed_ms'].isna().sum(),
            'raw_std': df_raw['speed_ms'].std(),
            'clean_std': df_clean['speed_ms'].std(),
        }

    # Flags
    if '_flag_cleaned' in df_clean.columns:
        report['flags'] = {
            'total_flagged': df_clean['_flag_cleaned'].sum(),
            'flagged_pct': 100 * df_clean['_flag_cleaned'].sum() / len(df_clean),
        }

    return report


# ═══════════════════════════════════════════════════════════════
#  7. HRV — DFA α₁
# ═══════════════════════════════════════════════════════════════
#
# Le generateur de RR synthetiques qui vivait ici a ete deplace dans
# tests/fixtures/synthetic.py. Il produisait un processus AR(1) a graine
# fixe, dont compute_dfa_alpha1 se servait en silence quand les vrais RR
# manquaient — c'est-a-dire toujours. La valeur renvoyee etait un flottant
# ordinaire, indiscernable d'une mesure, et atteignait la soupape de
# securite. Du code de production ne doit pas pouvoir fabriquer une donnee.

def dfa_alpha1_from_rr(rr_ms: np.ndarray, scales: tuple | None = None) -> float:
    """
    Calcule DFA α₁ (Detrended Fluctuation Analysis, court terme).

    DFA α₁ est l'exposant de scaling calculé sur les échelles courtes
    (typiquement 4-16 battements), indicateur de la dynamique fractale
    du rythme cardiaque corrélée au seuil aérobie.

    Interprétation physiologique (Scharhag-Rosenberger et al.) :
      α₁ < 0.75 : zone aérobie (équivalent lactate ~2 mmol/L)
      α₁ ~ 1.0  : zone de transition aérobie/anaérobie
      α₁ > 1.0  : zone anaérobie

    Algorithme :
      1. Intégration de la série RR centrée (marche aléatoire)
      2. Division en fenêtres non chevauchantes de taille n
      3. Détrending linéaire par fenêtre → calcul F(n) = RMS résidus
      4. Régression log(F(n)) vs log(n) → pente = α₁

    Parameters
    ----------
    rr_ms : intervalles RR en ms (array 1D)
    scales : tuple (min_scale, max_scale) en nombre de battements
             Par défaut (4, 16) — court terme standard pour DFA α₁

    Returns
    -------
    float : exposant DFA α₁
            np.nan si calcul impossible (données insuffisantes)
    """
    if scales is None:
        scales = (4, 16)

    min_scale, max_scale = scales
    n_rr = len(rr_ms)

    if n_rr < max_scale * 4:
        return np.nan

    # Centrer la série
    rr_centered = rr_ms - rr_ms.mean()

    # Intégration (profil cumulatif)
    y = np.cumsum(rr_centered)

    fluctuations = []
    scale_values = range(min_scale, max_scale + 1)

    for n in scale_values:
        n_windows = n_rr // n
        if n_windows < 2:
            continue

        # Couper la série en fenêtres de taille n
        y_windowed = y[:n_windows * n].reshape(n_windows, n)

        # Détrending linéaire par fenêtre
        x = np.arange(n)
        f_sq_sum = 0.0
        for w in range(n_windows):
            slope, intercept, _, _, _ = scipy_stats.linregress(x, y_windowed[w])
            trend = intercept + slope * x
            residuals = y_windowed[w] - trend
            f_sq_sum += np.mean(residuals**2)

        f_n = np.sqrt(f_sq_sum / n_windows)
        fluctuations.append((n, f_n))

    if len(fluctuations) < 3:
        return np.nan

    scales_arr = np.array([s for s, _ in fluctuations])
    fluct_arr = np.array([f for _, f in fluctuations])

    # Masque pour éviter log(0)
    valid = fluct_arr > 0
    if valid.sum() < 3:
        return np.nan

    # Régression log-log → pente = α₁
    log_scales = np.log(scales_arr[valid])
    log_fluct = np.log(fluct_arr[valid])
    slope, _, r_value, _, _ = scipy_stats.linregress(log_scales, log_fluct)

    # R² < 0.8 = calcul peu fiable
    if r_value**2 < 0.8:
        return np.nan

    return float(slope)
