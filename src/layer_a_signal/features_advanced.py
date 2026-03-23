"""
Layer A — Signal Lab : Advanced Signal Cleaning v2
Algorithmes robustes pour données 1Hz bruitées (Savitzky-Golay, Z-score, etc.)

Features:
  - detect_anomalies()         : Z-score glissant pour détecter spikes
  - smooth_savgol()            : Savitzky-Golay (meilleur que median pour séries temps)
  - advanced_interpolation()   : Interpolation smart (limites physiologiques)
  - flag_artifacts()           : Marquer les segments nettoyés pour traçabilité
  - clean_signal_v2()          : Pipeline complet
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from scipy import signal as scipy_signal


def load_params(config_path: str | Path | None = None) -> dict:
    """Charge les paramètres depuis params.yaml."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / 'config' / 'params.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


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
        print(f"  ✓ Savitzky-Golay: HR, speed, cadence")

    # ─── STEP 5: Recalcul dérivées ──────────────────────────────
    if 'speed_ms' in df.columns:
        df['speed_kmh'] = df['speed_ms'] * 3.6
        df['pace_min_km'] = np.where(
            df['speed_ms'] > 0.5,
            1000 / (df['speed_ms'] * 60),
            np.nan
        )

    if verbose:
        print(f"  ✓ Dérivées recalculées (speed_kmh, pace_min_km)")
        print(f"  ✅ Nettoyage v2 complet!")

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
