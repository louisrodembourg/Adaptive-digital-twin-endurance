"""
Layer A — Signal Lab : Feature Engineering Physiologique
Calcul des biomarqueurs à partir des données 1Hz nettoyées.

Features implémentées :
  - clean_signal()           : Nettoyage des artefacts (HR, vitesse)
  - efficiency_factor()      : EF global = Vitesse / FC
  - specific_ef()            : EF spécifique à l'allure cible (3'40"/km)
  - cardiac_decoupling()     : Dérive cardiaque sur sortie longue
  - compute_trimp()          : Training Impulse (charge interne)
  - compute_hr_zones()       : Temps passé dans chaque zone FC
  - compute_power_metrics()  : Normalized Power, Intensity Factor
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_params(config_path: str | Path | None = None) -> dict:
    """Charge les paramètres depuis params.yaml."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / 'config' / 'params.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
#  1. NETTOYAGE DU SIGNAL
# ═══════════════════════════════════════════════════════════════

def clean_signal(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Nettoie le DataFrame 1Hz :
      - Supprime les FC aberrantes (< 40 ou > 210)
      - Supprime les vitesses négatives
      - Interpole les petits trous (< 5s)
      - Filtre les premières secondes de warm-up GPS (speed=0)
    
    Retourne une copie nettoyée.
    """
    if params is None:
        params = load_params()
    
    sig = params.get('signal', {})
    hr_min = sig.get('hr_min_valid', 40)
    hr_max = sig.get('hr_max_valid', 210)
    
    df = df.copy()
    
    # Artefacts FC
    if 'hr' in df.columns:
        df.loc[(df['hr'] < hr_min) | (df['hr'] > hr_max), 'hr'] = np.nan
    
    # Vitesse négative ou aberrante
    if 'speed_ms' in df.columns:
        df.loc[df['speed_ms'] < 0, 'speed_ms'] = np.nan
        # Recalculer les dérivées
        df['speed_kmh'] = df['speed_ms'] * 3.6
        df['pace_min_km'] = np.where(
            df['speed_ms'] > 0.5,
            1000 / (df['speed_ms'] * 60),
            np.nan
        )
    
    # Interpolation linéaire des petits trous (max 5 points consécutifs)
    cols_to_interp = ['hr', 'speed_ms', 'power', 'cadence', 'altitude']
    for col in cols_to_interp:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=5)
    
    return df


def get_running_mask(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """
    Retourne un masque booléen : True si l'athlète court (speed > seuil).
    Utile pour filtrer les pauses, les arrêts aux feux, etc.
    """
    if params is None:
        params = load_params()
    
    min_speed = params.get('signal', {}).get('speed_min_running_ms', 1.5)
    
    if 'speed_ms' not in df.columns:
        return pd.Series(True, index=df.index)
    
    return df['speed_ms'] >= min_speed


# ═══════════════════════════════════════════════════════════════
#  2. EFFICIENCY FACTOR (EF)
# ═══════════════════════════════════════════════════════════════

def efficiency_factor(df: pd.DataFrame) -> float:
    """
    EF global = Vitesse moyenne (m/s) / FC moyenne (bpm)
    Calculé uniquement sur les segments de course.
    Plus c'est haut, plus l'athlète est efficace.
    """
    mask = get_running_mask(df)
    df_run = df.loc[mask].dropna(subset=['speed_ms', 'hr'])
    
    if df_run.empty or df_run['hr'].mean() == 0:
        return np.nan
    
    return df_run['speed_ms'].mean() / df_run['hr'].mean()


def specific_ef(df: pd.DataFrame, params: dict | None = None) -> float:
    """
    EF Spécifique = Vitesse / FC uniquement sur les segments
    courus à l'allure cible (3'40"/km ± tolérance).
    
    C'est LE marqueur clé pour l'objectif 1h18.
    Si EF_spec monte → tu deviens plus efficace à ton allure de course.
    """
    if params is None:
        params = load_params()
    
    target_speed = params['athlete']['target_speed_ms']       # 4.545 m/s
    tolerance = params['signal']['ef_pace_tolerance_pct'] / 100  # 0.05
    min_duration = params['signal']['ef_min_duration_s']       # 60s
    
    speed_low = target_speed * (1 - tolerance)
    speed_high = target_speed * (1 + tolerance)
    
    if 'speed_ms' not in df.columns or 'hr' not in df.columns:
        return np.nan
    
    # Filtrer les segments à l'allure cible
    mask = (
        (df['speed_ms'] >= speed_low) &
        (df['speed_ms'] <= speed_high) &
        (df['hr'].notna()) &
        (df['speed_ms'].notna())
    )
    
    df_target = df.loc[mask]
    
    if len(df_target) < min_duration:
        return np.nan  # Pas assez de données à cette allure
    
    return df_target['speed_ms'].mean() / df_target['hr'].mean()


# ═══════════════════════════════════════════════════════════════
#  3. CARDIAC DECOUPLING (Durabilité)
# ═══════════════════════════════════════════════════════════════

def cardiac_decoupling(df: pd.DataFrame, params: dict | None = None) -> float:
    """
    Cardiac Decoupling (%) = dérive du ratio EF entre la 1ère et la 2nde moitié.
    
    decoupling = (EF_1st_half - EF_2nd_half) / EF_1st_half * 100
    
    Interprétation :
      - < 5%  : Bonne durabilité aérobie
      - > 5%  : Fatigue / dérive → manque d'endurance à cette intensité
      - > 10% : Alerte forte
    
    Ne calcule que sur les portions où l'athlète court.
    """
    if params is None:
        params = load_params()
    
    mask = get_running_mask(df, params)
    df_run = df.loc[mask].dropna(subset=['speed_ms', 'hr'])
    
    if len(df_run) < 120:  # Minimum 2 min
        return np.nan
    
    split = params.get('signal', {}).get('decoupling_split_ratio', 0.5)
    mid = int(len(df_run) * split)
    
    first_half = df_run.iloc[:mid]
    second_half = df_run.iloc[mid:]
    
    ef_1 = first_half['speed_ms'].mean() / first_half['hr'].mean()
    ef_2 = second_half['speed_ms'].mean() / second_half['hr'].mean()
    
    if ef_1 == 0:
        return np.nan
    
    return (ef_1 - ef_2) / ef_1 * 100


# ═══════════════════════════════════════════════════════════════
#  4. TRIMP (Training Impulse) — Charge Interne
# ═══════════════════════════════════════════════════════════════

def compute_trimp(df: pd.DataFrame, params: dict | None = None) -> float:
    """
    TRIMP de Banister (exponentiel) :
    
    TRIMP = Σ (1/60) × ΔHR_ratio × 0.64 × e^(1.92 × ΔHR_ratio)
    
    où ΔHR_ratio = (HR - HR_rest) / (HR_max - HR_rest)
    
    Unité : charge arbitraire. Plus c'est haut, plus la séance est coûteuse.
    """
    if params is None:
        params = load_params()
    
    hr_max = params['athlete']['hr_max']
    hr_rest = params['athlete']['hr_rest']
    gender_factor = params['signal']['trimp_gender_exp']     # 0.64
    gender_exp = params['signal']['trimp_gender_factor']      # 1.92
    
    if 'hr' not in df.columns:
        return np.nan
    
    hr_series = df['hr'].dropna()
    if hr_series.empty:
        return 0.0
    
    # ΔHR_ratio pour chaque seconde
    delta_hr = (hr_series - hr_rest) / (hr_max - hr_rest)
    delta_hr = delta_hr.clip(0, 1)  # Bornage [0, 1]
    
    # TRIMP = Σ (durée_min × delta × b × e^(c × delta))
    trimp = (1/60) * delta_hr * gender_factor * np.exp(gender_exp * delta_hr)
    
    return trimp.sum()


# ═══════════════════════════════════════════════════════════════
#  5. ZONES FC — Temps dans chaque zone
# ═══════════════════════════════════════════════════════════════

def compute_hr_zones(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Calcul du temps passé (en secondes et %) dans chaque zone FC.
    
    Returns
    -------
    dict avec clés z1..z5, chaque entrée = {'seconds': int, 'pct': float}
    """
    if params is None:
        params = load_params()
    
    hr_max = params['athlete']['hr_max']
    zones_cfg = params['hr_zones']
    
    if 'hr' not in df.columns:
        return {}
    
    hr = df['hr'].dropna()
    total = len(hr)
    if total == 0:
        return {}
    
    result = {}
    for zone_name, (low_pct, high_pct) in zones_cfg.items():
        low_hr = hr_max * low_pct
        high_hr = hr_max * high_pct
        count = ((hr >= low_hr) & (hr < high_hr)).sum()
        result[zone_name] = {
            'seconds': int(count),
            'pct': round(count / total * 100, 1),
            'range_bpm': f"{int(low_hr)}-{int(high_hr)}"
        }
    
    return result


# ═══════════════════════════════════════════════════════════════
#  6. PUISSANCE — Normalized Power & Intensity Factor
# ═══════════════════════════════════════════════════════════════

def normalized_power(df: pd.DataFrame, window: int = 30) -> float:
    """
    Normalized Power (NP) = moyenne glissante 30s de la puissance,
    élevée à la puissance 4, moyennée, puis racine 4ème.
    
    NP = (mean(rolling_30s(power)^4))^(1/4)
    
    Plus représentatif de la charge métabolique que la puissance moyenne.
    """
    if 'power' not in df.columns:
        return np.nan
    
    p = df['power'].dropna()
    if len(p) < window:
        return p.mean() if len(p) > 0 else np.nan
    
    rolling_avg = p.rolling(window=window, min_periods=1).mean()
    
    return (rolling_avg.pow(4).mean()) ** 0.25


# ═══════════════════════════════════════════════════════════════
#  7. RÉSUMÉ COMPLET D'UNE SÉANCE
# ═══════════════════════════════════════════════════════════════

def session_features(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Calcule TOUTES les features d'une séance en un seul appel.
    
    Returns
    -------
    dict avec toutes les métriques calculées.
    """
    if params is None:
        params = load_params()
    
    mask = get_running_mask(df, params)
    duration_running_s = mask.sum()
    
    features = {
        # Métriques de base
        'duration_total_s':     len(df),
        'duration_running_s':   int(duration_running_s),
        'distance_m':           df['distance_m'].max() if 'distance_m' in df.columns else np.nan,
        'hr_mean':              df.loc[mask, 'hr'].mean() if 'hr' in df.columns else np.nan,
        'hr_max':               df.loc[mask, 'hr'].max() if 'hr' in df.columns else np.nan,
        'speed_mean_kmh':       df.loc[mask, 'speed_kmh'].mean() if 'speed_kmh' in df.columns else np.nan,
        'power_mean':           df.loc[mask, 'power'].mean() if 'power' in df.columns else np.nan,
        'cadence_mean':         df.loc[mask, 'cadence'].mean() if 'cadence' in df.columns else np.nan,
        
        # Features avancées
        'ef_global':            efficiency_factor(df),
        'ef_specific':          specific_ef(df, params),
        'cardiac_decoupling':   cardiac_decoupling(df, params),
        'trimp':                compute_trimp(df, params),
        'normalized_power':     normalized_power(df),
        
        # Zones FC
        'hr_zones':             compute_hr_zones(df, params),
    }
    
    return features
