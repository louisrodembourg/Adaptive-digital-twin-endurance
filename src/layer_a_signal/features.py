"""
Layer A — Signal Lab : Feature Engineering Physiologique
Calcul des biomarqueurs à partir des données 1Hz nettoyées.

Features implémentées :
  - clean_signal()           : Nettoyage avancé des artefacts [→ v2 par défaut]
  - efficiency_factor()      : EF global = Vitesse / FC
  - specific_ef()            : EF spécifique à l'allure cible (3'40"/km)
  - specific_ef_gap()        : EF spécifique sur vitesse GAP-corrigée
  - grade_adjusted_pace()    : Vitesse corrigée du dénivelé (Jack Daniels)
  - cardiac_decoupling()     : Dérive cardiaque sur sortie longue (warm-up exclu)
  - compute_trimp()          : Training Impulse (charge interne)
  - compute_trimp_strength() : Charge musculaire → TRIMP équivalent
  - compute_hr_zones()       : Temps passé dans chaque zone FC
  - normalized_power()       : Normalized Power (fenêtre 30s)
  - compute_dfa_alpha1()     : DFA α₁ — proxy du seuil aérobie (via RR)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import advanced cleaning (v2) and HRV tools
try:
    from .features_advanced import clean_signal_v2, generate_mock_rr, dfa_alpha1_from_rr
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False


from ..config import load_config, load_params as _shared_load_params


def load_params(config_path: str | Path | None = None) -> dict:
    """Parametres bruts — delegue au chargeur unique et mis en cache."""
    return _shared_load_params(config_path)


# ═══════════════════════════════════════════════════════════════
#  1. NETTOYAGE DU SIGNAL
# ═══════════════════════════════════════════════════════════════

def clean_signal(df: pd.DataFrame, params: dict | None = None, version: str = 'v2') -> pd.DataFrame:
    """
    Nettoie le DataFrame 1Hz.

    Parameters
    ----------
    df : DataFrame brut
    params : configuration (si None, charge depuis params.yaml)
    version : 'v2' (recommandé) ou 'v1' (basique, legacy)

    Returns
    -------
    DataFrame nettoyé avec colonnes optionnelles de flagging (_flag_*)

    Note
    ----
    v2 (par défaut) utilise Savitzky-Golay + z-score:
      - Supprime les spikes optiques
      - Lisse les courbes sans déphasage
      - Ajoute des colonnes de flagging pour traçabilité

    v1 (legacy) utilise juste bounds + interpolation linéaire
    """
    if params is None:
        params = load_params()

    # Utiliser v2 par défaut si disponible
    if version == 'v2' and HAS_ADVANCED:
        return clean_signal_v2(df, params, verbose=False)

    # Fallback v1 (ancienne méthode basique)
    return _clean_signal_v1_legacy(df, params)


def _clean_signal_v1_legacy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    [LEGACY] Nettoyage basique v1.
    Gardé pour compatibilité backward, mais v2 est recommandé.

    Supprime les FC aberrantes (< 40 ou > 210) et interpole.
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
    
    target_speed = load_config().goal.target_speed_ms
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
#  2b. GRADE ADJUSTED PACE (GAP) — Correction Dénivelé
# ═══════════════════════════════════════════════════════════════

def grade_adjusted_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute la colonne `speed_gap_ms` : vitesse corrigée du dénivelé.

    Formule Jack Daniels (empirique) :
      montée : GAP_factor = 1 + grade * 0.033
      descente : GAP_factor = 1 + grade * 0.018  (récupération partielle)

    où grade = Δaltitude / Δdistance (sans unité, ex: 0.05 = 5%)

    Permet de calculer EF_spec sur terrain vallonné de façon équitable.

    Parameters
    ----------
    df : DataFrame nettoyé avec colonnes 'altitude' et 'speed_ms'

    Returns
    -------
    df avec colonne 'speed_gap_ms' ajoutée
    """
    df = df.copy()

    if 'altitude' not in df.columns or 'speed_ms' not in df.columns:
        df['speed_gap_ms'] = df.get('speed_ms', np.nan)
        return df

    # Grade = pente en m/m (Δalt / distance parcourue en 1s)
    d_alt = df['altitude'].diff().fillna(0)
    # À 1Hz : distance ≈ speed_ms * 1s
    speed_safe = df['speed_ms'].clip(lower=0.1)
    grade = d_alt / speed_safe

    # Limiter les pentes aberrantes (capteur GPS bruité)
    grade = grade.clip(-0.30, 0.30)

    # Facteur GAP
    gap_factor = np.where(
        grade >= 0,
        1 + grade * 0.033,   # montée : coût énergétique ↑
        1 + grade * 0.018    # descente : récupération partielle
    )

    df['speed_gap_ms'] = df['speed_ms'] * gap_factor
    df['speed_gap_ms'] = df['speed_gap_ms'].clip(lower=0)

    return df


def specific_ef_gap(df: pd.DataFrame, params: dict | None = None) -> float:
    """
    EF Spécifique GAP = speed_gap_ms / FC uniquement sur les segments
    courus à l'allure cible (3'40"/km) APRÈS correction dénivelé.

    Nécessite que grade_adjusted_pace() ait été appelé au préalable.
    Retourne np.nan si 'speed_gap_ms' absent ou pas assez de données.
    """
    if params is None:
        params = load_params()

    target_speed = load_config().goal.target_speed_ms
    tolerance = params['signal']['ef_pace_tolerance_pct'] / 100
    min_duration = params['signal']['ef_min_duration_s']

    if 'speed_gap_ms' not in df.columns or 'hr' not in df.columns:
        return np.nan

    speed_low = target_speed * (1 - tolerance)
    speed_high = target_speed * (1 + tolerance)

    mask = (
        (df['speed_gap_ms'] >= speed_low) &
        (df['speed_gap_ms'] <= speed_high) &
        df['hr'].notna() &
        df['speed_gap_ms'].notna()
    )
    df_target = df.loc[mask]

    if len(df_target) < min_duration:
        return np.nan

    return df_target['speed_gap_ms'].mean() / df_target['hr'].mean()


# ═══════════════════════════════════════════════════════════════
#  3. CARDIAC DECOUPLING (Durabilité)
# ═══════════════════════════════════════════════════════════════

def cardiac_decoupling(df: pd.DataFrame, params: dict | None = None) -> float:
    """
    Cardiac Decoupling (%) = dérive du ratio EF entre la 1ère et la 2nde moitié.

    decoupling = (EF_1st_half - EF_2nd_half) / EF_1st_half * 100

    Interprétation :
      - < 5%  : Bonne durabilité aérobie
      - 5-10% : Attention, dérive cardiaque
      - > 10% : Alerte forte — endurance insuffisante à cette intensité

    Le warm-up (défaut : 3 min) est exclu du calcul pour éviter les biais
    liés à la montée en température cardiaque.
    Ne calcule que sur les portions où l'athlète court.
    """
    if params is None:
        params = load_params()

    sig = params.get('signal', {})
    mask = get_running_mask(df, params)
    df_run = df.loc[mask].dropna(subset=['speed_ms', 'hr']).reset_index(drop=True)

    # Exclure le warm-up initial (3 min = 180 lignes à 1Hz)
    warmup_s = sig.get('decoupling_warmup_exclude_s', 180)
    if len(df_run) > warmup_s:
        df_run = df_run.iloc[warmup_s:]

    if len(df_run) < 120:  # Minimum 2 min après warm-up
        return np.nan

    split = sig.get('decoupling_split_ratio', 0.5)
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
    
    cfg = load_config()
    hr_max = cfg.athlete.hr_max
    hr_rest = cfg.athlete.hr_rest
    # (b, c) derives du sexe : TRIMP = duree_min * dHR * b * exp(c * dHR).
    # Les deux coefficients viennent d'une seule source pour qu'ils ne
    # puissent plus etre intervertis independamment.
    gender_factor, gender_exp = cfg.athlete.trimp_coeffs
    
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
    
    hr_max = load_config().athlete.hr_max
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
#  5b. TRIMP MUSCULAIRE — Charge force → équivalent TRIMP
# ═══════════════════════════════════════════════════════════════

def compute_trimp_strength(sets: int, reps: int, weight_kg: float,
                           coefficient: float = 0.1) -> float:
    """
    Convertit une séance de musculation en score TRIMP équivalent.

    Formule empirique : TRIMP_strength = sets × reps × weight_kg × coefficient

    Le coefficient par défaut (0.1) est calibré pour qu'une séance type
    (3×10 @ 80kg) → TRIMP ≈ 24 (comparable à 30 min de Z2 léger).

    Parameters
    ----------
    sets : nombre de séries
    reps : répétitions par série
    weight_kg : charge utilisée (kg)
    coefficient : facteur de conversion (à calibrer)

    Returns
    -------
    float : score TRIMP équivalent
    """
    return sets * reps * weight_kg * coefficient


# ═══════════════════════════════════════════════════════════════
#  5c. DFA α₁ — Proxy du Seuil Aérobie
# ═══════════════════════════════════════════════════════════════

def compute_dfa_alpha1(df: pd.DataFrame, params: dict | None = None) -> float:
    """
    Calcule DFA α₁ (Detrended Fluctuation Analysis, exposant court terme).

    C'est un proxy non-invasif du seuil aérobie :
      DFA α₁ < 0.75 → zone aérobie (équivalent seuil lactate ~2 mmol/L)
      DFA α₁ ~ 1.0  → zone de transition
      DFA α₁ > 1.0  → zone anaérobie

    Si les données RR (intervalles battement à battement) ne sont pas
    disponibles dans le DataFrame, génère des intervalles RR synthétiques
    physiologiquement plausibles à partir de la FC (mode mock).

    Parameters
    ----------
    df : DataFrame nettoyé avec colonne 'hr' (et optionnellement 'rr_ms')
    params : configuration (si None, charge depuis params.yaml)

    Returns
    -------
    float : valeur DFA α₁ moyennée sur la séance
            np.nan si insuffisamment de données
    """
    if params is None:
        params = load_params()

    if 'hr' not in df.columns:
        return np.nan

    mask = get_running_mask(df, params)
    hr_series = df.loc[mask, 'hr'].dropna()

    if len(hr_series) < 300:  # Minimum ~5 min pour DFA fiable
        return np.nan

    # Utiliser les vrais RR si disponibles
    if 'rr_ms' in df.columns and df['rr_ms'].notna().sum() > 300:
        rr_series = df.loc[mask, 'rr_ms'].dropna().values
    else:
        # Mode mock : générer RR physiologiquement plausibles depuis HR
        if not HAS_ADVANCED:
            return np.nan
        rr_series = generate_mock_rr(hr_series.values)

    if len(rr_series) < 200:
        return np.nan

    if HAS_ADVANCED:
        return dfa_alpha1_from_rr(rr_series)

    return np.nan


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

    Inclut automatiquement le calcul GAP si la colonne 'altitude' est présente.

    Returns
    -------
    dict avec toutes les métriques calculées.
    """
    if params is None:
        params = load_params()

    # Ajouter GAP si altitude disponible
    if 'altitude' in df.columns:
        df = grade_adjusted_pace(df)

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

        # Efficiency Factor
        'ef_global':            efficiency_factor(df),
        'ef_specific':          specific_ef(df, params),
        'ef_specific_gap':      specific_ef_gap(df, params),

        # Durabilité & Charge
        'cardiac_decoupling':   cardiac_decoupling(df, params),
        'trimp':                compute_trimp(df, params),
        'normalized_power':     normalized_power(df),

        # DFA α₁ (proxy seuil aérobie)
        'dfa_alpha1':           compute_dfa_alpha1(df, params),

        # Zones FC
        'hr_zones':             compute_hr_zones(df, params),
    }

    return features
