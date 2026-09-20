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
    from .features_advanced import clean_signal_v2, dfa_alpha1_from_rr
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False


from ..config import ConfigError, load_config, load_params as _shared_load_params
from .timebase import active_duration_s, attach_timebase, dt_seconds, longest_contiguous_s


def load_params(config_path: str | Path | None = None) -> dict:
    """Parametres bruts — delegue au chargeur unique et mis en cache."""
    return _shared_load_params(config_path)


# ═══════════════════════════════════════════════════════════════
#  1. NETTOYAGE DU SIGNAL
# ═══════════════════════════════════════════════════════════════

def _target_speed_or_nan(cfg=None) -> float:
    """
    Allure cible, ou NaN tant que l'objectif n'est pas calibre.

    EF specifique se mesure par rapport a une allure de course, qui se derive
    elle-meme de la courbe de reference de l'athlete — laquelle s'ajuste sur
    les seances. A la premiere ingestion cette allure n'existe donc pas
    encore. Une metrique indisponible vaut NaN ; elle n'interrompt pas la
    chaine de traitement.
    """
    try:
        return (cfg or load_config()).goal.target_speed_ms
    except ConfigError:
        return np.nan


def clean_signal(df: pd.DataFrame, params: dict | None = None,
                 version: str = 'v2') -> pd.DataFrame:
    """
    Nettoie le DataFrame 1Hz.

    Parameters
    ----------
    df : DataFrame brut
    params : configuration (si None, charge depuis params.yaml)
    version : conserve pour compatibilite d'appel ; seul 'v2' existe desormais

    Returns
    -------
    DataFrame nettoyé avec colonnes optionnelles de flagging (_flag_*)

    Note
    ----
    Savitzky-Golay plus detection d'anomalies par z-score : les pics optiques
    partent, les gradients physiologiques restent, et des colonnes `_flag_*`
    tracent ce qui a ete touche.

    La base de temps (`dt_s`) est attachee ici, une fois, pour que toutes les
    metriques en aval raisonnent en secondes reelles plutot qu'en lignes.
    """
    if params is None:
        params = load_params()
    if not HAS_ADVANCED:
        raise ImportError("features_advanced est requis pour le nettoyage du signal.")

    cleaned = clean_signal_v2(df, params, verbose=False)
    return attach_timebase(cleaned, params.get('signal', {}).get('max_gap_s', 30.0))


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


def specific_ef(df: pd.DataFrame, params: dict | None = None, cfg=None) -> float:
    """
    EF Spécifique = Vitesse / FC uniquement sur les segments
    courus à l'allure cible (3'40"/km ± tolérance).
    
    C'est LE marqueur clé pour l'objectif 1h18.
    Si EF_spec monte → tu deviens plus efficace à ton allure de course.
    """
    if params is None:
        params = load_params()
    
    target_speed = _target_speed_or_nan(cfg)
    if not np.isfinite(target_speed):
        return np.nan
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
    
    dt = dt_seconds(df, params.get('signal', {}).get('max_gap_s', 30.0))

    # Le seuil porte sur la plus longue tenue continue a l'allure cible, pas
    # sur un total d'echantillons : soixante secondes eparpillees sur une
    # heure et demie ne demontrent pas la meme chose qu'une minute soutenue,
    # et c'est precisement la capacite a tenir que cette metrique mesure.
    if longest_contiguous_s(mask, dt) < min_duration:
        return np.nan

    df_target = df.loc[mask]
    w = dt.loc[mask]
    if w.sum() <= 0:
        return np.nan
    hr = float(np.average(df_target['hr'], weights=w))
    return float(np.average(df_target['speed_ms'], weights=w)) / hr if hr else np.nan


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

    cfg = load_config()
    sig = cfg.signal
    max_grade_pct = sig.get('gap_max_grade_pct', 25.0)
    coef_up = sig.get('gap_coef_uphill_per_pct', 0.033)
    coef_down = sig.get('gap_coef_downhill_per_pct', 0.018)

    # Distance parcourue par echantillon : vitesse x pas de temps reel.
    # L'ancienne version supposait 1 Hz, ce qui gonflait artificiellement la
    # pente des que la serie comportait un trou.
    dt = dt_seconds(df, sig.get('max_gap_s', 30.0))
    d_alt = df['altitude'].diff().fillna(0)
    distance = (df['speed_ms'].clip(lower=0.1) * dt).clip(lower=0.1)

    # Pente EN POURCENTS. Les coefficients de Daniels s'appliquent par point
    # de pourcentage : les utiliser sur une pente exprimee en fraction
    # sous-corrigeait d'un facteur cent, et rendait le GAP indiscernable de
    # la vitesse brute.
    grade_pct = (d_alt / distance) * 100.0
    grade_pct = grade_pct.clip(-max_grade_pct, max_grade_pct)

    gap_factor = np.where(
        grade_pct >= 0,
        1 + grade_pct * coef_up,     # montee : cout energetique superieur
        1 + grade_pct * coef_down,   # descente : allegement partiel
    )

    df['speed_gap_ms'] = df['speed_ms'] * gap_factor
    df['speed_gap_ms'] = df['speed_gap_ms'].clip(lower=0)

    return df


def specific_ef_gap(df: pd.DataFrame, params: dict | None = None, cfg=None) -> float:
    """
    EF Spécifique GAP = speed_gap_ms / FC uniquement sur les segments
    courus à l'allure cible (3'40"/km) APRÈS correction dénivelé.

    Nécessite que grade_adjusted_pace() ait été appelé au préalable.
    Retourne np.nan si 'speed_gap_ms' absent ou pas assez de données.
    """
    if params is None:
        params = load_params()

    target_speed = _target_speed_or_nan(cfg)
    if not np.isfinite(target_speed):
        return np.nan
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

    dt = dt_seconds(df, params.get('signal', {}).get('max_gap_s', 30.0))
    if longest_contiguous_s(mask, dt) < min_duration:
        return np.nan

    df_target = df.loc[mask]
    w = dt.loc[mask]
    if w.sum() <= 0:
        return np.nan
    hr = float(np.average(df_target['hr'], weights=w))
    return float(np.average(df_target['speed_gap_ms'], weights=w)) / hr if hr else np.nan


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
    max_gap = sig.get('max_gap_s', 30.0)
    mask = get_running_mask(df, params)

    work = df.assign(**{'dt_s': dt_seconds(df, max_gap)})
    df_run = work.loc[mask].dropna(subset=['speed_ms', 'hr']).reset_index(drop=True)
    if df_run.empty:
        return np.nan

    # Le decoupage se fait sur le temps ecoule, pas sur le numero de ligne :
    # une pause de dix minutes deplacerait sinon la mi-course.
    elapsed = df_run['dt_s'].cumsum()
    total_s = float(elapsed.iloc[-1])

    warmup_s = sig.get('decoupling_warmup_exclude_s', 180)
    if total_s > warmup_s:
        df_run = df_run.loc[elapsed > warmup_s].reset_index(drop=True)
        elapsed = df_run['dt_s'].cumsum()
        total_s = float(elapsed.iloc[-1]) if len(df_run) else 0.0

    if total_s < 120:  # moins de deux minutes utiles apres echauffement
        return np.nan

    split = sig.get('decoupling_split_ratio', 0.5)
    cut = total_s * split
    first_half = df_run.loc[elapsed <= cut]
    second_half = df_run.loc[elapsed > cut]
    if first_half.empty or second_half.empty:
        return np.nan

    ef_1 = _weighted_ef(first_half)
    ef_2 = _weighted_ef(second_half)
    if not ef_1:
        return np.nan

    return (ef_1 - ef_2) / ef_1 * 100


def _weighted_ef(part: pd.DataFrame) -> float:
    """EF moyen d'un segment, chaque echantillon pesant sa propre duree."""
    w = part['dt_s']
    if w.sum() <= 0:
        return 0.0
    speed = float(np.average(part['speed_ms'], weights=w))
    hr = float(np.average(part['hr'], weights=w))
    return speed / hr if hr else 0.0


# ═══════════════════════════════════════════════════════════════
#  4. TRIMP (Training Impulse) — Charge Interne
# ═══════════════════════════════════════════════════════════════

def compute_trimp(df: pd.DataFrame, params: dict | None = None, cfg=None) -> float:
    """
    TRIMP de Banister (exponentiel) :
    
    TRIMP = Σ (1/60) × ΔHR_ratio × 0.64 × e^(1.92 × ΔHR_ratio)
    
    où ΔHR_ratio = (HR - HR_rest) / (HR_max - HR_rest)
    
    Unité : charge arbitraire. Plus c'est haut, plus la séance est coûteuse.
    """
    if params is None:
        params = load_params()
    
    cfg = cfg or load_config()
    # hr_reserve porte la garde de calibration : sans FC max ni FC repos le
    # TRIMP n'a pas de sens, et mieux vaut le dire que produire un nombre.
    cfg.athlete.hr_reserve
    hr_max, hr_rest = cfg.athlete.hr_max, cfg.athlete.hr_rest
    # (b, c) derives du sexe : TRIMP = duree_min * dHR * b * exp(c * dHR).
    # Les deux coefficients viennent d'une seule source pour qu'ils ne
    # puissent plus etre intervertis independamment.
    gender_factor, gender_exp = cfg.athlete.trimp_coeffs
    
    if 'hr' not in df.columns:
        return np.nan
    
    hr_series = df['hr'].dropna()
    if hr_series.empty:
        return 0.0
    
    delta_hr = ((hr_series - hr_rest) / (hr_max - hr_rest)).clip(0, 1)

    # Chaque echantillon pese sa duree reelle. L'ancienne version comptait
    # 1/60 de minute par ligne, ce qui revient a supposer un echantillonnage
    # a 1 Hz sans trou et sous-estime la charge des que ce n'est pas le cas.
    dt_min = dt_seconds(df, cfg.signal.get('max_gap_s', 30.0)).reindex(
        hr_series.index).fillna(0.0) / 60.0

    trimp = dt_min * delta_hr * gender_factor * np.exp(gender_exp * delta_hr)
    return float(trimp.sum())


# ═══════════════════════════════════════════════════════════════
#  5. ZONES FC — Temps dans chaque zone
# ═══════════════════════════════════════════════════════════════

def compute_hr_zones(df: pd.DataFrame, params: dict | None = None, cfg=None) -> dict:
    """
    Calcul du temps passé (en secondes et %) dans chaque zone FC.
    
    Returns
    -------
    dict avec clés z1..z5, chaque entrée = {'seconds': int, 'pct': float}
    """
    if params is None:
        params = load_params()
    
    hr_max = (cfg or load_config()).athlete.hr_max
    zones_cfg = params['hr_zones']
    
    if 'hr' not in df.columns:
        return {}
    
    hr = df['hr'].dropna()
    if hr.empty:
        return {}

    # Secondes reellement passees dans la zone, et non nombre de lignes.
    dt = dt_seconds(df, params.get('signal', {}).get('max_gap_s', 30.0)).reindex(
        hr.index).fillna(0.0)
    total = float(dt.sum())
    if total <= 0:
        return {}

    result = {}
    for zone_name, (low_pct, high_pct) in zones_cfg.items():
        low_hr, high_hr = hr_max * low_pct, hr_max * high_pct
        seconds = float(dt.where((hr >= low_hr) & (hr < high_hr), 0.0).sum())
        result[zone_name] = {
            'seconds': int(round(seconds)),
            'pct': round(seconds / total * 100, 1),
            'range_bpm': f"{int(low_hr)}-{int(high_hr)}",
        }
    return result


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

    Exige de VRAIS intervalles RR, dans la colonne `rr_ms`.

    Sans eux la fonction renvoie NaN. Elle fabriquait auparavant des RR
    synthetiques par un processus AR(1) a graine fixe, et renvoyait le
    resultat comme un flottant ordinaire : une valeur inventee atteignait
    ainsi la soupape de securite et pouvait bloquer une seance a haute
    intensite. Un processus autoregressif calibre sur une FC ne contient
    aucune information autonome — son exposant DFA reflete le generateur,
    pas la physiologie.

    Ni GoldenCheetah OpenData ni les fichiers .fit Coros du projet ne
    fournissent de RR, donc cette metrique est actuellement indisponible.
    C'est un manque assume, pas un trou a combler par simulation.

    Parameters
    ----------
    df : DataFrame nettoye avec une colonne 'rr_ms'
    params : configuration (si None, charge depuis params.yaml)

    Returns
    -------
    float : DFA alpha-1 de la seance, ou NaN si les RR sont absents
    """
    if params is None:
        params = load_params()

    if 'rr_ms' not in df.columns or not HAS_ADVANCED:
        return np.nan

    mask = get_running_mask(df, params)
    rr_series = df.loc[mask, 'rr_ms'].dropna().to_numpy()
    if len(rr_series) < 200:
        return np.nan

    return dfa_alpha1_from_rr(rr_series)


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

def session_features(df: pd.DataFrame, params: dict | None = None, cfg=None) -> dict:
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

    max_gap = params.get('signal', {}).get('max_gap_s', 30.0)
    df = attach_timebase(df, max_gap)
    mask = get_running_mask(df, params)

    features = {
        # Durees en secondes reelles : len(df) comptait des lignes, donc une
        # seance avec pauses etait systematiquement sous-evaluee.
        'duration_total_s':     round(active_duration_s(df, None, max_gap)),
        'duration_running_s':   round(active_duration_s(df, mask, max_gap)),
        'distance_m':           df['distance_m'].max() if 'distance_m' in df.columns else np.nan,
        'hr_mean':              df.loc[mask, 'hr'].mean() if 'hr' in df.columns else np.nan,
        'hr_max':               df.loc[mask, 'hr'].max() if 'hr' in df.columns else np.nan,
        # Minimum du signal NETTOYE : utile comme indicateur de qualite, mais
        # ce n'est pas une FC de repos. Savitzky-Golay lisse les extremes et
        # releve ce minimum de plusieurs dizaines de battements ; la FC de
        # repos se lit dans min_heartrate publie par l'appareil.
        'hr_min_smoothed':      df['hr'].min() if 'hr' in df.columns else np.nan,
        'speed_mean_kmh':       df.loc[mask, 'speed_kmh'].mean() if 'speed_kmh' in df.columns else np.nan,
        'power_mean':           df.loc[mask, 'power'].mean() if 'power' in df.columns else np.nan,
        'cadence_mean':         df.loc[mask, 'cadence'].mean() if 'cadence' in df.columns else np.nan,

        # Efficiency Factor
        'ef_global':            efficiency_factor(df),
        'ef_specific':          specific_ef(df, params, cfg),
        'ef_specific_gap':      specific_ef_gap(df, params, cfg),

        # Durabilité & Charge
        'cardiac_decoupling':   cardiac_decoupling(df, params),
        'trimp':                compute_trimp(df, params, cfg),
        'normalized_power':     normalized_power(df),

        # DFA α₁ (proxy seuil aérobie)
        'dfa_alpha1':           compute_dfa_alpha1(df, params),

    }

    # Zones aplaties en colonnes scalaires : une colonne portant un dict ne
    # survit pas proprement a un aller-retour Parquet.
    for zone_name, zone in compute_hr_zones(df, params, cfg).items():
        key = zone_name.split('_')[0]
        features[f'hr_zone_{key}_s'] = zone['seconds']
        features[f'hr_zone_{key}_pct'] = zone['pct']

    return features
