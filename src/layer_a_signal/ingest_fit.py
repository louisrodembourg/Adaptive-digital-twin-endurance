"""
Layer A — Signal Lab : Ingestion des fichiers .fit (Coros)
Utilise le SDK officiel Garmin (garmin-fit-sdk) pour une lecture robuste.
"""

from garmin_fit_sdk import Decoder, Stream
import pandas as pd
import numpy as np
from pathlib import Path


def parse_fit(filepath: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse un fichier .fit et retourne deux DataFrames :
      - df_records : données seconde par seconde (FC, vitesse, puissance, etc.)
      - df_rr      : intervalles R-R bruts (si disponibles)

    Parameters
    ----------
    filepath : chemin vers le fichier .fit

    Returns
    -------
    (df_records, df_rr)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier .fit introuvable : {filepath}")

    stream = Stream.from_file(str(filepath))
    decoder = Decoder(stream)
    messages, errors = decoder.read()

    if errors:
        print(f"[ingest_fit] {len(errors)} erreur(s) non bloquante(s) lors du décodage.")

    # ─── RECORDS (1Hz) ───────────────────────────────────────────
    raw_records = messages.get('record_mesgs', [])
    if not raw_records:
        print("[ingest_fit] Aucun record trouvé dans le fichier.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(raw_records)

    # Mapping et renommage
    rename_map = {
        'timestamp':            'timestamp',
        'heart_rate':           'hr',
        'enhanced_speed':       'speed_ms',
        'enhanced_altitude':    'altitude',
        'cadence':              'cadence',
        'power':                'power',
        'vertical_oscillation': 'vert_osc',
        'stance_time':          'gct',
        'step_length':          'stride_length',
        'distance':             'distance_m',
        'position_lat':         'lat',
        'position_long':        'lon',
    }

    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df_clean = df[list(existing.keys())].rename(columns=existing).copy()

    # Conversions dérivées
    if 'speed_ms' in df_clean.columns:
        df_clean['speed_kmh'] = df_clean['speed_ms'] * 3.6
        df_clean['pace_min_km'] = np.where(
            df_clean['speed_ms'] > 0.5,
            1000 / (df_clean['speed_ms'] * 60),
            np.nan
        )

    # Timestamp en index
    if 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        df_clean = df_clean.set_index('timestamp')

    # ─── HRV / R-R ──────────────────────────────────────────────
    hrv_mesgs = messages.get('hrv_mesgs', [])
    rr_all = []
    for hrv_msg in hrv_mesgs:
        rr_values = hrv_msg.get('time', [])
        if isinstance(rr_values, (list, tuple)):
            rr_all.extend([v for v in rr_values if v is not None and 0.2 < v < 3.0])
        elif rr_values is not None and 0.2 < rr_values < 3.0:
            rr_all.append(rr_values)

    df_rr = pd.DataFrame({'rr_s': rr_all}) if rr_all else pd.DataFrame()

    return df_clean, df_rr


def get_session_summary(filepath: str | Path) -> dict:
    """
    Extrait les métadonnées de la session (durée totale, distance, etc.)
    depuis les messages 'session_mesgs'.
    """
    filepath = Path(filepath)
    stream = Stream.from_file(str(filepath))
    decoder = Decoder(stream)
    messages, _ = decoder.read()

    sessions = messages.get('session_mesgs', [])
    if not sessions:
        return {}

    s = sessions[0]
    return {
        'start_time':       s.get('start_time'),
        'total_timer_time': s.get('total_timer_time'),
        'total_distance':   s.get('total_distance'),
        'total_calories':   s.get('total_calories'),
        'avg_heart_rate':   s.get('avg_heart_rate'),
        'max_heart_rate':   s.get('max_heart_rate'),
        'avg_speed':        s.get('enhanced_avg_speed'),
        'max_speed':        s.get('enhanced_max_speed'),
        'avg_cadence':      s.get('avg_cadence'),
        'avg_power':        s.get('avg_power'),
        'max_power':        s.get('max_power'),
        'total_ascent':     s.get('total_ascent'),
        'total_descent':    s.get('total_descent'),
        'sport':            s.get('sport'),
        'sub_sport':        s.get('sub_sport'),
    }
