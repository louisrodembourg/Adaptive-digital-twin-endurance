"""
Layer A — Signal Lab : Ingestion des fichiers .fit (Coros)
Utilise le SDK officiel Garmin (garmin-fit-sdk) pour une lecture robuste.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream


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


def batch_ingest(directory: str | Path, verbose: bool = True) -> dict[str, dict]:
    """
    Ingère TOUS les fichiers .fit d'un dossier.

    Parameters
    ----------
    directory : chemin vers le dossier contenant les fichiers .fit
    verbose : affiche les logs d'ingestion

    Returns
    -------
    dict avec structure :
    {
        'filename.fit': {
            'records': pd.DataFrame,
            'rr': pd.DataFrame,
            'session_meta': dict,
            'status': 'success' ou 'error',
            'error_msg': str (si erreur)
        },
        ...
    }
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    fit_files = sorted(directory.glob('*.fit'))
    if not fit_files:
        if verbose:
            print(f"[batch_ingest] Aucun fichier .fit trouvé dans {directory}")
        return {}

    if verbose:
        print(f"[batch_ingest] Trouvé {len(fit_files)} fichier(s) .fit")

    results = {}

    for fit_file in fit_files:
        filename = fit_file.name
        if verbose:
            print(f"\n→ Ingestion {filename}...", end=" ")

        try:
            df_records, df_rr = parse_fit(fit_file)
            session_meta = get_session_summary(fit_file)

            results[filename] = {
                'records': df_records,
                'rr': df_rr,
                'session_meta': session_meta,
                'status': 'success',
                'error_msg': None,
            }

            if verbose:
                n_records = len(df_records)
                n_rr = len(df_rr)
                duration_s = session_meta.get('total_timer_time', 0)
                distance_m = session_meta.get('total_distance', 0)
                print(f"✓ ({n_records} records, {n_rr} RR, {duration_s}s, {distance_m}m)")

        except Exception as e:
            results[filename] = {
                'records': pd.DataFrame(),
                'rr': pd.DataFrame(),
                'session_meta': {},
                'status': 'error',
                'error_msg': str(e),
            }
            if verbose:
                print(f"✗ Erreur: {e}")

    return results


def export_to_parquet(
    batch_results: dict[str, dict],
    output_dir: str | Path,
    verbose: bool = True
) -> dict[str, Path]:
    """
    Exporte les DataFrames ingérés en fichiers Parquet.

    Crée une structure :
    - output_dir/records_YYYYMMDD_HHMMSS.parquet
    - output_dir/rr_YYYYMMDD_HHMMSS.parquet

    Parameters
    ----------
    batch_results : dict retourné par batch_ingest()
    output_dir : dossier de destination
    verbose : affiche les logs

    Returns
    -------
    dict avec chemins des fichiers créés
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Combiner tous les records et RR
    all_records = []
    all_rr = []

    for filename, result in batch_results.items():
        if result['status'] != 'success':
            continue

        df_rec = result['records'].copy()
        df_rr = result['rr'].copy()

        # Ajouter le source filename pour traçabilité
        df_rec['_source_file'] = filename
        if not df_rr.empty:
            df_rr['_source_file'] = filename

        all_records.append(df_rec)
        if not df_rr.empty:
            all_rr.append(df_rr)

    paths = {}

    # Export records
    if all_records:
        df_combined_rec = pd.concat(all_records, ignore_index=False)
        records_path = output_dir / f'records_{timestamp}.parquet'
        df_combined_rec.to_parquet(records_path, compression='snappy')
        paths['records'] = records_path
        if verbose:
            print(f"✓ Records exportés: {records_path.name} ({len(df_combined_rec)} lignes)")

    # Export RR
    if all_rr:
        df_combined_rr = pd.concat(all_rr, ignore_index=True)
        rr_path = output_dir / f'rr_{timestamp}.parquet'
        df_combined_rr.to_parquet(rr_path, compression='snappy')
        paths['rr'] = rr_path
        if verbose:
            print(f"✓ RR exportés: {rr_path.name} ({len(df_combined_rr)} lignes)")

    return paths
