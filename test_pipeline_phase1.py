#!/usr/bin/env python3
"""
Test Script — Phase 1 Complete Pipeline
Orchestre : Batch Ingestion → Feature Extraction → Parquet Export
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from layer_a_signal.ingest_fit import batch_ingest, export_to_parquet
from layer_a_signal.features import clean_signal, session_features
import pandas as pd


def main():
    print("=" * 70)
    print("PHASE 1 — PIPELINE COMPLETE")
    print("=" * 70)

    # Config
    raw_data_dir = PROJECT_ROOT / 'data' / 'raw'
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Input:  {raw_data_dir}")
    print(f"📂 Output: {processed_dir}")

    # ─── STEP 1: BATCH INGEST ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 1: BATCH INGEST — Parse tous les fichiers .fit")
    print("─" * 70)

    batch_results = batch_ingest(raw_data_dir, verbose=True)

    if not batch_results:
        print("❌ Aucun fichier ingéré. Abandon.")
        return

    n_success = sum(1 for r in batch_results.values() if r['status'] == 'success')
    n_errors = sum(1 for r in batch_results.values() if r['status'] == 'error')
    print(f"\n✓ {n_success} fichier(s) ingéré(s) avec succès")
    if n_errors > 0:
        print(f"⚠ {n_errors} erreur(s)")

    # ─── STEP 2: DETAILED SESSION SUMMARIES ───────────────────────────
    print("\n" + "─" * 70)
    print("STEP 2: SESSION SUMMARIES")
    print("─" * 70)

    for filename, result in batch_results.items():
        if result['status'] != 'success':
            continue

        meta = result['session_meta']
        df_rec = result['records']

        print(f"\n📌 {filename}")
        print(f"   Start:      {meta.get('start_time')}")
        print(f"   Duration:   {meta.get('total_timer_time')} s")
        print(f"   Distance:   {meta.get('total_distance')/1000:.2f} km")
        print(f"   Avg HR:     {meta.get('avg_heart_rate')} bpm")
        print(f"   Records:    {len(df_rec)} points (1Hz)")

    # ─── STEP 3: SIGNAL CLEANING ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 3: SIGNAL CLEANING & FEATURE EXTRACTION")
    print("─" * 70)

    session_reports = []

    for filename, result in batch_results.items():
        if result['status'] != 'success':
            continue

        df_raw = result['records']
        print(f"\n→ Processing {filename}...")

        # Clean signal
        df_clean = clean_signal(df_raw)
        print(f"   Nettoyage: ✓ HR nans={df_clean['hr'].isna().sum()}, Speed nans={df_clean['speed_ms'].isna().sum()}")

        # Extract features
        features = session_features(df_clean)

        # Build report
        report = {
            '_filename': filename,
            '_start_time': result['session_meta'].get('start_time'),
            **features
        }

        session_reports.append(report)

        # Print summary
        print(f"   EF Global:     {features['ef_global']:.4f}")
        print(f"   EF Spécifique: {features['ef_specific']:.4f}")
        print(f"   Cardiac Decoupling: {features['cardiac_decoupling']:.2f}%")
        print(f"   TRIMP:         {features['trimp']:.1f}")
        print(f"   HR Zones:      {list(features['hr_zones'].keys())}")

    # Create summary dataframe
    df_summary = pd.DataFrame(session_reports)

    # ─── STEP 4: EXPORT TO PARQUET ────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 4: EXPORT TO PARQUET")
    print("─" * 70)

    parquet_paths = export_to_parquet(batch_results, processed_dir, verbose=True)

    # Save session summary
    if not df_summary.empty:
        summary_path = processed_dir / 'session_summary.parquet'
        df_summary.to_parquet(summary_path, compression='snappy')
        print(f"✓ Session summary: {summary_path.name} ({len(df_summary)} séances)")

    # ─── STEP 5: VALIDATION & INSIGHTS ────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 5: VALIDATION & INSIGHTS")
    print("─" * 70)

    if parquet_paths.get('records'):
        df_all_records = pd.read_parquet(parquet_paths['records'])
        print(f"\n📊 Total Records Dataset:")
        print(f"   Shape: {df_all_records.shape}")
        print(f"   Columns: {list(df_all_records.columns)}")
        print(f"   Date range: {df_all_records.index.min()} → {df_all_records.index.max()}")
        print(f"   HR range: {df_all_records['hr'].min():.0f} - {df_all_records['hr'].max():.0f} bpm")
        print(f"   Speed range: {df_all_records['speed_kmh'].min():.1f} - {df_all_records['speed_kmh'].max():.1f} km/h")

    if parquet_paths.get('rr'):
        df_all_rr = pd.read_parquet(parquet_paths['rr'])
        print(f"\n💓 HRV/RR Dataset:")
        print(f"   Shape: {df_all_rr.shape}")
        print(f"   Total RR intervals: {len(df_all_rr)}")
        print(f"   RR range: {df_all_rr['rr_s'].min():.3f} - {df_all_rr['rr_s'].max():.3f} s")
    else:
        print(f"\n⚠ HRV/RR Dataset: AUCUNE donnée RR trouvée dans les fichiers .fit")
        print(f"   → Besoin d'activer 'HRV Logging' sur Coros ou utiliser HRV Logger externe")

    # ─── SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1 — RÉSUMÉ")
    print("=" * 70)
    print(f"✓ Fichiers ingérés: {n_success}")
    print(f"✓ Sessions analysées: {len(session_reports)}")
    print(f"✓ Fichiers Parquet générés: {len(parquet_paths)}")
    print(f"✓ Chemin sortie: {processed_dir}")
    print("\n🎯 Prochaine étape: Phase 2 — Nettoyage avancé du signal + DFA α₁")
    print("=" * 70)


if __name__ == '__main__':
    main()
