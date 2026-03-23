#!/usr/bin/env python3
"""
Data Quality Inspection — Valider les hypothèses de Phase 1 Analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from layer_a_signal.ingest_fit import batch_ingest

def inspect_data_quality():
    raw_data_dir = PROJECT_ROOT / 'data' / 'raw'
    batch_results = batch_ingest(raw_data_dir, verbose=False)

    print("=" * 80)
    print("DATA QUALITY INSPECTION")
    print("=" * 80)

    for filename, result in batch_results.items():
        if result['status'] != 'success':
            continue

        df = result['records']
        meta = result['session_meta']

        print(f"\n{'=' * 80}")
        print(f"📌 {filename}")
        print(f"{'=' * 80}")

        # ─── HR ANALYSIS ──────────────────────────────────────────────
        if 'hr' in df.columns:
            hr = df['hr'].dropna()
            print(f"\n💓 HEART RATE ({len(hr)}/{len(df)} valid):")
            print(f"   Mean:     {hr.mean():.1f} bpm")
            print(f"   Median:   {hr.median():.1f} bpm")
            print(f"   Std:      {hr.std():.1f} bpm")
            print(f"   Min-Max:  {hr.min():.0f} - {hr.max():.0f} bpm")

            # Detect jumps (Δ HR > 5 bpm/s = artefact optique)
            hr_diff = hr.diff().abs()
            big_jumps = (hr_diff > 5).sum()
            if big_jumps > 0:
                print(f"   ⚠️  Sauts > 5 bpm/s: {big_jumps} détectés")
                print(f"      Max jump: {hr_diff.max():.0f} bpm")

        # ─── SPEED ANALYSIS ──────────────────────────────────────────
        if 'speed_kmh' in df.columns:
            speed = df['speed_kmh'].dropna()
            print(f"\n🏃 SPEED ({len(speed)}/{len(df)} valid):")
            print(f"   Mean:     {speed.mean():.2f} km/h")
            print(f"   Median:   {speed.median():.2f} km/h")
            print(f"   Std:      {speed.std():.2f} km/h")
            print(f"   Min-Max:  {speed.min():.2f} - {speed.max():.2f} km/h")

            # Target pace analysis
            target = 3.667  # 3'40"/km in km/h
            tolerance_pct = 0.05
            in_target = ((speed >= target * (1 - tolerance_pct)) &
                        (speed <= target * (1 + tolerance_pct))).sum()
            print(f"   Points a 3'40\"/km +/-5%: {in_target} / {len(speed)} ({100*in_target/len(speed):.1f}%)")

            # Zeros detection
            zeros = (speed == 0).sum()
            if zeros > 0:
                print(f"   ⚠️  Vitesse = 0: {zeros} points (GPS perdu?)")

        # ─── POWER ANALYSIS ──────────────────────────────────────────
        if 'power' in df.columns:
            power = df['power'].dropna()
            non_zero = (power > 0).sum()
            print(f"\n⚡ POWER ({non_zero}/{len(power)} non-zero values):")
            if non_zero > 0:
                print(f"   Mean:     {power[power > 0].mean():.1f} W")
                print(f"   Max:      {power.max():.1f} W")
            else:
                print(f"   ⚠️  Aucune valeur de puissance! (montre sans capteur?)")

        # ─── ALTITUDE ANALYSIS ───────────────────────────────────────
        if 'altitude' in df.columns:
            alt = df['altitude'].dropna()
            if len(alt) > 0 and alt.std() > 0:
                print(f"\n⛰️  ALTITUDE ({len(alt)}/{len(df)} valid):")
                print(f"   Min-Max:  {alt.min():.0f} - {alt.max():.0f} m")
                total_gain = ((alt.diff() > 0) * alt.diff()).sum()  # Simplified
                print(f"   Total ascent (approx): {total_gain:.0f} m")
                print(f"   From session meta:     {meta.get('total_ascent', 0)} m")

        # ─── SIGNAL CONTINUITY ────────────────────────────────────────
        print(f"\n📊 SIGNAL CONTINUITY:")
        duration_s = len(df)
        duration_min = duration_s / 60
        expected_points = duration_s  # 1 Hz = 1 point/sec
        actual_points = len(df)
        print(f"   Duration: {duration_min:.1f} min ({duration_s}s)")
        print(f"   Expected points (1Hz): {expected_points}")
        print(f"   Actual points: {actual_points}")
        print(f"   Frequency: {actual_points/duration_s:.2f} Hz (expected 1.0)")

        # ─── RR DATA ──────────────────────────────────────────────────
        df_rr = result['rr']
        print(f"\n💓 HRV/RR DATA: {len(df_rr)} intervals")
        if len(df_rr) > 0:
            rr = df_rr['rr_s']
            print(f"   Mean RR:  {rr.mean()*1000:.1f} ms")
            print(f"   Range:    {rr.min()*1000:.1f} - {rr.max()*1000:.1f} ms")
            print(f"   RMSSD: [would calculate if data present]")
        else:
            print(f"   ⚠️  AUCUNE donnée RR trouvée!")

        # ─── SESSION METADATA ─────────────────────────────────────────
        print(f"\n📋 SESSION METADATA:")
        print(f"   Sport: {meta.get('sport')}")
        print(f"   Distance: {meta.get('total_distance')/1000:.2f} km")
        print(f"   Duration: {meta.get('total_timer_time'):.1f}s = {meta.get('total_timer_time')/60:.1f} min")
        print(f"   Avg HR: {meta.get('avg_heart_rate')} bpm")
        print(f"   Avg Speed: {meta.get('avg_speed')*3.6:.2f} km/h" if meta.get('avg_speed') else "   Avg Speed: N/A")
        print(f"   Avg Cadence: {meta.get('avg_cadence')} spm")

        # ─── SUMMARY VERDICT ──────────────────────────────────────────
        print(f"\n✅ QUALITY VERDICT:")
        issues = []
        if big_jumps > 100:
            issues.append("❌ Trop de sauts HR (> 100)")
        if non_zero == 0 and 'power' in df.columns:
            issues.append("⚠️  Power data missing")
        if len(df_rr) == 0:
            issues.append("⚠️  HRV/RR data missing")
        if zeros > len(speed) * 0.1:
            issues.append("⚠️  Trop de points vitesse=0 (> 10%)")

        if not issues:
            print("   ✅ Bonne qualité de données")
        else:
            for issue in issues:
                print(f"   {issue}")

    print("\n" + "=" * 80)
    print("END OF INSPECTION")
    print("=" * 80)

if __name__ == '__main__':
    inspect_data_quality()
