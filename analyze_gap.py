#!/usr/bin/env python3
"""
Advanced Analysis: EF_spec Issue & GAP Calculation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from layer_a_signal.ingest_fit import batch_ingest

def gap_analysis():
    """
    Analyser pourquoi EF_spec = NaN sur longue.fit
    Et calculer GAP (Grade Adjusted Pace) pour voir l'impact
    """
    raw_data_dir = PROJECT_ROOT / 'data' / 'raw'
    batch_results = batch_ingest(raw_data_dir, verbose=False)

    print("=" * 80)
    print("GAP (Grade Adjusted Pace) ANALYSIS")
    print("=" * 80)

    for filename, result in batch_results.items():
        if result['status'] != 'success':
            continue

        df = result['records']
        meta = result['session_meta']

        print(f"\n{'=' * 80}")
        print(f"📌 {filename}")
        print(f"{'=' * 80}")

        # ─── BASIC SPEED STATS ────────────────────────────────────────
        speed_kmh = df['speed_kmh'].dropna()
        print(f"\n🏃 BASIC SPEED STATS:")
        print(f"   Mean speed: {speed_kmh.mean():.2f} km/h")
        print(f"   Std dev:    {speed_kmh.std():.2f} km/h")
        print(f"   Min-Max:    {speed_kmh.min():.2f} - {speed_kmh.max():.2f} km/h")

        pace_min_km_mean = 60 / speed_kmh.mean()  # 60 min / speed in km/h
        print(f"   Mean pace:  {pace_min_km_mean:.2f}' / {(pace_min_km_mean % 1 * 60):.0f}\" per km")

        # ─── GAP CALCULATION ──────────────────────────────────────────
        print(f"\n⛰️  ELEVATION DATA:")
        altitude = df['altitude'].dropna()
        total_distance = meta.get('total_distance', 0) / 1000  # km
        total_ascent = meta.get('total_ascent', 0)  # m
        total_descent = meta.get('total_descent', 0)  # m

        print(f"   Total distance: {total_distance:.2f} km")
        print(f"   Total ascent:   {total_ascent} m")
        print(f"   Total descent:  {total_descent} m")

        # GAP formula (Garmin standard):
        # GAP attempts to estimate pace on flat terrain
        # Simplified: For every 100m of elevation gained, add X seconds per km

        if total_distance > 0:
            elevation_gradient = (total_ascent - total_descent) / total_distance  # m/km
            print(f"   Elevation gradient: {elevation_gradient:.1f} m/km")

            # Rough GAP calculation:
            # For climbing: +12 sec/km per 100m elevation gain
            # For descent: -8 sec/km per 100m elevation loss
            climb_penalty_s_per_km = (total_ascent / 100) * 12 / total_distance if total_ascent > 0 else 0
            descent_bonus_s_per_km = (total_descent / 100) * 8 / total_distance if total_descent > 0 else 0

            gap_penalty_s_per_km = climb_penalty_s_per_km - descent_bonus_s_per_km
            gap_pace_s = (pace_min_km_mean * 60) + gap_penalty_s_per_km

            print(f"\n   GAP CALCULATION:")
            print(f"   Actual pace:  {pace_min_km_mean * 60:.0f} sec/km")
            print(f"   Climb penalty: +{climb_penalty_s_per_km:.1f} sec/km")
            print(f"   Descent bonus: -{descent_bonus_s_per_km:.1f} sec/km")
            print(f"   GAP pace:     {gap_pace_s:.0f} sec/km = {gap_pace_s/60:.2f}' / {(gap_pace_s % 60):.0f}\" per km")

        # ─── EF_SPEC ISSUE ────────────────────────────────────────────
        print(f"\n📊 EF_SPEC ANALYSIS (Target: 3'40\"/km = 16.4 km/h):")
        tolerance_pct = 0.05
        target_speed_kmh = 16.4

        in_range = ((speed_kmh >= target_speed_kmh * (1 - tolerance_pct)) &
                    (speed_kmh <= target_speed_kmh * (1 + tolerance_pct)))

        n_in_range = in_range.sum()
        n_total = len(speed_kmh)

        print(f"   Points in range: {n_in_range} / {n_total} ({100*n_in_range/n_total:.1f}%)")

        if n_in_range == 0:
            print(f"\n   ✅ DIAGNOSIS: NOT A BUG")
            print(f"   → {filename} averages {speed_kmh.mean():.1f} km/h")
            print(f"   → Target pace {target_speed_kmh:.1f} km/h is NOT part of this session")
            print(f"   → This is expected!")

            if filename == 'hrv.fit':
                print(f"\n   💡 hrv.fit is a HIGH-INTENSITY session (HIT/VO2max)")
                print(f"      Mean HR: 173 bpm (your max is ~205)")
                print(f"      → Running much faster than target race pace")

            elif filename == 'longue.fit':
                print(f"\n   💡 longue.fit is an ENDURANCE session")
                print(f"      Mean speed: {speed_kmh.mean():.1f} km/h (slower than target)")
                print(f"      Mean HR: 144 bpm (your max is ~205)")
                print(f"      {total_ascent} m elevation on {total_distance:.1f} km")
                print(f"      → Running at conservative endurance pace")

        # ─── DISTRIBUTION ────────────────────────────────────────────
        print(f"\n📈 SPEED DISTRIBUTION:")
        percentiles = [5, 25, 50, 75, 95]
        for p in percentiles:
            val = speed_kmh.quantile(p / 100)
            print(f"   P{p:02d}: {val:.2f} km/h ({60/val:.2f}' per km)")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    gap_analysis()
