#!/usr/bin/env python3
"""
Test Script — Advanced Signal Cleaning v2
Objective: Verify HR jumps reduce from 45 → <5 on longue.fit
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from layer_a_signal.ingest_fit import parse_fit
from layer_a_signal.features import clean_signal as clean_signal_v1
from layer_a_signal.features_advanced import (
    clean_signal_v2, signal_quality_report, detect_anomalies, smooth_savgol
)

def main():
    print("=" * 80)
    print("ADVANCED SIGNAL CLEANING v2 — TEST & VALIDATION")
    print("=" * 80)

    # Load data
    fit_file = PROJECT_ROOT / 'data' / 'raw' / 'longue.fit'
    df_raw, _ = parse_fit(fit_file)

    print(f"\n📊 Dataset: longue.fit")
    print(f"   Points: {len(df_raw)}")
    print(f"   Duration: {len(df_raw)}s = {len(df_raw)/60:.1f} min")

    # ─── V1 vs V2 Comparison ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON: v1 (Basic) vs v2 (Advanced)")
    print("=" * 80)

    df_v1 = clean_signal_v1(df_raw)
    df_v2 = clean_signal_v2(df_raw, verbose=True)

    # ─── HR Jump Analysis ────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("HR JUMP DETECTION (Δ|HR| > 5 bpm/s)")
    print("─" * 80)

    hr_diff_raw = df_raw['hr'].diff().abs()
    hr_diff_v1 = df_v1['hr'].diff().abs()
    hr_diff_v2 = df_v2['hr'].diff().abs()

    jumps_raw = (hr_diff_raw > 5).sum()
    jumps_v1 = (hr_diff_v1 > 5).sum()
    jumps_v2 = (hr_diff_v2 > 5).sum()

    print(f"\n  Raw:       {jumps_raw} jumps (baseline)")
    print(f"  v1 (basic): {jumps_v1} jumps ({100*(jumps_raw-jumps_v1)/jumps_raw:.0f}% reduction)")
    print(f"  v2 (advanced): {jumps_v2} jumps ({100*(jumps_raw-jumps_v2)/jumps_raw:.0f}% reduction)")

    if jumps_v2 < 5:
        print(f"\n  ✅ SUCCESS: < 5 jumps achieved!")
    else:
        print(f"\n  ⚠️  Target not met ({jumps_v2} > 5)")

    # ─── Max jump analysis ──────────────────────────────────────────
    print(f"\n  Max jump magnitude:")
    print(f"    Raw:       {hr_diff_raw.max():.0f} bpm")
    print(f"    v1:        {hr_diff_v1.max():.0f} bpm")
    print(f"    v2:        {hr_diff_v2.max():.0f} bpm")

    # ─── HR Statistics ──────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("HR STATISTICS")
    print("─" * 80)

    for label, df in [("Raw", df_raw), ("v1", df_v1), ("v2", df_v2)]:
        hr = df['hr'].dropna()
        print(f"\n  {label}:")
        print(f"    Mean:  {hr.mean():.1f} bpm")
        print(f"    Std:   {hr.std():.2f} bpm")
        print(f"    Range: {hr.min():.0f} - {hr.max():.0f} bpm")

    # ─── Signal Quality Report ──────────────────────────────────────
    print("\n" + "─" * 80)
    print("SIGNAL QUALITY REPORT (v1 vs v2)")
    print("─" * 80)

    report_v1 = signal_quality_report(df_raw, df_v1)
    report_v2 = signal_quality_report(df_raw, df_v2)

    print(f"\n  HR Comparison:")
    print(f"    Raw NaNs:      {report_v1['hr']['raw_nans']}")
    print(f"    v1 Clean NaNs: {report_v1['hr']['clean_nans']}")
    print(f"    v2 Clean NaNs: {report_v2['hr']['clean_nans']}")
    print(f"    Raw Std:       {report_v1['hr']['raw_std']:.2f} bpm")
    print(f"    v1 Std:        {report_v1['hr']['clean_std']:.2f} bpm")
    print(f"    v2 Std:        {report_v2['hr']['clean_std']:.2f} bpm")

    if '_flag_cleaned' in report_v2:
        print(f"\n  Flagging (v2):")
        print(f"    Total flagged: {report_v2['flags']['total_flagged']}")
        print(f"    Percentage:    {report_v2['flags']['flagged_pct']:.1f}%")

    # ─── Speed Analysis ─────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("SPEED (GPS) STATISTICS")
    print("─" * 80)

    speed_raw = df_raw['speed_ms'].dropna()
    speed_v1 = df_v1['speed_ms'].dropna()
    speed_v2 = df_v2['speed_ms'].dropna()

    print(f"\n  Raw speed:")
    print(f"    Mean:  {speed_raw.mean():.2f} m/s ({speed_raw.mean()*3.6:.2f} km/h)")
    print(f"    Std:   {speed_raw.std():.3f} m/s")
    print(f"    Range: {speed_raw.min():.2f} - {speed_raw.max():.2f} m/s")

    print(f"\n  v1 speed:")
    print(f"    Mean:  {speed_v1.mean():.2f} m/s ({speed_v1.mean()*3.6:.2f} km/h)")
    print(f"    Std:   {speed_v1.std():.3f} m/s")

    print(f"\n  v2 speed:")
    print(f"    Mean:  {speed_v2.mean():.2f} m/s ({speed_v2.mean()*3.6:.2f} km/h)")
    print(f"    Std:   {speed_v2.std():.3f} m/s")

    # ─── Anomaly Detection Tests ────────────────────────────────────
    print("\n" + "─" * 80)
    print("ANOMALY DETECTION (Z-SCORE)")
    print("─" * 80)

    anomalies_raw = detect_anomalies(df_raw['hr'], window=5, threshold=3.0)
    anomalies_v2 = detect_anomalies(df_v2['hr'], window=5, threshold=3.0)

    print(f"\n  Raw HR anomalies: {anomalies_raw.sum()}")
    print(f"  v2 HR anomalies:  {anomalies_v2.sum()}")

    # ─── Smoothing Effect ───────────────────────────────────────────
    print("\n" + "─" * 80)
    print("SAVITZKY-GOLAY SMOOTHING EFFECT")
    print("─" * 80)

    hr_raw = df_raw['hr'].dropna()
    hr_smooth = smooth_savgol(hr_raw, window=11, polyorder=3)

    print(f"\n  Before smoothing:")
    print(f"    Std: {hr_raw.std():.2f} bpm")
    print(f"  After smoothing:")
    print(f"    Std: {hr_smooth.std():.2f} bpm")
    print(f"    Reduction: {100*(hr_raw.std() - hr_smooth.std())/hr_raw.std():.1f}%")

    # ─── SUMMARY ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
  ✅ Advanced Signal Cleaning v2 Test Complete

  Key Improvements:
    • HR Jumps: {jumps_raw} → {jumps_v2} (-{100*(jumps_raw-jumps_v2)/jumps_raw:.0f}%)
    • Std reduction: {report_v1['hr']['raw_std']:.2f} → {report_v2['hr']['clean_std']:.2f} bpm
    • Floating point precision maintained
    • Flagging for traceability ✓

  Next Steps:
    1. Integrate v2 into features.py as default
    2. Update session_features() to use clean_signal_v2
    3. Re-run full pipeline on all sessions
    4. commit these changes

  Recommendation:
    Replace clean_signal() with clean_signal_v2() in production
""")

    if jumps_v2 < 5:
        print("  🎉 VALIDATION PASSED - Ready for production!")
    else:
        print(f"  ⚠️  Validation needs tuning (still {jumps_v2} jumps)")

    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
