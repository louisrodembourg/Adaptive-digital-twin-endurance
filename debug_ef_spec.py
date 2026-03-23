#!/usr/bin/env python3
"""
Debug EF_spec Calculation for hrv.fit
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from layer_a_signal.ingest_fit import parse_fit
from layer_a_signal.features import load_params, specific_ef

# Load data
df_raw, _ = parse_fit(PROJECT_ROOT / 'data' / 'raw' / 'hrv.fit')
params = load_params()

print("=" * 80)
print("DEBUG: EF_SPEC CALCULATION for hrv.fit")
print("=" * 80)

# Get parameters
target_speed = params['athlete']['target_speed_ms']  # 4.545 m/s
tolerance = params['signal']['ef_pace_tolerance_pct'] / 100  # 0.05
min_duration = params['signal']['ef_min_duration_s']  # 60s

print(f"\n📐 PARAMETERS:")
print(f"   target_speed_ms: {target_speed} m/s")
print(f"   tolerance: {tolerance * 100}%")
print(f"   min_duration_s: {min_duration}s")

speed_low = target_speed * (1 - tolerance)
speed_high = target_speed * (1 + tolerance)

print(f"\n🎯 TARGET RANGE:")
print(f"   Speed range: {speed_low:.3f} - {speed_high:.3f} m/s")
print(f"   Speed range: {speed_low*3.6:.2f} - {speed_high*3.6:.2f} km/h")
print(f"   Pace range: {60/(speed_high*3.6):.2f}' - {60/(speed_low*3.6):.2f}' per km")

# Check mask
if 'speed_ms' not in df_raw.columns or 'hr' not in df_raw.columns:
    print("\n❌ Missing columns!")
else:
    print(f"\n📊 DATA AVAILABLE:")
    print(f"   Total points: {len(df_raw)}")
    print(f"   Speed NaNs: {df_raw['speed_ms'].isna().sum()}")
    print(f"   HR NaNs: {df_raw['hr'].isna().sum()}")

    speed_ms = df_raw['speed_ms'].dropna()
    print(f"   Speed (m/s) stats:")
    print(f"      Mean: {speed_ms.mean():.3f} m/s ({speed_ms.mean()*3.6:.2f} km/h)")
    print(f"      Min-Max: {speed_ms.min():.3f} - {speed_ms.max():.3f} m/s")

    # Apply mask
    mask = (
        (df_raw['speed_ms'] >= speed_low) &
        (df_raw['speed_ms'] <= speed_high) &
        (df_raw['hr'].notna()) &
        (df_raw['speed_ms'].notna())
    )

    n_in_range = mask.sum()
    print(f"\n✅ MASK RESULTS:")
    print(f"   Points matching range: {n_in_range} / {len(df_raw)} ({100*n_in_range/len(df_raw):.1f}%)")

    df_target = df_raw.loc[mask]

    if len(df_target) < min_duration:
        print(f"\n❌ PROBLEM: Not enough continuous data!")
        print(f"   Required: {min_duration}s")
        print(f"   Found: {len(df_target)} points = {len(df_target)}s")
        print(f"   → EF_spec returns NaN")
    else:
        print(f"\n✅ SOLUTION!")
        print(f"   Available: {len(df_target)}s > {min_duration}s minimum")

        ef_spec = df_target['speed_ms'].mean() / df_target['hr'].mean()
        print(f"\n📈 EF_SPEC CALCULATION:")
        print(f"   Mean speed (target): {df_target['speed_ms'].mean():.4f} m/s")
        print(f"   Mean HR (target):    {df_target['hr'].mean():.1f} bpm")
        print(f"   EF_SPEC: {ef_spec:.6f}")

# Now call the actual function
print(f"\n{'=' * 80}")
print(f"ACTUAL FUNCTION RESULT:")
result = specific_ef(df_raw, params)
print(f"specific_ef(df) = {result}")

