#!/usr/bin/env python3
"""
Compare EF_spec: Raw vs Cleaned Signal
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from layer_a_signal.ingest_fit import parse_fit
from layer_a_signal.features import load_params, specific_ef, clean_signal

# Load data
df_raw, _ = parse_fit(PROJECT_ROOT / 'data' / 'raw' / 'hrv.fit')
params = load_params()

print("=" * 80)
print("ISSUE: EF_SPEC Before vs After clean_signal()")
print("=" * 80)

# RAW
ef_raw = specific_ef(df_raw, params)
print(f"\n✅ RAW hrv.fit:    EF_spec = {ef_raw:.6f}")

# CLEANED
df_clean = clean_signal(df_raw, params)
ef_clean = specific_ef(df_clean, params)
print(f"❓ CLEANED hrv.fit: EF_spec = {ef_clean:.6f}")

if ef_raw != ef_clean:
    print(f"\n⚠️  MISMATCH DETECTED!")
    print(f"   Raw:    {ef_raw}")
    print(f"   Clean:  {ef_clean}")

    # Investigate what changed
    print(f"\n🔍 DETAILED COMPARISON:")
    print(f"   Raw points:   {len(df_raw)}")
    print(f"   Clean points: {len(df_clean)}")
    print(f"   Speed cols same: {(df_raw['speed_ms'] == df_clean['speed_ms']).all() or 'NaNs differ'}")
    print(f"   HR cols same:    {(df_raw['hr'] == df_clean['hr']).all() or 'NaNs differ'}")

    # Check NaNs before/after
    print(f"\n   NaNs before clean_signal():")
    print(f"      speed_ms: {df_raw['speed_ms'].isna().sum()}")
    print(f"      hr:       {df_raw['hr'].isna().sum()}")

    print(f"\n   NaNs after clean_signal():")
    print(f"      speed_ms: {df_clean['speed_ms'].isna().sum()}")
    print(f"      hr:       {df_clean['hr'].isna().sum()}")
