# PHASE 1 → PHASE 2 : ACTION PLAN

**Statut:** Phase 1 complétée avec succès
**Date:** 2026-03-23
**Prochaine phases:** 3 pistes parallélisables

---

## 🎯 CHOIX IMMÉDIAT (Décider Aujourd'hui)

### **OPTION A: Quick Wins (Recommandé)**
Faire le nettoyage avancé du signal en priorité.

```
Durée: 2-3 jours
Effort: Moyen (code clair, bien défini)
Output: Signal propre + features robustes
Blockers: Aucun
```

**À faire:**
1. [x] Implémenter détection sauts HR
2. [x] Filtre médian adaptif
3. [x] Flagging système
4. [x] Tester sur 2 séances réelles

---

### **OPTION B: Résoudre HRV d'Abord**
Décider et implémenter stratégie HRV.

```
Durée: Jour 1 = setup, Jour 2-8 = test réel
Effort: Low (setup), Medium (if app-based)
Output: RR data ou mock strategy
Blockers: Dépend de Coros feedback
```

**À faire:**
1. [ ] Essayer activer HRV Logging sur Coros
2. [ ] Si oui → test séance + vérifier data
3. [ ] Si non → setup HRV Logger app OU mock RR

---

### **OPTION C: Paralléliser Tout**
3 tasks en même temps (si équipe).

```
Task 1 (You): Signal Cleaning (2-3 jours)
Task 2 (You): HRV Decision (1 jour)
Task 3 (Async): Test new sessions with better setup
```

---

## 📋 IMMEDIATE ACTIONS (Today)

### ✅ For OPTION A (Recommended)

**Step 1: Backup & Branch**
```bash
git branch feature/phase2-signal-cleaning
git add -A
git commit -m "Phase 1 complete: batch ingest + export"
```

**Step 2: Create Phase 2 Features Module**
```bash
# src/layer_a_signal/features_advanced.py
# Functions:
#  - detect_hr_jumps(df, threshold=5)
#  - median_filter_adaptive(df, window=5)
#  - flag_artifact_segments(df)
#  - clean_signal_v2(df)
```

**Step 3: Implement Tests**
```bash
# tests/test_signal_cleaning.py
# Mock data with known artifacts
# Verify detection + correction
```

**Step 4: Run on longue.fit**
Verify the 45 jumps are eliminated.

---

### ⚠️ For OPTION B (HRV Decision)

**Step 1: Test Coros Settings**
```bash
1. Open Settings on Coros PACE 4
2. Training → HRV → Check if available
3. If yes: Enable + do short test run
4. If no: → HRV Logger app route
```

**Step 2: Document Result**
```bash
# Create: docs/HRV_STRATEGY.md
Strategy: [Active Coros | HRV Logger | Mock RR]
Timeline: [test date + duration]
Expected Data: [when available]
```

---

## 📊 CURRENT STATE (Quick Reference)

**Data Ready:**
```
✅ 2 x .fit files parsed
✅ 6,146 records extracted
✅ Parquet exported
✅ Features calculated (except DFA α₁, RR)

⚠️ 45 HR jumps in longue.fit (need cleaning)
⚠️ 0 RR intervals (need HRV setup)
```

**Pipeline Health:**
```
✅ Batch import: WORKING
✅ Feature extraction: WORKING (EF, TRIMP, etc)
✅ Export: WORKING
```

**Next Unlock:** Layer B (Bayesian model) requires:
```
1. Clean signal ← Phase 2a (quick)
2. RR data ← Phase 2b (depends on HRV)
```

---

## 🚀 ESTIMATED TIMELINE

| Phase | Task | Duration | Start | End |
|-------|------|----------|-------|-----|
| 2a | Signal Cleaning | 2-3 d | Now | Fri |
| 2b | HRV Decision | 1 d | Now | Tomorrow |
| 2c | Test Sessions | 1 w | Tomorrow | Next week |
| 3 | Bayesian Model | 1-2 w | Fri | 2 weeks |
| 4 | Optimizer (OR-Tools) | 1-2 w | 2 weeks | 4 weeks |
| 5 | Dashboard | 1 w | 4 weeks | 5 weeks |

---

## 💡 PRO TIPS FOR NEXT PHASES

### Phase 2a (Signal Cleaning)

**Tip 1: Visualize before/after**
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(df_raw['hr'], label='Raw')
ax2.plot(df_clean['hr'], label='Cleaned')
plt.show()
```

**Tip 2: Test on synthetic data first**
```python
# Create artificial jumps, verify detection
df_test = create_synthetic_with_artifacts()
assert detect_hr_jumps(df_test) > 0
```

**Tip 3: Document thresholds**
```yaml
# config/signal_cleaning.yaml
hr_jump_threshold_bpm: 5
median_filter_window: 5
physiological_constraints:
  max_hr_change_per_sec: 5
```

### Phase 2b (HRV Strategy)

**Tip 1: Plan for both paths**
```python
if hrv_available:
    rr_df = read_hrv_from_fit()
else:
    rr_df = read_hrv_csv_from_app()  # HRV Logger

if rr_df.empty:
    rr_df = generate_synthetic_rr()  # Fallback
```

**Tip 2: RR normalization**
```python
# Ensure RR intervals are in seconds
# Typical: 0.5 - 1.5 seconds (600-1200 ms)
assert (rr_df['rr_s'] > 0.2) & (rr_df['rr_s'] < 3).all()
```

---

## 📞 Questions Before Starting?

If stuck on:
- **Signal cleaning implementation** → Check Phase 2 Plan
- **HRV setup on Coros** → Check Coros manual
- **Parquet format questions** → Use `inspect_data_quality.py` as reference
- **Git workflow** → Standard feature branch workflow

---

## ✅ Checklist Before Phase 2

- [ ] Phase 1 report reviewed
- [ ] HRV strategy decided (Option A, B, or C)
- [ ] Branch created (`feature/phase2-*`)
- [ ] Test infrastructure ready
- [ ] Next session planned (if testing HRV)

---

*Ready to start? Pick an option and go!*
*Questions? Check PHASE1_FINAL_REPORT.md for deep dives.*
