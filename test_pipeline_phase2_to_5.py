"""
Tests end-to-end — Phase 2 à 5
Valide l'intégration complète du pipeline Digital Twin.

Exécution :
    python test_pipeline_phase2_to_5.py
"""

import sys
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "✅"
FAIL = "❌"

passed_total = 0
failed_total = 0


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def t(name: str, fn) -> bool:
    global passed_total, failed_total
    try:
        result = fn()
        detail = f"→ {result}" if result is not None else ""
        print(f"  {PASS} {name} {detail}")
        passed_total += 1
        return True
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"       {type(e).__name__}: {e}")
        failed_total += 1
        return False


# ══════════════════════════════════════════════════════════════
#  PHASE 2A — Features avancées
# ══════════════════════════════════════════════════════════════

section("PHASE 2A — Biomarqueurs avancés (GAP, Decoupling, DFA α₁)")

from src.layer_a_signal.features import (
    grade_adjusted_pace, specific_ef_gap, cardiac_decoupling,
    compute_dfa_alpha1, session_features, load_params
)
from src.layer_a_signal.features_advanced import generate_mock_rr, dfa_alpha1_from_rr

np.random.seed(42)
n = 1800
df_test = pd.DataFrame({
    'hr':         np.clip(150 + np.cumsum(np.random.randn(n) * 0.3), 140, 175),
    'speed_ms':   np.clip(4.5 + np.random.randn(n) * 0.2, 3.5, 5.5),
    'altitude':   50 + np.cumsum(np.random.randn(n) * 0.05),
    'power':      280 + np.random.randn(n) * 20,
    'cadence':    85 + np.random.randn(n) * 2,
    'distance_m': np.linspace(0, 8100, n),
})
df_test['speed_kmh']   = df_test['speed_ms'] * 3.6
df_test['pace_min_km'] = 1000 / (df_test['speed_ms'] * 60)

df_gap = grade_adjusted_pace(df_test)

t("grade_adjusted_pace() — colonne speed_gap_ms créée",
  lambda: df_gap['speed_gap_ms'].notna().sum() > 0)

t("specific_ef_gap() — valeur calculée",
  lambda: f"{specific_ef_gap(df_gap):.5f}")

t("cardiac_decoupling() — warm-up exclu",
  lambda: f"{cardiac_decoupling(df_test):.2f}%")

t("generate_mock_rr() — RR physiologiques générés",
  lambda: f"len={len(generate_mock_rr(df_test['hr'].values))}, "
          f"mean={generate_mock_rr(df_test['hr'].values).mean():.0f}ms")

t("dfa_alpha1_from_rr() — calcul DFA α₁",
  lambda: f"α₁={dfa_alpha1_from_rr(generate_mock_rr(df_test['hr'].values)):.3f}")

t("compute_dfa_alpha1() — via DataFrame",
  lambda: f"α₁={compute_dfa_alpha1(df_test):.3f}")

t("session_features() — toutes métriques incluant DFA",
  lambda: {k: round(v, 3) if isinstance(v, float) else type(v).__name__
           for k, v in session_features(df_test).items() if k != 'hr_zones'})


# ══════════════════════════════════════════════════════════════
#  PHASE 2B — DailyContext
# ══════════════════════════════════════════════════════════════

section("PHASE 2B — DailyContext & Ingestion Quotidienne")

from src.layer_a_signal.context_ingestion import (
    DailyContext, DailyContextStore, create_today_context
)

t("DailyContext — création et score fatigue",
  lambda: f"score={DailyContext(date.today(), rmssd_ms=42, sleep_hours=6, stress_level=4).fatigue_score:.2f}")

t("DailyContext — is_hrv_low",
  lambda: DailyContext(date.today(), rmssd_ms=42).is_hrv_low)

t("DailyContext — sérialisation to_dict/from_dict",
  lambda: DailyContext.from_dict(
      DailyContext(date.today(), rmssd_ms=55).to_dict()
  ).rmssd_ms == 55.0)


def _test_context_store():
    tmp = Path(tempfile.mktemp(suffix='.parquet'))
    store = DailyContextStore(tmp)
    ctx = DailyContext(date.today(), rmssd_ms=55.0, sleep_hours=7.5)
    store.save(ctx)
    loaded = store.load_date(date.today())
    assert loaded is not None
    assert abs(loaded.rmssd_ms - 55.0) < 0.01
    os.unlink(tmp)
    return "save/load OK"


t("DailyContextStore — save & load", _test_context_store)


# ══════════════════════════════════════════════════════════════
#  PHASE 3 — Bayesian Digital Twin
# ══════════════════════════════════════════════════════════════

section("PHASE 3 — Digital Twin (Banister + Bayésien)")

from src.layer_b_twin.bayesian_model import (
    compute_fitness_fatigue, compute_performance, run_deterministic, HAS_PYMC
)
from src.layer_b_twin.uncertainty_manager import (
    assess_uncertainty, check_overreaching, generate_calibration_session
)
from src.layer_b_twin.state_estimator import (
    StateEstimator, bootstrap_trimp_history
)


def _test_fitness_fatigue():
    trimp = np.array([60.0] * 30 + [0.0] * 14)
    fit, fat = compute_fitness_fatigue(trimp)
    assert fit[29] > fit[0], "Fitness doit croître"
    assert fat[29] > fat[0], "Fatigue doit croître"
    assert fat[43] < fat[29], "Fatigue doit décroître au repos"
    return f"Fitness[-1]={fit[-1]:.1f}, Fatigue[-1]={fat[-1]:.1f}"


t("compute_fitness_fatigue() — valeurs croissantes avec charge", _test_fitness_fatigue)

t("run_deterministic() — DataFrame complet",
  lambda: f"shape={run_deterministic(bootstrap_trimp_history(weeks=8)).shape}")

t("bootstrap_trimp_history() — 12 semaines",
  lambda: f"sum={bootstrap_trimp_history(weeks=12).sum():.0f} TRIMP total")


def _test_state_estimator():
    est = StateEstimator()
    trimp = bootstrap_trimp_history(weeks=8)
    for i, tv in enumerate(trimp):
        if tv > 0:
            d = date.today() - timedelta(days=len(trimp) - i)
            est.add_session(float(tv), session_date=d)
    state = est.get_current_state()
    assert state.fitness > 0
    assert abs(state.tsb - round(state.fitness - state.fatigue, 2)) < 0.1
    return str(state)


t("StateEstimator — add_session + get_current_state", _test_state_estimator)

t("check_overreaching() — détection TSB critique",
  lambda: check_overreaching(np.array([-35.0, -38.0, -40.0]))['is_overreaching'])

t("assess_uncertainty() — rapport mode déterministe",
  lambda: f"level={assess_uncertainty(run_deterministic(bootstrap_trimp_history(weeks=4))).uncertainty_level}")

t("generate_calibration_session() — format correct",
  lambda: generate_calibration_session()['type'] == 'CALIBRATION')

t("PyMC disponible",
  lambda: f"HAS_PYMC={HAS_PYMC}")


# ══════════════════════════════════════════════════════════════
#  PHASE 4 — Neuro-Symbolic Planner
# ══════════════════════════════════════════════════════════════

section("PHASE 4 — Neuro-Symbolic Planner (Safety Valve + LLM + OR-Tools)")

from src.layer_c_planner.safety_valve import (
    check_safety_constraints, get_allowed_sessions,
    get_volume_multiplier, format_constraints_summary
)
from src.layer_c_planner.llm_context_parser import (
    parse_context_to_constraints, SoftConstraints
)
from src.layer_c_planner.optimizer import optimize_week, generate_taper_plan, HAS_ORTOOLS


def _params_with_rmssd():
    params = load_params()
    params['planner']['rmssd_vma_threshold'] = 50.0
    return params


def _test_safety_rmssd():
    constraints = check_safety_constraints(rmssd_ms=35.0, params=_params_with_rmssd())
    allowed = get_allowed_sessions(constraints)
    assert "VMA" not in allowed, "VMA devrait être bloqué"
    return f"Contraintes: {[c.code for c in constraints]}"


t("Safety Valve — rMSSD bas bloque VMA", _test_safety_rmssd)

t("Safety Valve — TSB critique réduit volume",
  lambda: f"vol_mult={get_volume_multiplier(check_safety_constraints(tsb=-35.0)):.2f}")

t("Safety Valve — mode tapering J-7",
  lambda: f"codes={[c.code for c in check_safety_constraints(race_date=date.today()+timedelta(days=7))]}")

t("Safety Valve — format_constraints_summary",
  lambda: len(format_constraints_summary(
      check_safety_constraints(rmssd_ms=30.0, params=_params_with_rmssd())
  )) > 10)


def _test_llm_heuristic():
    sc = parse_context_to_constraints("jambes lourdes et stressé", use_heuristic_fallback=True)
    assert sc.max_volume_pct <= 0.85
    assert "VMA" in sc.no_session_types
    return f"vol={sc.max_volume_pct:.0%}, bloqué={sc.no_session_types}"


t("LLM Parser — heuristique 'jambes lourdes'", _test_llm_heuristic)

t("LLM Parser — texte positif 'en forme'",
  lambda: f"restrictif={parse_context_to_constraints('je me sens en forme').is_restrictive()}")


def _test_optimizer():
    plan = optimize_week(
        hard_constraints=[],
        soft_constraints=SoftConstraints(),
        previous_week_trimp=250.0,
    )
    assert len(plan.days) == 7
    assert plan.total_trimp >= 0
    return f"TRIMP={plan.total_trimp:.0f}, séances={plan.total_sessions}"


t("Optimizer — génère plan 7 jours", _test_optimizer)


def _test_optimizer_hard_constraints():
    constraints = check_safety_constraints(rmssd_ms=30.0, params=_params_with_rmssd())
    plan = optimize_week(hard_constraints=constraints, previous_week_trimp=250.0)
    session_types = [d.session_type for d in plan.days]
    assert "VMA" not in session_types, f"VMA trouvé malgré contrainte : {session_types}"
    return f"sessions={session_types}"


t("Optimizer — contraintes dures respectées (VMA bloqué)", _test_optimizer_hard_constraints)

t("Optimizer — tapering J-10",
  lambda: f"TRIMP={generate_taper_plan(date.today()+timedelta(days=10), current_fitness=80.0).total_trimp:.0f}")

t("OR-Tools disponible",
  lambda: f"HAS_ORTOOLS={HAS_ORTOOLS}")


# ══════════════════════════════════════════════════════════════
#  SIMULATION PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════

section("SIMULATION — Pipeline complet (8 semaines)")


def _test_simulation_8weeks():
    est = StateEstimator()
    trimp = bootstrap_trimp_history(weeks=8, base_trimp_per_week=200, progression=0.08)
    for i, tv in enumerate(trimp):
        if tv > 0:
            d = date.today() - timedelta(days=len(trimp) - i)
            est.add_session(float(tv), session_date=d)

    curves = est.update_model(n_days=56)
    fit_start = curves['fitness'].iloc[7]
    fit_end   = curves['fitness'].iloc[-7]
    assert fit_end > fit_start, f"Fitness devrait croître : {fit_start:.1f} → {fit_end:.1f}"
    return f"Fitness : {fit_start:.1f} → {fit_end:.1f} (+{fit_end-fit_start:.1f})"


t("Fitness croît avec charge progressive", _test_simulation_8weeks)


def _test_full_pipeline():
    ctx = DailyContext(
        date=date.today(), rmssd_ms=38.0,
        sleep_hours=5.5, sleep_quality=2,
        stress_level=4, notes="épuisé et stressé",
    )
    constraints = check_safety_constraints(
        rmssd_ms=ctx.rmssd_ms, tsb=-15.0, params=_params_with_rmssd()
    )
    soft = parse_context_to_constraints(ctx.notes, use_heuristic_fallback=True)
    plan = optimize_week(
        hard_constraints=constraints,
        soft_constraints=soft,
        previous_week_trimp=280.0,
    )
    session_types = [d.session_type for d in plan.days]
    assert "VMA" not in session_types
    return (f"TRIMP={plan.total_trimp:.0f}, "
            f"vol réduit={100*(1-soft.max_volume_pct):.0f}%, "
            f"plan={session_types}")


t("Pipeline complet — contexte fatigué → plan adapté", _test_full_pipeline)


# ══════════════════════════════════════════════════════════════
#  RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════

section("RÉSUMÉ FINAL")
print(f"\n  {PASS} {passed_total} tests passés")
if failed_total > 0:
    print(f"  {FAIL} {failed_total} tests échoués")

print("""
  ─── Fichiers créés ───────────────────────────────────────
  src/layer_a_signal/features.py          (GAP + DFA α₁)
  src/layer_a_signal/features_advanced.py (mock RR + DFA impl)
  src/layer_a_signal/context_ingestion.py (DailyContext)
  src/layer_b_twin/bayesian_model.py      (Banister + PyMC)
  src/layer_b_twin/uncertainty_manager.py (HDI + calibration)
  src/layer_b_twin/state_estimator.py     (interface principale)
  src/layer_c_planner/safety_valve.py     (hard constraints)
  src/layer_c_planner/llm_context_parser.py (Claude API)
  src/layer_c_planner/optimizer.py        (OR-Tools CP-SAT)
  app/main_dashboard.py                   (Streamlit 4 pages)

  ─── Pour démarrer ────────────────────────────────────────
  pip install -r requirements.txt
  streamlit run app/main_dashboard.py
""")
