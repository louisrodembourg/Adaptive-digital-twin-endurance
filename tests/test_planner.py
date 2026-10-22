"""
Planificateur : ce qu'il optimise, et ce qu'il refuse de promettre.

`layer_c` n'etait importe par rien — ni test, ni tableau de bord, ni
commande — donc jamais execute. Ce fichier le couvre, et verrouille d'abord le
changement de fond : l'objectif historique maximisait la specificite sous un
plafond de volume, ce qui prescrit « entraine-toi autant que la soupape
l'autorise ». La mesure dit qu'on ne connait pas la dose-reponse de l'athlete
(quatre repondeurs sur treize, aucune variable connue ne les separe), et sous
cette incertitude le cout d'une charge inutile est reel la ou son benefice ne
l'est pas. On cherche donc la charge la plus faible qui delivre le stimulus.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import load_params
from src.layer_b_twin.load_model import compute_load_curves_by_sport
from src.layer_c_planner.dose_response import (
    ABSENT,
    DETECTED,
    UNDETERMINED,
    DoseResponse,
    assess_dose_response,
)
from src.layer_c_planner.optimizer import (
    SESSION_TRIMP_BASE,
    StimulusTarget,
    WeeklyPlan,
    _chronic_daily_load,
    optimize_week,
)
from src.layer_c_planner.safety_valve import check_safety_constraints

MONDAY = date(2026, 10, 26)


def _daily(n=400, start="2025-01-01", bike=60.0, run=30.0, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D"),
        "trimp_bike": rng.gamma(2.0, bike / 2, n),
        "trimp_run": rng.gamma(2.0, run / 2, n),
        "trimp_other": np.zeros(n),
    })
    return frame.assign(trimp=frame.trimp_bike + frame.trimp_run + frame.trimp_other)


def _plan(**kwargs):
    kwargs.setdefault("previous_week_trimp", 400.0)
    kwargs.setdefault("week_start", MONDAY)
    return optimize_week(check_safety_constraints(), **kwargs)


# ── l'objectif ───────────────────────────────────────────────────────────

def test_minimising_load_prescribes_less_than_maximising_specificity():
    """
    Le changement de fond, en une assertion. Pour les memes contraintes et le
    meme stimulus exige, chercher la charge minimale doit prescrire moins que
    maximiser la specificite — sinon le nouvel objectif ne sert a rien.
    """
    daily = _daily()
    minimal = _plan(daily_load=daily, objective="min_load")
    maximal = _plan(daily_load=daily, objective="max_specificity")
    assert minimal.total_trimp < maximal.total_trimp


def test_minimising_load_still_delivers_the_demanded_stimulus():
    """Moins de charge, mais pas moins de seances cles : c'est le contrat."""
    plan = _plan(daily_load=_daily(), stimulus=StimulusTarget(min_key_sessions=2))
    key = [d for d in plan.days
           if d.session_type in ("VMA", "THRESHOLD", "SPECIFIC")]
    assert len(key) >= 2


def test_an_unknown_objective_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="objectif inconnu"):
        _plan(objective="maximiser_le_plaisir")


def test_the_retention_floor_prevents_prescribing_detraining():
    """
    Minimiser la charge sans plancher donnerait deux seances cles et cinq
    jours de repos : un minimum litteral et une regression physiologique.
    """
    retention = load_params()["planner"]["min_weekly_retention_pct"] / 100
    previous = 400.0
    plan = _plan(previous_week_trimp=previous, daily_load=_daily())
    assert plan.total_trimp >= previous * retention - 1e-6


def test_an_explicit_floor_overrides_the_configured_one():
    plan = _plan(daily_load=_daily(),
                 stimulus=StimulusTarget(min_total_trimp=120.0))
    assert plan.total_trimp >= 120.0
    assert plan.total_trimp < 400.0 * 0.8


# ── la borne aigu/chronique, enfin branchee ──────────────────────────────

def test_the_acute_chronic_bound_caps_the_week():
    """
    `acute_chronic_ratio` existait dans load_model.py, sa docstring la
    presentait comme « la seule regle de securite reellement declenchable »,
    et elle n'etait branchee sur rien.
    """
    chronic = 10.0
    low = pd.DataFrame({"date": pd.date_range("2026-09-01", periods=40, freq="D"),
                        "trimp": chronic})
    upper = load_params()["planner"]["acwr_upper_bound"]
    plan = _plan(previous_week_trimp=400.0, daily_load=low)
    assert plan.total_trimp <= 7 * upper * chronic + 1e-6


def test_a_collapsed_chronic_load_lowers_the_floor_instead_of_zeroing_the_week():
    """
    Quand le plancher depasse le plafond de securite, l'abandonner ferait
    rendre un plan a zero TRIMP — le minimum litteral, et un conseil inutile.
    On rabaisse le plancher : « le maximum que tu peux faire sans danger ».
    """
    low = pd.DataFrame({"date": pd.date_range("2026-09-01", periods=40, freq="D"),
                        "trimp": 10.0})
    plan = _plan(previous_week_trimp=400.0, daily_load=low)
    assert plan.total_trimp > 0
    assert any("rabaisse" in note for note in plan.notes)


def test_without_a_load_history_the_bound_declares_itself_inactive():
    """Une borne silencieusement inactive est pire qu'une borne absente."""
    plan = _plan()
    assert any("aigu/chronique inactive" in note for note in plan.notes)


def test_a_chronic_mean_needs_a_full_window():
    """Une moyenne sur quinze jours appelee « chronique » serait un abus."""
    short = pd.DataFrame({"date": pd.date_range("2026-09-01", periods=15, freq="D"),
                          "trimp": 50.0})
    assert _chronic_daily_load(short) is None
    full = pd.DataFrame({"date": pd.date_range("2026-09-01", periods=28, freq="D"),
                         "trimp": 50.0})
    assert _chronic_daily_load(full) == pytest.approx(50.0)
    assert _chronic_daily_load(None) is None
    assert _chronic_daily_load(pd.DataFrame({"date": [], "trimp": []})) is None


# ── monotonie et bornes par type ─────────────────────────────────────────

def test_a_test_session_is_not_prescribed_four_times():
    """
    Sans borne par type, l'optimiseur empilait le type le moins couteux qui
    satisfaisait ses contraintes : quatre CALIBRATION, qui est un TEST.
    """
    caps = load_params()["planner"]["max_per_type_per_week"]
    plan = _plan(daily_load=_daily())
    counts = pd.Series([d.session_type for d in plan.days]).value_counts()
    for session_type, cap in caps.items():
        assert counts.get(session_type, 0) <= cap, session_type


def test_no_single_type_fills_the_week():
    plan = _plan(daily_load=_daily())
    counts = pd.Series([d.session_type for d in plan.days
                        if d.session_type != "REST"]).value_counts()
    default_cap = load_params()["planner"]["max_same_type_per_week"]
    assert counts.max() <= default_cap


def test_every_window_holds_a_light_day():
    """
    Le contenu pratique du controle de monotonie : la monotonie de Foster est
    un rapport moyenne/ecart-type, non lineaire dans les variables de
    decision, donc on exige une journee legere dans chaque fenetre glissante.
    """
    span = 3
    plan = _plan(daily_load=_daily(),
                 stimulus=StimulusTarget(max_consecutive_training_days=span))
    light = [d.session_type in ("REST", "RECOVERY") for d in plan.days]
    for first in range(7 - span):
        assert any(light[first:first + span + 1]), first


def test_at_least_one_full_rest_day():
    plan = _plan(daily_load=_daily())
    assert any(d.session_type == "REST" for d in plan.days)


def test_two_hard_sessions_never_follow_each_other():
    plan = _plan(daily_load=_daily())
    hard = [d.session_type in ("VMA", "THRESHOLD", "SPECIFIC") for d in plan.days]
    assert not any(hard[j] and hard[j + 1] for j in range(6))


def test_the_weekly_progression_is_bounded():
    previous = 100.0
    progression = 1 + load_params()["planner"]["max_weekly_progression_pct"] / 100
    plan = _plan(previous_week_trimp=previous, daily_load=_daily())
    assert plan.total_trimp <= previous * progression + 1e-6


# ── ce que le plan refuse de promettre ───────────────────────────────────

def test_a_plan_without_a_verdict_promises_nothing():
    statement = _plan(daily_load=_daily()).objective_statement()
    assert "Aucune mesure de dose-réponse" in statement
    assert "n'optimise donc aucune" in statement


def test_an_absent_dose_response_is_stated_in_the_plan():
    """
    Le plan doit dire qu'il n'optimise aucune performance predite. Sans cette
    phrase, un plan se lit comme une promesse de progression.
    """
    verdict = DoseResponse(ABSENT, skill=-0.05, ci_low=-0.4, ci_high=0.3, folds=20)
    plan = _plan(daily_load=_daily(), dose_response=verdict)
    statement = plan.objective_statement()
    assert "non detectee" in statement
    assert "ne peut pas en promettre" in statement
    assert "[-0.400 · +0.300]" in statement


def test_a_detected_dose_response_is_not_a_licence_to_maximise():
    verdict = DoseResponse(DETECTED, skill=0.34, ci_low=0.06, ci_high=0.53, folds=38)
    statement = _plan(daily_load=_daily(), dose_response=verdict).objective_statement()
    assert "démontré" in statement
    assert "ne le maximise pas" in statement


def test_the_report_lists_what_could_not_be_applied():
    report = _plan().xai_report()
    assert "n'a pas pu appliquer" in report


def test_an_infeasible_stimulus_is_announced_not_swallowed():
    """
    La version precedente retombait silencieusement sur l'heuristique des
    qu'une contrainte rendait le probleme infaisable : un plan degrade se
    lisait alors comme un plan optimal.
    """
    plan = _plan(daily_load=_daily(),
                 stimulus=StimulusTarget(min_key_sessions=7, key_specificity=1.0))
    assert any("Stimulus specifique abandonne" in note for note in plan.notes)


def test_the_plan_carries_its_objective():
    assert _plan(daily_load=_daily()).objective == "min_load"
    assert _plan(daily_load=_daily(),
                 objective="max_specificity").objective == "max_specificity"


# ── verdict de dose-reponse ──────────────────────────────────────────────

def _responder(daily, strength=4.0, seed=1):
    """Un athlete chez qui la charge explique vraiment la performance."""
    curves = compute_load_curves_by_sport(daily, 42.0, 12.0, ("trimp_bike",))
    signal = (curves.fitness_bike - curves.fatigue_bike).to_numpy()
    signal = (signal - signal.mean()) / (signal.std() or 1.0)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": daily["date"],
        "pi": 250.0 + strength * 10.0 * signal + rng.normal(0, 2.0, len(daily)),
    })


def test_a_real_effect_is_detected():
    daily = _daily(n=700)
    verdict = assess_dose_response(_responder(daily), daily)
    assert verdict.verdict == DETECTED
    assert verdict.may_claim_performance_effect
    assert verdict.ci_low > 0


def test_pure_noise_is_not_detected():
    daily = _daily(n=700)
    rng = np.random.default_rng(7)
    noise = pd.DataFrame({"date": daily["date"],
                          "pi": 250.0 + rng.normal(0, 10.0, len(daily))})
    verdict = assess_dose_response(noise, daily)
    assert verdict.verdict == ABSENT
    assert not verdict.may_claim_performance_effect


def test_the_interval_decides_and_not_the_point_estimate():
    """
    Un gain ponctuel positif dont l'intervalle contient zero ne vaut pas mieux
    qu'un gain nul : c'est la regle du depot, et elle vaut ici aussi.
    """
    verdict = DoseResponse(ABSENT, skill=0.25, ci_low=-0.05, ci_high=0.5, folds=12)
    assert verdict.skill > 0
    assert not verdict.may_claim_performance_effect


@pytest.mark.parametrize(("observations", "fragment"), [
    (pd.DataFrame(), "aucune observation"),
    (pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "pi": [250.0]}),
     "il en faut au moins trois"),
])
def test_missing_data_is_undetermined_and_not_absent(observations, fragment):
    """
    « Non detectee » et « indeterminable » ne disent pas la meme chose :
    presenter une absence de mesure comme une absence d'effet est exactement
    l'erreur que ce depot s'emploie a ne pas commettre.
    """
    verdict = assess_dose_response(observations, _daily())
    assert verdict.verdict == UNDETERMINED
    assert fragment in verdict.reason
    assert not verdict.may_claim_performance_effect


def test_a_short_history_is_undetermined():
    daily = _daily(n=200)
    verdict = assess_dose_response(_responder(daily), daily)
    assert verdict.verdict == UNDETERMINED
    assert "plis" in verdict.reason


def test_a_missing_load_column_says_what_to_do():
    daily = _daily()[["date", "trimp"]]
    verdict = assess_dose_response(pd.DataFrame({"date": daily["date"], "pi": 250.0}),
                                   daily)
    assert verdict.verdict == UNDETERMINED
    assert "cli ingest" in verdict.reason


def test_an_unknown_load_channel_is_undetermined():
    daily = _daily()
    verdict = assess_dose_response(_responder(daily), daily, load_column="trimp_other")
    assert verdict.verdict == UNDETERMINED


def test_an_undetermined_verdict_describes_its_reason():
    verdict = assess_dose_response(pd.DataFrame(), _daily())
    assert "indeterminable" in verdict.describe()
    assert verdict.reason in verdict.describe()


# ── la cle de configuration qui n'etait jamais lue ───────────────────────

def test_the_configured_hrv_threshold_is_actually_read():
    """
    `safety_valve` lisait `planner.rmssd_vma_threshold`, qui n'existe pas dans
    params.yaml : la valeur configuree sous `planner.hrv.rmssd_threshold_ms`
    etait ignoree au profit d'un defaut code en dur, alors que le commentaire
    du fichier la presentait comme « seule source de verite pour ce seuil ».
    """
    params = {"planner": {"hrv": {"rmssd_threshold_ms": 90.0}}}
    constraints = check_safety_constraints(rmssd_ms=80.0, params=params)
    assert any(c.code == "NO_HIGH_INTENSITY" for c in constraints)

    # Sous le seuil configure, aucune contrainte HRV ne doit se declencher.
    permissive = {"planner": {"hrv": {"rmssd_threshold_ms": 10.0}}}
    assert not any(c.code == "NO_HIGH_INTENSITY"
                   for c in check_safety_constraints(rmssd_ms=80.0, params=permissive))


def test_a_plan_is_a_weekly_plan_of_seven_days():
    plan = _plan(daily_load=_daily())
    assert isinstance(plan, WeeklyPlan)
    assert len(plan.days) == 7
    assert plan.total_trimp == pytest.approx(
        sum(SESSION_TRIMP_BASE[d.session_type] for d in plan.days))


def test_a_declared_unavailability_is_honoured_not_suggested():
    """
    `avoid_days` vient de l'athlete — « je ne peux pas m'entrainer ce jour-la ».
    Pose en `AddHint`, il etait ignorable par le solveur, et une
    indisponibilite declaree se retrouvait planifiee.
    """
    from src.layer_c_planner.llm_context_parser import SoftConstraints

    plan = optimize_week(
        check_safety_constraints(),
        soft_constraints=SoftConstraints(avoid_days=[1, 3]),
        previous_week_trimp=250.0, week_start=MONDAY, daily_load=_daily(),
    )
    assert plan.days[1].session_type == "REST"
    assert plan.days[3].session_type == "REST"


def test_availability_outranks_the_prescribed_stimulus():
    """
    Quand les deux s'opposent, c'est le stimulus qui cede : l'athlete connait
    son emploi du temps, le planificateur non.
    """
    from src.layer_c_planner.llm_context_parser import SoftConstraints

    plan = optimize_week(
        check_safety_constraints(),
        soft_constraints=SoftConstraints(avoid_days=[0, 1, 2, 3, 4, 5]),
        previous_week_trimp=250.0, week_start=MONDAY, daily_load=_daily(),
        stimulus=StimulusTarget(min_key_sessions=3),
    )
    assert sum(d.session_type == "REST" for d in plan.days) >= 6
