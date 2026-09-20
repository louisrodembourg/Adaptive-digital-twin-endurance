"""Configuration : cache, gardes de calibration, derivations."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import ConfigError, TRIMP_COEFFS, clear_config_cache, load_config


def test_config_is_cached():
    clear_config_cache()
    assert load_config() is load_config()


def test_trimp_coefficients_derive_from_sex():
    # Le YAML stockait b et c sous des cles interverties, et features.py les
    # relisait interverties aussi : deux erreurs qui s'annulaient. Les deriver
    # d'une seule source rend l'inversion impossible a reintroduire.
    cfg = load_config()
    assert cfg.athlete.trimp_coeffs == TRIMP_COEFFS[cfg.athlete.sex]
    assert replace(cfg.athlete, sex="male").trimp_coeffs == (0.64, 1.92)
    assert replace(cfg.athlete, sex="female").trimp_coeffs == (0.86, 1.67)


def test_unknown_sex_is_rejected():
    with pytest.raises(ConfigError):
        replace(load_config().athlete, sex="martien").trimp_coeffs


def test_uncalibrated_athlete_refuses_to_answer():
    # hr_max/hr_rest valent null tant que `cli calibrate` n'a pas tourne :
    # mieux vaut une erreur explicite qu'un TRIMP calcule sur 205/65 inventes.
    athlete = replace(load_config().athlete, hr_max=None)
    with pytest.raises(ConfigError, match="hr_max"):
        athlete.hr_reserve


def test_hr_reserve_requires_ordered_bounds():
    athlete = replace(load_config().athlete, hr_max=100, hr_rest=120)
    with pytest.raises(ConfigError):
        athlete.hr_reserve


def test_target_time_derived_from_improvement():
    goal = replace(load_config().goal, reference_time_s=4920.0, target_time_s=None,
                   improvement_pct=5.0)
    assert goal.resolved_target_time_s == pytest.approx(4674.0)
    # Reproduit l'ancien objectif code en dur (1h22 -> 1h18) sans le coder en dur.
    assert goal.resolved_target_time_s / 60 == pytest.approx(77.9, abs=0.1)


def test_explicit_target_time_wins_over_improvement():
    goal = replace(load_config().goal, reference_time_s=4920.0, target_time_s=4680.0)
    assert goal.resolved_target_time_s == 4680.0
    assert goal.target_speed_ms == pytest.approx(21097.5 / 4680.0)


def test_goal_without_reference_refuses_to_guess():
    goal = replace(load_config().goal, reference_time_s=None, target_time_s=None)
    with pytest.raises(ConfigError, match="reference_time_s"):
        goal.resolved_target_time_s


def test_with_paths_reroots_without_mutating_original(cfg, tmp_path):
    assert cfg.paths.processed == tmp_path / "processed"
    assert load_config().paths.processed != cfg.paths.processed


def test_single_source_of_truth_for_rmssd_threshold():
    # Le seuil existait en trois exemplaires : YAML a 0.0 (qui desactivait la
    # regle), DailyContext a 50.0, dashboard a 50. Une seule valeur desormais.
    hrv = load_config().planner["hrv"]
    assert hrv["rmssd_threshold_ms"] == 50
    assert hrv["enabled"] is False  # aucune source RR disponible
