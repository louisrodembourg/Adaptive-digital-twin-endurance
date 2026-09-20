"""TRIMP : verrouillage des coefficients de Banister sur valeur d'or."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


def _reference_trimp(hr: np.ndarray, hr_rest: float, hr_max: float,
                     b: float, c: float) -> float:
    """Banister, ecrit independamment du code de production."""
    d = np.clip((hr - hr_rest) / (hr_max - hr_rest), 0.0, 1.0)
    return float(np.sum((1 / 60) * d * b * np.exp(c * d)))


@pytest.fixture
def hr_series():
    # 600 s a FC croissante : couvre toute la plage des zones.
    return np.linspace(100.0, 190.0, 600)


def test_trimp_matches_independent_implementation(monkeypatch, hr_series):
    from src.layer_a_signal import features

    cfg = load_config()
    cfg = replace(cfg, athlete=replace(cfg.athlete, hr_max=190.0, hr_rest=50.0))
    monkeypatch.setattr(features, "load_config", lambda *a, **k: cfg)

    got = features.compute_trimp(pd.DataFrame({"hr": hr_series}))
    want = _reference_trimp(hr_series, 50.0, 190.0, 0.64, 1.92)
    assert got == pytest.approx(want, rel=1e-9)


def test_trimp_golden_value(monkeypatch, hr_series):
    """
    Valeur d'or figee. Si ce nombre bouge, les coefficients ont ete touches.

    Les cles YAML trimp_gender_factor / trimp_gender_exp portaient des noms
    inverses par rapport a leur role, et features.py les relisait croisees :
    l'arithmetique tombait juste par double negation. Corriger un seul des
    deux cotes aurait casse le TRIMP en silence.
    """
    from src.layer_a_signal import features

    cfg = load_config()
    cfg = replace(cfg, athlete=replace(cfg.athlete, hr_max=190.0, hr_rest=50.0))
    monkeypatch.setattr(features, "load_config", lambda *a, **k: cfg)

    assert features.compute_trimp(pd.DataFrame({"hr": hr_series})) == pytest.approx(
        18.641217, abs=1e-5
    )


def test_female_coefficients_give_a_different_load(monkeypatch, hr_series):
    from src.layer_a_signal import features

    base = load_config()
    loads = {}
    for sex in ("male", "female"):
        cfg = replace(base, athlete=replace(base.athlete, sex=sex,
                                            hr_max=190.0, hr_rest=50.0))
        monkeypatch.setattr(features, "load_config", lambda *a, c=cfg, **k: c)
        loads[sex] = features.compute_trimp(pd.DataFrame({"hr": hr_series}))
    assert not math.isclose(loads["male"], loads["female"], rel_tol=1e-3)


def test_trimp_is_zero_without_heart_rate():
    from src.layer_a_signal import features

    assert features.compute_trimp(pd.DataFrame({"hr": []})) == 0.0
    assert np.isnan(features.compute_trimp(pd.DataFrame({"speed": [1.0]})))
