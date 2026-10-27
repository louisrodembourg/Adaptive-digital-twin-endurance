"""
L'instantane de demo est versionne et servi en ligne : c'est la seule donnee
du depot qu'un visiteur voit sans rien lancer. Il doit donc tenir la regle
que le tableau de bord a longtemps violee — rien de genere ne peut y passer
pour mesure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_ENV, clear_config_cache, load_config

DEMO = Path(__file__).resolve().parents[1] / "demo" / "processed"


def test_l_instantane_ne_contient_que_des_seances_publiques_reelles():
    sessions = pd.read_parquet(DEMO / "sessions.parquet")
    assert len(sessions) > 0
    assert set(sessions["source"]) == {"opendata_osf"}

    calibration = json.loads((DEMO / "athlete_calibration.json").read_text())
    assert calibration["provenance"]["n_synthetic"] == 0
    assert calibration["provenance"]["is_trustworthy"]


def test_la_variable_d_environnement_remplace_le_dossier_processed(monkeypatch):
    monkeypatch.setenv(PROCESSED_ENV, str(DEMO))
    clear_config_cache()
    try:
        assert load_config().paths.processed == DEMO
    finally:
        monkeypatch.delenv(PROCESSED_ENV)
        clear_config_cache()
    assert load_config().paths.processed != DEMO
