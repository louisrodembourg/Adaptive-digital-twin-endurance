"""
Harnais de tests.

Deux garanties, parce qu'une seule ne suffit pas : les chemins sont reracines
sur tmp_path (structurel), ET une garde verifie apres chaque test que rien n'a
bouge dans le vrai data/ (filet). Le depot s'est deja fait polluer exactement
comme ca : test_pipeline_phase2_to_5.py instanciait un StateEstimator nu, dont
les chemins etaient des constantes de classe, et ecrivait dans data/processed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, clear_config_cache, load_config

REAL_DATA_DIR = PROJECT_ROOT / "data"


def _snapshot(root: Path) -> dict[str, float]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture(autouse=True)
def guard_real_data_dir():
    """Echoue le test qui touche au vrai data/, quelle qu'en soit la raison."""
    before = _snapshot(REAL_DATA_DIR)
    yield
    after = _snapshot(REAL_DATA_DIR)
    if before != after:
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
        pytest.fail(
            "Ce test a ecrit dans le vrai repertoire data/ :\n"
            f"  ajoutes   : {added}\n"
            f"  supprimes : {removed}\n"
            f"  modifies  : {changed}\n"
            "Injecter des chemins temporaires (fixture `cfg`) au lieu des chemins par defaut."
        )


@pytest.fixture
def cfg(tmp_path):
    """Configuration du projet reracinee sur un repertoire temporaire."""
    clear_config_cache()
    c = load_config().with_paths(tmp_path)
    c.paths.ensure()
    return c
