"""
Provenance : empecher structurellement qu'une donnee inventee passe pour
mesuree.

Le tableau de bord enregistrait quatre-vingt-quatre seances generees avec
l'etiquette "fit_file", parce que `add_session` avait un defaut et que
l'appelant l'omettait.
"""

from __future__ import annotations

import pytest

from src.pipeline.provenance import (
    DataSource,
    ProvenanceError,
    check_writable,
    coerce_source,
    summarize,
)


def test_only_synthetic_is_not_real():
    assert not DataSource.SYNTHETIC.is_real
    for s in (DataSource.OPENDATA_OSF, DataSource.FIT_FILE, DataSource.MANUAL):
        assert s.is_real


def test_unknown_source_is_rejected_with_the_valid_list():
    with pytest.raises(ProvenanceError, match="opendata_osf"):
        coerce_source("gut_feeling")


def test_synthetic_needs_explicit_consent():
    with pytest.raises(ProvenanceError, match="allow_synthetic"):
        check_writable(DataSource.SYNTHETIC, allow_synthetic=False)
    assert check_writable(DataSource.SYNTHETIC, allow_synthetic=True) is DataSource.SYNTHETIC


def test_real_sources_never_need_consent():
    assert check_writable(DataSource.OPENDATA_OSF, allow_synthetic=False)


def test_summary_counts_and_flags_contamination():
    clean = summarize(["opendata_osf"] * 3)
    assert clean.is_trustworthy and clean.n_real == 3

    tainted = summarize(["opendata_osf"] * 3 + ["synthetic"])
    assert not tainted.is_trustworthy
    assert "SYNTHETIQUE" in tainted.banner


def test_empty_store_says_what_to_do():
    empty = summarize([])
    assert not empty.is_trustworthy
    assert "ingest" in empty.banner


def test_banner_names_the_real_source():
    assert "GoldenCheetah" in summarize(["opendata_osf"] * 2).banner
