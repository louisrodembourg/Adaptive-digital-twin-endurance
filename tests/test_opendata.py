"""Client OpenData : lecture d'archive, criblage, cache. Tout hors ligne."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.ingestion.opendata import (
    OpenDataClient,
    OpenDataError,
    RemoteAthlete,
    ScreeningCriteria,
    evaluate_athlete,
    list_activities,
    parse_summary,
    read_activity,
    read_summary,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_athlete.zip"


@pytest.fixture
def summary():
    return read_summary(FIXTURE)


def test_summary_decodes_athlete_and_rides(summary):
    assert summary.gender == "M"
    assert summary.year_of_birth == 1985
    assert len(summary.rides) == 6
    assert len(summary.runs) == 6
    assert summary.hr_coverage == 1.0


def test_channel_code_detects_heart_rate_without_opening_csv(summary):
    # Le champ `data` du JSON encode les canaux presents : le lire evite de
    # decompresser un CSV par activite pour qualifier un athlete.
    assert all(r.has_heartrate for r in summary.rides)
    assert not parse_summary(
        {"RIDES": [{"filename": "x.csv", "sport": "Run", "data": "TDS----------"}]}, "u"
    ).runs_with_hr


def test_span_and_sport_counts(summary):
    assert summary.span_days > 120
    assert summary.sport_counts == {"Run": 6}


def test_metric_ignores_missing_and_nan(summary):
    ride = summary.rides[0]
    assert ride.metric("trimp_points") == pytest.approx(45.0)
    assert ride.metric("inexistant") is None


def test_dates_survive_the_goldencheetah_format(summary):
    assert all(r.date is not None for r in summary.rides)


def test_read_activity_types_columns_and_keeps_gaps():
    names = list_activities(FIXTURE)
    assert len(names) == 6
    rows = read_activity(FIXTURE, names[0])
    assert len(rows) == 600
    assert set(rows[0]) == {"secs", "km", "power", "hr", "cad", "alt"}
    assert all(isinstance(r["secs"], float) for r in rows)


def test_gapped_activity_has_a_real_hole_in_secs():
    # La fixture porte une pause de 120 s : les metriques ne doivent jamais
    # confondre nombre de lignes et duree ecoulee.
    rows = read_activity(FIXTURE, list_activities(FIXTURE)[-1])
    deltas = [rows[i + 1]["secs"] - rows[i]["secs"] for i in range(len(rows) - 1)]
    assert max(deltas) >= 120


def test_missing_activity_is_reported_clearly():
    with pytest.raises(OpenDataError, match="absente"):
        read_activity(FIXTURE, "jamais_vue.csv")


def test_archive_without_summary_is_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("only.csv", "secs,km\n0,0\n")
    with pytest.raises(OpenDataError, match="resume JSON"):
        read_summary(bad)


def test_malformed_rides_are_skipped_not_fatal():
    s = parse_summary({"ATHLETE": {}, "RIDES": [None, "junk", {"filename": "a.csv"}]}, "u")
    assert len(s.rides) == 1


# -- criblage --------------------------------------------------------------

def _remote() -> RemoteAthlete:
    return RemoteAthlete(file_id="f", name="u.zip", size_bytes=1000, md5="")


def test_screening_accepts_a_qualifying_athlete(summary):
    crit = ScreeningCriteria(min_runs=5, min_span_days=100, min_hr_coverage=0.9)
    assert evaluate_athlete(_remote(), summary, crit).passed


@pytest.mark.parametrize(
    "crit,expected",
    [
        (ScreeningCriteria(min_runs=500), "courses"),
        (ScreeningCriteria(min_runs=5, min_span_days=10_000), "historique"),
        (ScreeningCriteria(min_runs=5, min_span_days=100, min_hr_coverage=1.01), "FC"),
        (ScreeningCriteria(min_runs=5, min_span_days=100, min_hr_coverage=0.9,
                           max_elevation_m_per_km=0.1), "denivele"),
    ],
)
def test_screening_rejects_and_says_why(summary, crit, expected):
    result = evaluate_athlete(_remote(), summary, crit)
    assert not result.passed
    assert expected in result.rejected_for


# -- cache -----------------------------------------------------------------

def test_cache_hit_avoids_the_network(tmp_path):
    client = OpenDataClient(cache_dir=tmp_path, session=_ExplodingSession())
    remote = RemoteAthlete(file_id="f", name="u.zip", size_bytes=0, md5="")
    cached = client.cached_path("u")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(FIXTURE.read_bytes())
    assert client.fetch_athlete(remote) == cached


def test_listing_failure_mentions_the_closed_s3_bucket(tmp_path):
    client = OpenDataClient(cache_dir=tmp_path, session=_ExplodingSession())
    with pytest.raises(OpenDataError, match="S3"):
        list(client.list_athletes(limit=1))


class _ExplodingSession:
    """Toute sortie reseau est une erreur de test."""

    def get(self, *a, **k):
        import requests

        raise requests.RequestException("le reseau ne doit pas etre sollicite")
