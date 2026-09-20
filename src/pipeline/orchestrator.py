"""
Orchestrateur : de la seance brute aux entrees du modele.

C'est le module qui manquait. `features.py` et `ingest_fit.py` n'etaient
importes par aucun fichier de `src/` ni de `app/` : rien ne transformait une
seance en charge quotidienne, et Layer B se nourrissait d'un historique
invente par le tableau de bord lui-meme.

Cinq etages, testables separement :

    ingest -> clean -> featurize -> build_daily_load
                                 -> build_perf_observations

Trois sorties, toutes horodatees et tracees :
  sessions.parquet          une ligne par seance, avec sa provenance
  daily_load.parquet        charge quotidienne sans trou, zeros les jours off
  perf_observations.parquet les efforts assez maximaux pour servir de mesure
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, ConfigError, load_config, load_params
from ..ingestion.opendata import (
    OpenDataClient,
    OpenDataError,
    RemoteAthlete,
    list_activities,
    read_activity,
    read_summary,
    resolve_ride_files,
)
from ..layer_a_signal import features as fx
from ..layer_a_signal.performance import (
    ReferenceCurve,
    fit_reference_curve,
    mean_max_speed,
)
from .provenance import DataSource, ProvenanceSummary, summarize

log = logging.getLogger(__name__)

SESSIONS_FILE = "sessions.parquet"
DAILY_LOAD_FILE = "daily_load.parquet"
PERF_OBS_FILE = "perf_observations.parquet"
CALIBRATION_FILE = "athlete_calibration.json"

# Fenetres de puissance critique reprises du resume, en minutes.
CRITICAL_POWER_WINDOWS: tuple[int, ...] = (5, 10, 20, 30, 60)


# ── specification de source ──────────────────────────────────────────────

@dataclass(frozen=True)
class OpenDataSpec:
    """Un athlete du jeu OpenData, identifie par son UUID."""

    athlete_uuid: str
    source: DataSource = DataSource.OPENDATA_OSF


@dataclass(frozen=True)
class LocalZipSpec:
    """Une archive deja sur disque — ce dont les tests se servent."""

    path: Path
    source: DataSource = DataSource.OPENDATA_OSF


@dataclass(frozen=True)
class FitDirSpec:
    """Un repertoire de fichiers .fit."""

    path: Path
    source: DataSource = DataSource.FIT_FILE


SourceSpec = OpenDataSpec | LocalZipSpec | FitDirSpec


# ── structures de donnees ─────────────────────────────────────────────────

@dataclass(frozen=True)
class RawSession:
    session_id: str
    athlete_id: str
    start_time: datetime
    sport: str
    source: DataSource
    source_ref: str
    records: pd.DataFrame = field(repr=False)
    summary: dict = field(repr=False, default_factory=dict)

    @property
    def is_run(self) -> bool:
        return self.sport.lower().startswith("run")


@dataclass
class IngestReport:
    n_new: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        line = f"{self.n_new} nouvelle(s), {self.n_skipped} deja presente(s), {self.n_failed} en echec"
        if self.failures:
            line += "\n" + "\n".join(f"    {ref} : {why}" for ref, why in self.failures[:5])
            if len(self.failures) > 5:
                line += f"\n    ... et {len(self.failures) - 5} autre(s)"
        return line


def make_session_id(athlete_id: str, start_time: datetime) -> str:
    """Identifiant stable : permet de reprendre une ingestion interrompue."""
    raw = f"{athlete_id}|{start_time.isoformat()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ── etage 1 : ingestion ───────────────────────────────────────────────────

def ingest(spec: SourceSpec, cfg: Config | None = None) -> Iterator[RawSession]:
    """Produit les seances brutes d'une source, sans rien calculer."""
    cfg = cfg or load_config()
    if isinstance(spec, OpenDataSpec):
        client = OpenDataClient(cache_dir=cfg.paths.cache)
        path = _locate_archive(client, spec.athlete_uuid, cfg)
        yield from _ingest_zip(path, spec.source, spec.athlete_uuid)
    elif isinstance(spec, LocalZipSpec):
        yield from _ingest_zip(Path(spec.path), spec.source, Path(spec.path).stem)
    elif isinstance(spec, FitDirSpec):
        yield from _ingest_fit_dir(Path(spec.path), spec.source)
    else:
        raise TypeError(f"Specification de source inconnue : {spec!r}")


def _locate_archive(client: OpenDataClient, uuid: str, cfg: Config) -> Path:
    cached = client.cached_path(uuid)
    if cached.exists():
        return cached
    entry = _manifest_entry(cfg, uuid)
    if entry is None:
        raise OpenDataError(
            f"Athlete {uuid} absent du cache et du manifeste. L'ajouter a "
            f"{cfg.data_source.manifest} ou lancer `cli screen-athletes`."
        )
    remote = RemoteAthlete(
        file_id=entry["file_id"], name=f"{uuid}.zip",
        size_bytes=int(entry.get("size_mb", 0) * 1e6), md5=entry.get("md5", ""),
    )
    return client.fetch_athlete(remote)


def _manifest_entry(cfg: Config, uuid: str) -> dict | None:
    path = cfg.paths.root.parent / cfg.data_source.manifest
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return next((a for a in data.get("athletes", []) if a["uuid"] == uuid), None)


def _ingest_zip(path: Path, source: DataSource, athlete_id: str) -> Iterator[RawSession]:
    summary = read_summary(path)
    names = list_activities(path)
    matched = resolve_ride_files(summary, names)

    for idx, ride in enumerate(summary.rides):
        filename = matched.get(idx)
        if filename is None or ride.date is None:
            continue
        try:
            rows = read_activity(path, filename)
        except OpenDataError as exc:
            log.warning("Activite %s illisible : %s", filename, exc)
            continue
        if not rows:
            continue
        yield RawSession(
            session_id=make_session_id(summary.uuid, ride.date),
            athlete_id=summary.uuid,
            start_time=ride.date,
            sport=ride.sport,
            source=source,
            source_ref=f"{path.name}:{filename}",
            records=_numeric_frame(rows),
            summary=dict(ride.metrics),
        )


def _numeric_frame(rows: list[dict]) -> pd.DataFrame:
    """
    Construit un DataFrame a colonnes numeriques.

    Les valeurs manquantes des CSV arrivent en None ; melangees a des
    flottants, pandas en fait des colonnes de dtype object sur lesquelles
    toute soustraction echoue. On force la conversion une fois ici plutot
    que de s'en remettre a l'inference.
    """
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _ingest_fit_dir(directory: Path, source: DataSource) -> Iterator[RawSession]:
    from ..layer_a_signal.ingest_fit import get_session_summary, parse_fit

    for fit in sorted(Path(directory).glob("*.fit")):
        try:
            records, _ = parse_fit(str(fit))
            meta = get_session_summary(str(fit))
        except Exception as exc:  # le SDK leve des exceptions variees
            log.warning("Fichier %s illisible : %s", fit.name, exc)
            continue
        start = meta.get("start_time") or datetime.fromtimestamp(fit.stat().st_mtime)
        # Les traces GPS ne doivent jamais atteindre les artefacts persistes.
        records = records.drop(columns=[c for c in ("lat", "lon") if c in records.columns])
        yield RawSession(
            session_id=make_session_id(fit.stem, start),
            athlete_id=fit.stem,
            start_time=start,
            sport=str(meta.get("sport", "run")),
            source=source,
            source_ref=fit.name,
            records=records,
            summary=meta,
        )


def estimate_hr_bounds(spec: SourceSpec, cfg: Config) -> tuple[float, float] | None:
    """
    Estime FC max et FC repos depuis les resumes, sans decoder les CSV.

    Le TRIMP a besoin de ces deux bornes, mais elles-memes se deduisent de
    l'historique : sans amorcage la chaine ne peut pas demarrer. GoldenCheetah
    publie max_heartrate et min_heartrate pour chaque activite, ce qui suffit
    a briser la circularite pour un cout negligeable.

    On prend des percentiles plutot que les extremes : une seule valeur
    aberrante fixerait sinon l'echelle de toute la charge d'entrainement.
    """
    archive = _archive_for(spec, cfg)
    if archive is None:
        return None

    summary = read_summary(archive)
    highs = [v for r in summary.rides if (v := r.metric("max_heartrate")) and 100 < v < 230]
    lows = [v for r in summary.rides if (v := r.metric("min_heartrate")) and 25 < v < 120]
    if len(highs) < 10 or len(lows) < 10:
        return None

    hr_max = float(np.percentile(highs, 99))
    hr_rest = float(np.percentile(lows, 1))
    if hr_max - hr_rest < 40:
        return None
    return hr_max, hr_rest


def _archive_for(spec: SourceSpec, cfg: Config) -> Path | None:
    if isinstance(spec, LocalZipSpec):
        return Path(spec.path)
    if isinstance(spec, OpenDataSpec):
        client = OpenDataClient(cache_dir=cfg.paths.cache)
        return _locate_archive(client, spec.athlete_uuid, cfg)
    return None


# ── etage 2 : nettoyage ───────────────────────────────────────────────────

def clean(raw: RawSession, cfg: Config | None = None) -> RawSession:
    """Nettoyage du signal et attache de la base de temps."""
    params = load_params()
    df = _normalise_columns(raw.records)
    cleaned = fx.clean_signal(df, params)
    if "altitude" in cleaned.columns:
        cleaned = fx.grade_adjusted_pace(cleaned)
    return RawSession(
        session_id=raw.session_id, athlete_id=raw.athlete_id,
        start_time=raw.start_time, sport=raw.sport, source=raw.source,
        source_ref=raw.source_ref, records=cleaned, summary=raw.summary,
    )


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amene les colonnes OpenData au vocabulaire de Layer A.

    OpenData fournit une distance cumulee (`km`) et pas de vitesse ; Layer A
    raisonne en `speed_ms`. On derive l'une de l'autre plutot que d'imposer
    un format a la couche d'acces.
    """
    out = df.copy()
    if "speed_ms" not in out.columns and "km" in out.columns and "secs" in out.columns:
        metres = pd.to_numeric(out["km"], errors="coerce").ffill().bfill() * 1000.0
        metres = np.maximum.accumulate(metres.to_numpy(dtype=float))
        secs = pd.to_numeric(out["secs"], errors="coerce").ffill().to_numpy(dtype=float)
        dt = np.diff(secs, prepend=secs[0] - 1.0)
        dt = np.where(dt > 0, dt, np.nan)
        out["speed_ms"] = np.clip(np.diff(metres, prepend=metres[0]) / dt, 0.0, None)
        out["speed_ms"] = pd.Series(out["speed_ms"]).fillna(0.0)
        out["distance_m"] = metres
    if "speed_kmh" not in out.columns and "speed_ms" in out.columns:
        out["speed_kmh"] = out["speed_ms"] * 3.6
    if "altitude" not in out.columns and "alt" in out.columns:
        out["altitude"] = pd.to_numeric(out["alt"], errors="coerce")
    if "cadence" not in out.columns and "cad" in out.columns:
        out["cadence"] = pd.to_numeric(out["cad"], errors="coerce")
    return out


# ── etage 3 : features ────────────────────────────────────────────────────

def featurize(raw: RawSession, cfg: Config | None = None) -> dict:
    """Une ligne de sessions.parquet."""
    params = load_params()
    row: dict = {
        "session_id": raw.session_id,
        "athlete_id": raw.athlete_id,
        "date": pd.Timestamp(raw.start_time).normalize(),
        "start_time": pd.Timestamp(raw.start_time),
        "sport": raw.sport,
        "source": raw.source.value,
        "source_ref": raw.source_ref,
        "is_run": raw.is_run,
    }
    row.update(fx.session_features(raw.records, params, cfg))

    # TRIMP publie par GoldenCheetah : permet de recouper notre calcul contre
    # une implementation independante, ce qui vaut mieux qu'une auto-assertion.
    reference = raw.summary.get("trimp_points")
    row["trimp_reference"] = _as_float(reference)
    # FC minimale telle que l'appareil l'a enregistree. Le minimum du signal
    # nettoye ne convient pas : Savitzky-Golay lisse justement les extremes,
    # et releve ce minimum de plusieurs dizaines de battements.
    row["hr_min_reported"] = _as_float(raw.summary.get("min_heartrate"))

    # Puissances critiques publiees par GoldenCheetah. C'est la mesure de
    # performance du velo, qui represente 80 % de la charge de cet athlete
    # mais n'etait jusqu'ici jamais mesuree. Disponibles pour 417 seances
    # velo sans decoder un seul CSV.
    for window in CRITICAL_POWER_WINDOWS:
        row[f"cp{window}_w"] = _as_float(raw.summary.get(f"{window}m_critical_power"))
    row["elevation_gain_m"] = _as_float(raw.summary.get("elevation_gain"))
    row["distance_km_reported"] = _as_float(raw.summary.get("total_distance"))
    return row


def _as_float(value) -> float:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f if f == f else float("nan")


# ── etage 4 : charge quotidienne ──────────────────────────────────────────

def build_daily_load(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Agrege les seances en charge quotidienne, sur un calendrier sans trou.

    Toutes les disciplines comptent : la charge d'entrainement est systemique,
    et chez les athletes retenus — des triathletes — ne compter que la course
    laisserait une seance tous les six jours, bien trop epars pour un modele a
    convolution exponentielle.

    Les jours sans seance valent zero et doivent exister : le modele les
    parcourt pour faire decroitre fitness et fatigue.
    """
    if sessions.empty:
        return pd.DataFrame(columns=["date", "trimp", "n_sessions", "sources"])

    df = sessions.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["trimp"] = pd.to_numeric(df["trimp"], errors="coerce").fillna(0.0)

    grouped = df.groupby("date").agg(
        trimp=("trimp", "sum"),
        n_sessions=("session_id", "count"),
        sources=("source", lambda s: "|".join(sorted(set(s)))),
    )
    calendar = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    out = grouped.reindex(calendar)
    out["trimp"] = out["trimp"].fillna(0.0)
    out["n_sessions"] = out["n_sessions"].fillna(0).astype(int)
    out["sources"] = out["sources"].fillna("")
    return out.rename_axis("date").reset_index()


# ── etage 5 : observations de performance ─────────────────────────────────

def build_perf_observations(
    sessions: pd.DataFrame,
    mmp_by_session: dict[str, dict[int, float]],
    reference: ReferenceCurve,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Retient les efforts assez maximaux pour servir de mesure."""
    cfg = cfg or load_config()
    perf_cfg = cfg.performance
    rows = []

    for _, s in sessions.iterrows():
        if not s.get("is_run", False):
            continue
        mmp = mmp_by_session.get(s["session_id"])
        if not mmp:
            continue

        best_index, best_d, best_v = float("-inf"), None, float("nan")
        for d, v in mmp.items():
            if d < perf_cfg.get("min_duration_s", 300) or not np.isfinite(v):
                continue
            ref_v = reference.speed_at(d)
            if np.isfinite(ref_v) and ref_v > 0 and v / ref_v > best_index:
                best_index, best_d, best_v = v / ref_v, d, v

        if best_d is None or best_index < perf_cfg.get("min_index", 0.90):
            continue

        flags = []
        hr_max_session = s.get("hr_max")
        athlete_hr_max = cfg.athlete.hr_max
        if athlete_hr_max and np.isfinite(_as_float(hr_max_session)) and (
            _as_float(hr_max_session)
            < perf_cfg.get("min_hr_frac_of_max", 0.9) * athlete_hr_max
        ):
            flags.append("hr_not_maximal")
        gain, dist = _as_float(s.get("elevation_gain_m")), _as_float(s.get("distance_km_reported"))
        if np.isfinite(gain) and np.isfinite(dist) and dist > 0 and (
            gain / dist > perf_cfg.get("max_elevation_gain_m_per_km", 15)
        ):
            flags.append("hilly")

        rows.append({
            "session_id": s["session_id"],
            "date": s["date"],
            "perf_index": float(best_index),
            "duration_s": int(best_d),
            "speed_ms": float(best_v),
            "source": s["source"],
            "quality_flags": "|".join(flags),
        })

    return pd.DataFrame(rows, columns=[
        "session_id", "date", "perf_index", "duration_s", "speed_ms",
        "source", "quality_flags",
    ])


# ── pipeline complet ──────────────────────────────────────────────────────

def run_pipeline(
    spec: SourceSpec,
    cfg: Config | None = None,
    *,
    limit: int | None = None,
    resume: bool = True,
    allow_synthetic: bool = False,
    progress=None,
) -> IngestReport:
    """
    Execute la chaine complete et ecrit les trois artefacts.

    Idempotent : les seances deja presentes sont reconnues a leur identifiant
    et ignorees. Sur un athlete a neuf cents activites, la reprise evite de
    tout recalculer apres une interruption.
    """
    cfg = cfg or load_config()
    cfg.paths.ensure()
    report = IngestReport()

    # Amorcage : sans FC max ni FC repos, aucun TRIMP n'est calculable.
    if not cfg.athlete.is_calibrated:
        bounds = estimate_hr_bounds(spec, cfg)
        if bounds is None:
            raise ConfigError(
                "FC max et FC repos inconnues, et le jeu de donnees ne permet "
                "pas de les estimer. Les renseigner dans config/params.yaml."
            )
        hr_max, hr_rest = bounds
        log.info("Bornes cardiaques estimees : max %.0f, repos %.0f", hr_max, hr_rest)
        cfg = replace(cfg, athlete=replace(cfg.athlete, hr_max=hr_max, hr_rest=hr_rest))

    existing = _read_parquet(cfg.paths.processed / SESSIONS_FILE)
    known = set(existing["session_id"]) if resume and not existing.empty else set()

    rows: list[dict] = []
    mmp_by_session: dict[str, dict[int, float]] = {}
    durations = tuple(cfg.performance.get("mmp_durations_s", (300, 600, 1200, 1800, 2700, 3600)))
    max_gap = cfg.signal.get("max_gap_s", 30.0)

    for raw in ingest(spec, cfg):
        if raw.session_id in known:
            report.n_skipped += 1
            continue
        if not raw.source.is_real and not allow_synthetic:
            report.n_failed += 1
            report.failures.append((raw.source_ref, "source synthetique non autorisee"))
            continue
        try:
            cleaned = clean(raw, cfg)
            rows.append(featurize(cleaned, cfg))
            if cleaned.is_run:
                mmp_by_session[raw.session_id] = mean_max_speed(
                    cleaned.records, durations, max_gap
                )
            report.n_new += 1
            if progress:
                progress(report.n_new, raw.source_ref)
        except Exception as exc:
            report.n_failed += 1
            report.failures.append((raw.source_ref, f"{type(exc).__name__}: {exc}"))
        if limit and report.n_new >= limit:
            break

    if not rows:
        return report

    sessions = pd.DataFrame(rows)
    if resume and not existing.empty:
        sessions = pd.concat([existing, sessions], ignore_index=True)
    # keep="last" : une seance recalculee remplace son ancienne version. Avec
    # le defaut ("first"), --rebuild ne reconstruisait rien puisque les lignes
    # existantes gagnaient la deduplication.
    sessions = sessions.drop_duplicates("session_id", keep="last").sort_values("start_time")
    _write_parquet(sessions, cfg.paths.processed / SESSIONS_FILE)

    _write_parquet(build_daily_load(sessions), cfg.paths.processed / DAILY_LOAD_FILE)

    if mmp_by_session:
        try:
            reference = fit_reference_curve(list(mmp_by_session.values()), durations)
            obs = build_perf_observations(sessions, mmp_by_session, reference, cfg)
            _write_parquet(obs, cfg.paths.processed / PERF_OBS_FILE)
            _write_calibration(cfg, sessions, reference, obs)
        except Exception as exc:
            log.warning("Courbe de reference non ajustee : %s", exc)

    return report


def _write_calibration(cfg: Config, sessions: pd.DataFrame,
                       reference: ReferenceCurve, obs: pd.DataFrame) -> None:
    """
    Fige ce qui ne doit plus etre devine : FC max et repos estimees sur
    l'historique reel, et courbe de reference qui definit ce que vaut 1.0.
    """
    # La FC de repos s'estime sur les minima de seance, pas sur les moyennes.
    # Le premier percentile des FC MOYENNES vaut 115 bpm chez cet athlete,
    # contre 62 pour les minima : ecrire 115 dans params.yaml ramenerait la
    # reserve cardiaque de 132 a 79 bpm et annulerait tout travail sous
    # 115 bpm, faussant l'integralite du TRIMP et donc de Layer B.
    hr_max = pd.to_numeric(sessions.get("hr_max"), errors="coerce").dropna()
    hr_min = pd.to_numeric(sessions.get("hr_min_reported"), errors="coerce").dropna()
    hr_min = hr_min[(hr_min > 25) & (hr_min < 120)]  # ecarte les artefacts de capteur
    payload = {
        "athlete_id": cfg.athlete.id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_sessions": len(sessions),
        "n_perf_observations": len(obs),
        "hr_max_estimated": float(np.percentile(hr_max, 99)) if len(hr_max) else None,
        "hr_rest_estimated": float(np.percentile(hr_min, 1)) if len(hr_min) else None,
        "reference_curve": reference.to_dict(),
        "provenance": provenance_of(sessions).to_dict(),
    }
    path = cfg.paths.processed / CALIBRATION_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def provenance_of(sessions: pd.DataFrame) -> ProvenanceSummary:
    if sessions.empty or "source" not in sessions.columns:
        return summarize([])
    return summarize(sessions["source"].tolist())


def load_sessions(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return _read_parquet(cfg.paths.processed / SESSIONS_FILE)


def load_daily_load(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return _read_parquet(cfg.paths.processed / DAILY_LOAD_FILE)


def load_perf_observations(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return _read_parquet(cfg.paths.processed / PERF_OBS_FILE)


def load_calibration(cfg: Config | None = None) -> dict | None:
    cfg = cfg or load_config()
    path = cfg.paths.processed / CALIBRATION_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
