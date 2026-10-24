"""
Client GoldenCheetah OpenData (projet OSF 6hfpz, DOI 10.17605/OSF.IO/6HFPZ).

Un zip par athlete, contenant :
  - <uuid>.json  resume : ATHLETE{gender, yob}, RIDES[] avec date, sport,
                 `data` (code de presence des canaux) et METRICS
  - un CSV par activite, colonnes `secs,km,power,hr,cad,alt`

Ces fichiers ne contiennent ni coordonnees GPS, ni intervalles RR, ni donnee
identifiante.

Le client est cache-first : tout zip telecharge est conserve sous
`paths.cache/opendata/`, verifie par md5, et le reseau n'est sollicite qu'en
cas d'absence. Tests et tableau de bord fonctionnent donc hors ligne.

Note : le bucket S3 historique est ferme au public ; l'API OSF est la seule
voie d'acces.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

OSF_API = "https://api.osf.io/v2"
OSF_DOWNLOAD = "https://osf.io/download/{file_id}/"
DEFAULT_NODE = "6hfpz"
DATASET_DOI = "10.17605/OSF.IO/6HFPZ"

CSV_COLUMNS = ("secs", "km", "power", "hr", "cad", "alt")

# Le champ `data` du JSON code les canaux presents, une lettre par canal.
# Le lire evite d'ouvrir les CSV pour qualifier un athlete.
CHANNEL_HEARTRATE = "H"
CHANNEL_ALTITUDE = "A"
CHANNEL_CADENCE = "C"


class OpenDataError(RuntimeError):
    """Echec d'acces ou archive invalide."""


@dataclass(frozen=True)
class RemoteAthlete:
    """Entree du listing OSF, avant tout telechargement."""

    file_id: str
    name: str
    size_bytes: int
    md5: str

    @property
    def uuid(self) -> str:
        return self.name.removesuffix(".zip")

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6


@dataclass(frozen=True)
class Ride:
    """Une activite, telle que decrite par le resume (sans ouvrir son CSV)."""

    filename: str
    date: datetime | None
    sport: str
    channels: str
    metrics: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def has_heartrate(self) -> bool:
        return CHANNEL_HEARTRATE in self.channels

    @property
    def is_run(self) -> bool:
        return self.sport.lower().startswith("run")

    def metric(self, key: str) -> float | None:
        """
        Valeur numerique d'une metrique GoldenCheetah.

        Certaines metriques sont stockees en paire [valeur, duree] (average_hr
        par exemple) et d'autres en chaine ; on ne garde que la valeur.
        """
        value = self.metrics.get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if f == f else None  # ecarte les NaN


@dataclass(frozen=True)
class AthleteSummary:
    """Le fichier <uuid>.json, decode."""

    uuid: str
    gender: str | None
    year_of_birth: int | None
    rides: list[Ride]

    @property
    def runs(self) -> list[Ride]:
        return [r for r in self.rides if r.is_run]

    @property
    def runs_with_hr(self) -> list[Ride]:
        return [r for r in self.runs if r.has_heartrate]

    @property
    def sport_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rides:
            counts[r.sport] = counts.get(r.sport, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def span_days(self) -> int:
        dates = [r.date for r in self.rides if r.date is not None]
        return (max(dates) - min(dates)).days if len(dates) > 1 else 0

    @property
    def hr_coverage(self) -> float:
        runs = self.runs
        return len(self.runs_with_hr) / len(runs) if runs else 0.0


def _parse_ride_date(raw: Any) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    for fmt in ("%Y/%m/%d %H:%M:%S UTC", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    log.debug("Date d'activite non reconnue : %r", raw)
    return None


def _csv_name_from_date(when: datetime | None) -> str:
    """
    Nom probable du CSV, deduit de la date de l'activite.

    Indicatif seulement : le resume horodate en UTC tandis que les fichiers
    sont nommes en heure locale. `resolve_ride_files` fait l'appariement reel.
    """
    return f"{when:%Y_%m_%d_%H_%M_%S}.csv" if when else ""


def _timestamp_from_csv_name(name: str) -> datetime | None:
    try:
        return datetime.strptime(Path(name).stem, "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return None


def parse_summary(raw: dict[str, Any], uuid: str) -> AthleteSummary:
    """Decode le JSON de resume. Tolere les champs absents ou mal typés."""
    athlete = raw.get("ATHLETE") or {}
    yob = athlete.get("yob")
    try:
        yob = int(yob) if yob else None
    except (TypeError, ValueError):
        yob = None

    rides = []
    for entry in raw.get("RIDES") or []:
        if not isinstance(entry, dict):
            continue
        when = _parse_ride_date(entry.get("date"))
        # Les metriques vivent dans un sous-bloc METRICS ; quelques archives
        # les aplatissent au premier niveau. On accepte les deux.
        metrics = entry.get("METRICS")
        if not isinstance(metrics, dict):
            metrics = {k: v for k, v in entry.items()
                       if k not in {"filename", "date", "sport", "data", "METRICS"}}
        rides.append(
            Ride(
                filename=str(entry.get("filename") or _csv_name_from_date(when)),
                date=when,
                sport=str(entry.get("sport", "") or "Unknown"),
                channels=str(entry.get("data", "") or ""),
                metrics=metrics,
            )
        )
    return AthleteSummary(
        uuid=uuid,
        gender=athlete.get("gender"),
        year_of_birth=yob,
        rides=rides,
    )


class OpenDataClient:
    """Acces cache-first au jeu de donnees."""

    def __init__(
        self,
        cache_dir: Path,
        node: str = DEFAULT_NODE,
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) / "opendata"
        self.node = node
        self.timeout = timeout
        self._session = session or requests.Session()

    # -- listing -----------------------------------------------------------

    def list_athletes(self, limit: int | None = None,
                      page_size: int = 100) -> Iterator[RemoteAthlete]:
        """Parcourt le listing OSF, page par page, sans rien telecharger."""
        url = f"{OSF_API}/nodes/{self.node}/files/osfstorage/"
        params: dict[str, Any] | None = {"page[size]": page_size}
        seen = 0

        while url:
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise OpenDataError(
                    f"Listing OSF indisponible ({exc}). Le bucket S3 historique "
                    f"etant ferme, l'API OSF est la seule voie d'acces."
                ) from exc

            payload = resp.json()
            for item in payload.get("data", []):
                attrs = item.get("attributes", {})
                if attrs.get("kind") != "file":
                    continue
                yield RemoteAthlete(
                    file_id=item["id"],
                    name=attrs.get("name", ""),
                    size_bytes=int(attrs.get("size") or 0),
                    md5=(attrs.get("extra", {}).get("hashes", {}) or {}).get("md5", ""),
                )
                seen += 1
                if limit is not None and seen >= limit:
                    return

            url = payload.get("links", {}).get("next")
            params = None  # l'URL `next` porte deja sa pagination

    # -- telechargement ----------------------------------------------------

    def cached_path(self, uuid: str) -> Path:
        return self.cache_dir / f"{uuid}.zip"

    def fetch_athlete(self, athlete: RemoteAthlete, *, force: bool = False) -> Path:
        """Renvoie le chemin local du zip, en le telechargeant si besoin."""
        dest = self.cached_path(athlete.uuid)
        if dest.exists() and not force:
            if not athlete.md5 or _md5(dest) == athlete.md5:
                return dest
            log.warning("Cache corrompu pour %s, retelechargement.", athlete.uuid)

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".zip.part")
        try:
            with self._session.get(
                OSF_DOWNLOAD.format(file_id=athlete.file_id),
                stream=True, timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            raise OpenDataError(f"Telechargement de {athlete.uuid} echoue : {exc}") from exc

        if athlete.md5 and _md5(tmp) != athlete.md5:
            tmp.unlink(missing_ok=True)
            raise OpenDataError(
                f"md5 incorrect pour {athlete.uuid} : archive corrompue en transit."
            )
        tmp.replace(dest)
        return dest


# -- lecture d'archive (hors ligne, sans client) ---------------------------

def read_summary(zip_path: Path) -> AthleteSummary:
    """Lit le resume JSON d'un zip local."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".json")]
        if not names:
            raise OpenDataError(f"{zip_path.name} ne contient aucun resume JSON.")
        with zf.open(names[0]) as fh:
            raw = json.load(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
    uuid = (Path(names[0]).stem or zip_path.stem).strip("{}")
    return parse_summary(raw, uuid)


class ArchiveReader:
    """
    Archive maintenue ouverte pour la duree d'une ingestion.

    `read_activity` ouvrait le zip a chaque activite et `_resolve_member`
    rappelait `namelist()` a chaque lecture. Sur une archive de 38 Mo
    comportant 917 activites, cela coutait 14,9 ms par activite contre
    0,56 ms archive ouverte — 13,7 s contre 0,5 s au total.

    S'utilise en gestionnaire de contexte :

        with ArchiveReader(path) as archive:
            for name in archive.activities:
                rows = archive.read(name)
    """

    def __init__(self, zip_path: Path) -> None:
        self.path = Path(zip_path)
        self._zf: zipfile.ZipFile | None = None
        self._by_basename: dict[str, str] = {}

    def __enter__(self) -> ArchiveReader:
        self._zf = zipfile.ZipFile(self.path)
        # Index basename -> chemin complet, construit une fois.
        self._by_basename = {Path(n).name: n for n in self._zf.namelist()}
        return self

    def __exit__(self, *exc) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    @property
    def _archive(self) -> zipfile.ZipFile:
        if self._zf is None:
            raise OpenDataError("ArchiveReader utilise hors de son contexte.")
        return self._zf

    @property
    def activities(self) -> list[str]:
        return sorted(n for n in self._archive.namelist() if n.lower().endswith(".csv"))

    def summary(self) -> AthleteSummary:
        names = [n for n in self._archive.namelist() if n.lower().endswith(".json")]
        if not names:
            raise OpenDataError(f"{self.path.name} ne contient aucun resume JSON.")
        with self._archive.open(names[0]) as fh:
            raw = json.load(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
        return parse_summary(raw, (Path(names[0]).stem or self.path.stem).strip("{}"))

    def read(self, filename: str) -> list[dict[str, float | None]]:
        member = self._resolve(filename)
        with self._archive.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            return [{c: _to_float(row.get(c)) for c in CSV_COLUMNS}
                    for row in csv.DictReader(text)]

    def _resolve(self, filename: str) -> str:
        if filename in self._by_basename.values():
            return filename
        member = self._by_basename.get(Path(filename).name)
        if member is None:
            raise OpenDataError(f"Activite {filename!r} absente de l'archive.")
        return member


def read_activity(zip_path: Path, filename: str) -> list[dict[str, float | None]]:
    """
    Lit une activite isolee.

    Ouvre et referme l'archive : pratique pour un acces ponctuel, a eviter
    dans une boucle — utiliser `ArchiveReader` dans ce cas.
    """
    with ArchiveReader(zip_path) as archive:
        return archive.read(filename)


def list_activities(zip_path: Path) -> list[str]:
    with ArchiveReader(zip_path) as archive:
        return archive.activities


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# -- criblage --------------------------------------------------------------

@dataclass(frozen=True)
class ScreeningCriteria:
    """
    Ce qu'on exige d'un athlete pour qu'il serve de sujet de demonstration.

    Le modele de Banister a besoin d'un historique long et continu ; l'index
    de performance a besoin de frequence cardiaque ; et le denivele doit
    rester modere, sinon les meilleurs efforts refletent le terrain plus que
    la forme (un athlete de trail reste utile pour illustrer la correction
    GAP, mais fait un mauvais sujet principal).
    """

    min_runs: int = 150
    min_span_days: int = 365
    min_hr_coverage: float = 0.95
    max_elevation_m_per_km: float = 15.0
    max_size_mb: float = 60.0


@dataclass(frozen=True)
class ScreeningResult:
    remote: RemoteAthlete
    summary: AthleteSummary
    elevation_m_per_km: float | None
    passed: bool
    rejected_for: str | None

    def describe(self) -> str:
        s = self.summary
        elev = f"{self.elevation_m_per_km:.0f}" if self.elevation_m_per_km is not None else "?"
        verdict = "RETENU" if self.passed else f"ecarte ({self.rejected_for})"
        return (
            f"{s.uuid[:8]} {s.gender or '?'}{s.year_of_birth or '????'} "
            f"{len(s.runs):4d} courses ({s.hr_coverage:4.0%} FC) "
            f"{s.span_days:4d} j  {elev:>3s} m/km  -> {verdict}"
        )


def _median_elevation_per_km(summary: AthleteSummary) -> float | None:
    ratios = []
    for r in summary.runs:
        gain, dist = r.metric("elevation_gain"), r.metric("total_distance")
        if gain is not None and dist and dist > 1.0:
            ratios.append(gain / dist)
    if not ratios:
        return None
    ratios.sort()
    return ratios[len(ratios) // 2]


def evaluate_athlete(remote: RemoteAthlete, summary: AthleteSummary,
                     criteria: ScreeningCriteria) -> ScreeningResult:
    elevation = _median_elevation_per_km(summary)
    reason = None
    if len(summary.runs) < criteria.min_runs:
        reason = f"{len(summary.runs)} courses < {criteria.min_runs}"
    elif summary.span_days < criteria.min_span_days:
        reason = f"historique {summary.span_days} j < {criteria.min_span_days}"
    elif summary.hr_coverage < criteria.min_hr_coverage:
        reason = f"couverture FC {summary.hr_coverage:.0%} < {criteria.min_hr_coverage:.0%}"
    elif elevation is not None and elevation > criteria.max_elevation_m_per_km:
        reason = f"denivele {elevation:.0f} m/km > {criteria.max_elevation_m_per_km:.0f}"
    return ScreeningResult(remote, summary, elevation, reason is None, reason)


def resolve_ride_files(
    summary: AthleteSummary,
    activity_names: list[str],
    max_offset_hours: float = 14.0,
) -> dict[int, str]:
    """
    Apparie chaque activite du resume au CSV correspondant de l'archive.

    Le resume horodate en UTC alors que les fichiers sont nommes en heure
    locale, avec un decalage qui depend du fuseau de l'athlete et de l'heure
    d'ete. Plutot que de deviner ce decalage, on l'estime : on prend l'ecart
    median entre les deux series, puis on apparie au plus proche autour de
    cette valeur. Chaque CSV n'est attribue qu'une fois.
    """
    stamped = [(n, ts) for n in activity_names
               if (ts := _timestamp_from_csv_name(n)) is not None]
    rides = [(i, r.date) for i, r in enumerate(summary.rides) if r.date is not None]
    if not stamped or not rides:
        return {}

    # Decalage dominant, estime sur les appariements naifs les plus proches.
    offsets = []
    for _, ride_date in rides:
        best = min(stamped, key=lambda s: abs((s[1] - ride_date).total_seconds()))
        offsets.append((best[1] - ride_date).total_seconds())
    offsets.sort()
    shift = offsets[len(offsets) // 2]

    tolerance = max_offset_hours * 3600
    remaining = dict(stamped)
    matched: dict[int, str] = {}
    for idx, ride_date in rides:
        target = ride_date.timestamp() + shift
        candidate = min(
            remaining.items(),
            key=lambda kv: abs(kv[1].timestamp() - target),
            default=None,
        )
        if candidate and abs(candidate[1].timestamp() - target) <= tolerance:
            matched[idx] = candidate[0]
            del remaining[candidate[0]]
    return matched
