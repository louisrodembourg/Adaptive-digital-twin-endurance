"""
Configuration centralisee du projet.

Un seul point de chargement, mis en cache, remplacant les sept copies de
`load_params()` qui reparsaient params.yaml a chaque appel de metrique.

Usage
-----
    from src.config import load_config
    cfg = load_config()
    cfg.signal.max_gap_s
    cfg.athlete.hr_max

Les tests injectent une configuration isolee via `Config.with_paths(tmp)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "params.yaml"

# Coefficients de Banister pour le TRIMP exponentiel :
#   TRIMP = duree_min * dHR * b * exp(c * dHR)
# Derives du sexe plutot que stockes, pour qu'ils ne puissent pas etre
# desynchronises l'un de l'autre dans le YAML.
TRIMP_COEFFS: dict[str, tuple[float, float]] = {
    "male": (0.64, 1.92),
    "female": (0.86, 1.67),
}


class ConfigError(ValueError):
    """Configuration absente, incoherente ou non calibree."""


def _require(value: Any, field: str) -> Any:
    if value is None:
        raise ConfigError(
            f"'{field}' n'est pas renseigne. Lancer `python -m src.cli calibrate` "
            f"ou completer config/params.yaml."
        )
    return value


@dataclass(frozen=True)
class Athlete:
    id: str | None
    display_name: str | None
    sex: str
    hr_max: float | None
    hr_rest: float | None
    weight_kg: float | None

    @property
    def trimp_coeffs(self) -> tuple[float, float]:
        """(b, c) du TRIMP exponentiel, derives du sexe."""
        try:
            return TRIMP_COEFFS[self.sex]
        except KeyError:
            raise ConfigError(
                f"athlete.sex = {self.sex!r} ; attendu : {sorted(TRIMP_COEFFS)}"
            ) from None

    @property
    def hr_reserve(self) -> float:
        hr_max = _require(self.hr_max, "athlete.hr_max")
        hr_rest = _require(self.hr_rest, "athlete.hr_rest")
        if hr_max <= hr_rest:
            raise ConfigError(f"hr_max ({hr_max}) doit depasser hr_rest ({hr_rest}).")
        return float(hr_max - hr_rest)

    @property
    def is_calibrated(self) -> bool:
        return self.hr_max is not None and self.hr_rest is not None


@dataclass(frozen=True)
class Goal:
    event: str
    distance_m: float
    reference_time_s: float | None
    target_time_s: float | None
    improvement_pct: float
    race_date: date | None

    @property
    def resolved_target_time_s(self) -> float:
        """Chrono vise, explicite ou derive de l'amelioration demandee."""
        if self.target_time_s is not None:
            return float(self.target_time_s)
        ref = _require(self.reference_time_s, "goal.reference_time_s")
        return float(ref) * (1.0 - self.improvement_pct / 100.0)

    @property
    def target_speed_ms(self) -> float:
        return self.distance_m / self.resolved_target_time_s

    @property
    def target_pace_min_km(self) -> float:
        return (self.resolved_target_time_s / 60.0) / (self.distance_m / 1000.0)

    @property
    def days_to_race(self) -> int | None:
        if self.race_date is None:
            return None
        return (self.race_date - date.today()).days


@dataclass(frozen=True)
class DataSourceCfg:
    kind: str
    osf_node: str
    athlete_ref: str | None
    manifest: str
    fit_dir: str


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    cache: Path
    traces: Path

    def ensure(self) -> None:
        for p in (self.raw, self.processed, self.cache, self.traces):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    athlete: Athlete
    goal: Goal
    data_source: DataSourceCfg
    paths: Paths
    hr_zones: dict[str, list[float]]
    signal: dict[str, Any]
    performance: dict[str, Any]
    twin: dict[str, Any]
    planner: dict[str, Any]
    source_path: Path

    def with_paths(self, root: Path | str) -> Config:
        """Configuration identique, reracinee ailleurs (utilise par les tests)."""
        root = Path(root)
        return replace(
            self,
            paths=Paths(
                root=root,
                raw=root / "raw",
                processed=root / "processed",
                cache=root / "cache",
                traces=root / "processed" / "traces",
            ),
        )


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _build(raw: dict[str, Any], path: Path) -> Config:
    root = path.resolve().parents[1]
    p = raw.get("paths", {})

    def _p(key: str, default: str) -> Path:
        return root / p.get(key, default)

    a, g, ds = raw["athlete"], raw["goal"], raw["data_source"]
    return Config(
        athlete=Athlete(
            id=a.get("id"),
            display_name=a.get("display_name"),
            sex=a.get("sex", "male"),
            hr_max=a.get("hr_max"),
            hr_rest=a.get("hr_rest"),
            weight_kg=a.get("weight_kg"),
        ),
        goal=Goal(
            event=g.get("event", "half_marathon"),
            distance_m=float(g["distance_m"]),
            reference_time_s=g.get("reference_time_s"),
            target_time_s=g.get("target_time_s"),
            improvement_pct=float(g.get("improvement_pct", 0.0)),
            race_date=_parse_date(g.get("race_date")),
        ),
        data_source=DataSourceCfg(
            kind=ds.get("kind", "opendata_osf"),
            osf_node=ds.get("osf_node", "6hfpz"),
            athlete_ref=ds.get("athlete_ref"),
            manifest=ds.get("manifest", "config/opendata_manifest.json"),
            fit_dir=ds.get("fit_dir", "data/raw"),
        ),
        paths=Paths(
            root=root / "data",
            raw=_p("raw", "data/raw"),
            processed=_p("processed", "data/processed"),
            cache=_p("cache", "data/cache"),
            traces=_p("traces", "data/processed/traces"),
        ),
        hr_zones=raw.get("hr_zones", {}),
        signal=raw.get("signal", {}),
        performance=raw.get("performance", {}),
        twin=raw.get("twin", {}),
        planner=raw.get("planner", {}),
        source_path=path,
    )


@lru_cache(maxsize=None)
def _load_cached(path_str: str) -> Config:
    path = Path(path_str)
    if not path.exists():
        raise ConfigError(f"Fichier de configuration introuvable : {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _build(raw, path)


def load_config(path: Path | str | None = None) -> Config:
    """Charge la configuration (mise en cache par chemin)."""
    return _load_cached(str(Path(path) if path else DEFAULT_CONFIG_PATH))


def clear_config_cache() -> None:
    """Vide le cache — a appeler apres avoir reecrit params.yaml."""
    _load_cached.cache_clear()
    _load_raw_cached.cache_clear()


@lru_cache(maxsize=None)
def _load_raw_cached(path_str: str) -> dict[str, Any]:
    with Path(path_str).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_params(path: Path | str | None = None) -> dict[str, Any]:
    """
    Dictionnaire brut de params.yaml, mis en cache.

    Conserve pour les blocs purement numeriques (`signal`, `twin`, `planner`,
    `hr_zones`) dont l'acces par cle reste le plus lisible. Tout ce qui touche
    a l'athlete ou a l'objectif doit passer par `load_config()`, qui applique
    les gardes de calibration et les derivations.
    """
    return _load_raw_cached(str(Path(path) if path else DEFAULT_CONFIG_PATH))
