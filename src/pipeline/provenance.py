"""
Provenance des donnees.

Le tableau de bord appelait `bootstrap_trimp_history()` puis enregistrait
quatre-vingt-quatre seances inventees avec `source="fit_file"`. Tous les
graphiques affiches etaient synthetiques, et rien dans le systeme ne
permettait de s'en apercevoir : `add_session` avait un defaut, l'appelant
l'omettait, et l'etiquette mentait.

Le remede n'est pas une convention mais une contrainte. Le type de source est
obligatoire, l'ecriture de donnees synthetiques exige un consentement
explicite, et chaque artefact persiste porte sa propre origine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ProvenanceError(RuntimeError):
    """Tentative d'ecrire une donnee synthetique sans l'avoir autorisee."""


class DataSource(str, Enum):
    """D'ou vient une seance. Aucune valeur par defaut, nulle part."""

    OPENDATA_OSF = "opendata_osf"   # GoldenCheetah OpenData
    FIT_FILE = "fit_file"           # fichier .fit d'une montre
    MANUAL = "manual"               # saisie par l'utilisateur
    SYNTHETIC = "synthetic"         # genere — tests et demonstrations seulement

    @property
    def is_real(self) -> bool:
        return self is not DataSource.SYNTHETIC

    @property
    def label(self) -> str:
        return {
            DataSource.OPENDATA_OSF: "GoldenCheetah OpenData",
            DataSource.FIT_FILE: "fichier .fit",
            DataSource.MANUAL: "saisie manuelle",
            DataSource.SYNTHETIC: "SYNTHETIQUE",
        }[self]


def coerce_source(value: DataSource | str) -> DataSource:
    """Convertit une chaine en DataSource, en refusant l'inconnu."""
    if isinstance(value, DataSource):
        return value
    try:
        return DataSource(str(value))
    except ValueError:
        raise ProvenanceError(
            f"Source inconnue : {value!r}. Attendu : "
            f"{', '.join(s.value for s in DataSource)}."
        ) from None


@dataclass(frozen=True)
class ProvenanceSummary:
    """Ce qu'on affiche pour qu'un lecteur sache ce qu'il regarde."""

    n_total: int
    n_real: int
    n_synthetic: int
    sources: tuple[str, ...]

    @property
    def is_trustworthy(self) -> bool:
        return self.n_total > 0 and self.n_synthetic == 0

    @property
    def banner(self) -> str:
        if self.n_total == 0:
            return "Aucune donnee — lancer `python -m src.cli ingest`."
        if self.n_synthetic:
            return (
                f"DONNEES SYNTHETIQUES : {self.n_synthetic} seance(s) sur "
                f"{self.n_total} sont generees, pas mesurees."
            )
        labels = ", ".join(coerce_source(s).label for s in self.sources)
        return f"{self.n_total} seances reelles — {labels}"

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "sources": list(self.sources),
            "is_trustworthy": self.is_trustworthy,
        }


def summarize(sources: Iterable[DataSource | str]) -> ProvenanceSummary:
    values = [coerce_source(s) for s in sources]
    synthetic = sum(1 for s in values if not s.is_real)
    return ProvenanceSummary(
        n_total=len(values),
        n_real=len(values) - synthetic,
        n_synthetic=synthetic,
        sources=tuple(sorted({s.value for s in values})),
    )


def check_writable(source: DataSource, allow_synthetic: bool) -> DataSource:
    """
    Garde appelee avant toute ecriture de seance.

    Une donnee synthetique n'entre dans un magasin que si l'appelant l'a
    demande explicitement a la construction. En pratique : seuls les tests.
    """
    if not source.is_real and not allow_synthetic:
        raise ProvenanceError(
            "Refus d'enregistrer une seance synthetique dans un magasin reel. "
            "Construire l'objet avec allow_synthetic=True si c'est voulu "
            "(tests et demonstrations uniquement)."
        )
    return source
