"""
Validation hors echantillon pour series temporelles.

Un modele d'entrainement se juge sur sa capacite a predire ce qui n'est pas
encore arrive. Un melange aleatoire laisserait le modele apprendre le futur
et produirait des scores flatteurs mais faux : tous les decoupages ici
respectent donc l'ordre du temps.

L'utilite concrete sur ce projet : les scores d'ajustement suggeraient un
modele « faible mais positif » (R2 = 0,03 a 0,14 selon les variantes), alors
que hors echantillon aucune variante ne bat la moyenne. C'est l'ecart entre
les deux colonnes qui constitue le resultat.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import mae, r2, rmse, skill_score

# Un modele : (dates_train, y_train, dates_test) -> predictions
Predictor = Callable[[pd.Series, np.ndarray, pd.Series], np.ndarray]


def blocked_splits(n: int, n_splits: int = 5, min_train: int | None = None
                   ) -> Iterator[tuple[slice, slice]]:
    """
    Decoupage temporel a fenetre extensible.

    L'entrainement part du debut et s'etend ; le test est le bloc suivant,
    jamais anterieur. Aucune information future ne franchit la frontiere.
    """
    if n_splits < 2 or n < n_splits * 2:
        return
    fold = n // n_splits
    min_train = min_train or fold
    for i in range(1, n_splits):
        train_end = fold * i
        if train_end < min_train:
            continue
        test_end = fold * (i + 1) if i < n_splits - 1 else n
        yield slice(0, train_end), slice(train_end, test_end)


def expanding_window_splits(n: int, min_train: int = 10, step: int = 1
                            ) -> Iterator[tuple[slice, slice]]:
    """
    Leave-future-out : ajuster sur les k premieres, predire la suivante.

    Plus exigeant que le decoupage en blocs, et plus proche de l'usage reel
    d'un systeme qui conseille une seance a la fois.
    """
    for k in range(min_train, n, step):
        yield slice(0, k), slice(k, min(k + step, n))


def horizon_splits(
    dates: pd.Series,
    horizon_days: int = 28,
    min_train_days: int = 180,
    step_days: int = 28,
) -> Iterator[tuple[slice, slice]]:
    """
    Decoupage a horizon CALENDAIRE fixe, par opposition aux deux precedents.

    `blocked_splits` et `expanding_window_splits` comptent des observations.
    Sur une cible irreguliere — l'enveloppe telle qu'elle etait rapportee a
    des ecarts de 1 a 140 jours entre points voisins — un bloc de test couvre
    donc une duree tres variable, et le score agrege melange des previsions
    a une semaine et a cinq mois. Comme mesure d'un horizon de prevision cela
    ne veut rien dire.

    Ici l'entrainement s'arrete a une date, et le test est l'ensemble des
    observations tombant dans les `horizon_days` jours qui suivent. L'origine
    avance de `step_days` a chaque pli. C'est la question que poserait un
    athlete : « ou sera mon niveau dans quatre semaines ? »

    Les dates doivent etre triees — `cross_validate` s'en charge.
    """
    if horizon_days < 1 or step_days < 1:
        raise ValueError("L'horizon et le pas doivent valoir au moins un jour.")
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)))
    if d.empty:
        return
    origin = d.min() + pd.Timedelta(days=min_train_days)
    last = d.max()
    while origin < last:
        train_end = int(d.searchsorted(origin, side="right"))
        test_end = int(d.searchsorted(origin + pd.Timedelta(days=horizon_days), side="right"))
        # Un pli sans observation a predire, ou sans historique, n'apprend
        # rien a personne : on avance plutot que de le compter comme un echec.
        if train_end > 0 and test_end > train_end:
            yield slice(0, train_end), slice(train_end, test_end)
        origin += pd.Timedelta(days=step_days)


@dataclass
class FoldResult:
    n_train: int
    n_test: int
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)


@dataclass
class ValidationResult:
    """Resultat complet d'une variante, ajustement et hors echantillon."""

    name: str
    r2_in_sample: float
    r2_out_of_sample: float
    rmse_out_of_sample: float
    mae_out_of_sample: float
    n_observations: int
    n_folds: int
    skill_vs_mean: float = float("nan")
    # Predictions concatenees et identifiant du pli dont chaque point provient.
    # Conservees pour pouvoir comparer deux variantes point a point : elles
    # predisent les memes points sous le meme decoupage, donc leurs scores
    # agreges ne suffisent pas a dire si un ecart depasse le bruit.
    y_true: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    blocks: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    @property
    def is_better_than_guessing(self) -> bool:
        return np.isfinite(self.r2_out_of_sample) and self.r2_out_of_sample > 0

    @property
    def overfitting_gap(self) -> float:
        """Ecart ajustement / hors echantillon. Grand = surapprentissage."""
        return self.r2_in_sample - self.r2_out_of_sample

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "r2_in": self.r2_in_sample,
            "r2_out": self.r2_out_of_sample,
            "gap": self.overfitting_gap,
            "rmse_out": self.rmse_out_of_sample,
            "mae_out": self.mae_out_of_sample,
            "skill_vs_mean": self.skill_vs_mean,
            "n": self.n_observations,
            "folds": self.n_folds,
        }


def cross_validate(
    name: str,
    predictor: Predictor,
    dates: pd.Series,
    y: np.ndarray,
    n_splits: int = 5,
    expanding: bool = False,
    splits: list[tuple[slice, slice]] | None = None,
) -> ValidationResult:
    """
    Evalue une variante et renvoie ses scores d'ajustement et hors echantillon.

    Le R2 hors echantillon est calcule contre la moyenne du jeu
    d'ENTRAINEMENT : utiliser celle du test donnerait au modele une
    information dont il ne disposait pas.

    `splits` permet d'imposer un decoupage deja construit — `horizon_splits`
    notamment, qui a besoin des dates et non du seul nombre d'observations.
    Toutes les variantes d'une meme comparaison doivent recevoir le MEME
    decoupage, sans quoi les colonnes ne sont plus comparables.
    """
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    n = len(y)

    if splits is None:
        splits = list(expanding_window_splits(n) if expanding else blocked_splits(n, n_splits))
    splits = list(splits)
    folds: list[FoldResult] = []
    trues, preds, train_means = [], [], []

    for tr, te in splits:
        try:
            p = np.asarray(predictor(dates[tr], y[tr], dates[te]), dtype=float)
        except Exception:
            continue
        if p.shape != y[te].shape:
            continue
        folds.append(FoldResult(tr.stop - tr.start, te.stop - te.start, y[te], p))
        trues.append(y[te])
        preds.append(p)
        train_means.append(np.nanmean(y[tr]))

    if not folds:
        return ValidationResult(name, float("nan"), float("nan"), float("nan"),
                                float("nan"), n, 0)

    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    # Reference par pli : la moyenne connue au moment de la prediction.
    mean_pred = np.concatenate([np.full(f.n_test, m) for f, m in zip(folds, train_means, strict=False)])

    try:
        in_sample = np.asarray(predictor(dates, y, dates), dtype=float)
        r2_in = r2(y, in_sample)
    except Exception:
        r2_in = float("nan")

    return ValidationResult(
        name=name,
        r2_in_sample=r2_in,
        r2_out_of_sample=r2(y_true, y_pred, baseline_mean=float(np.nanmean(mean_pred))),
        rmse_out_of_sample=rmse(y_true, y_pred),
        mae_out_of_sample=mae(y_true, y_pred),
        n_observations=n,
        n_folds=len(folds),
        skill_vs_mean=skill_score(y_true, y_pred, mean_pred),
        y_true=y_true,
        y_pred=y_pred,
        blocks=np.concatenate([np.full(f.n_test, i) for i, f in enumerate(folds)]),
    )


def compare_results(
    variants: dict[str, Predictor],
    dates: pd.Series,
    y: np.ndarray,
    n_splits: int = 5,
    expanding: bool = False,
    splits: list[tuple[slice, slice]] | None = None,
) -> dict[str, ValidationResult]:
    """
    Comme `compare`, mais rend les resultats complets plutot qu'un tableau.

    Les predictions par pli y sont conservees : sans elles, impossible de
    comparer deux variantes autrement qu'en soustrayant deux R2, ce qui ne
    dit rien de la solidite de l'ecart.
    """
    if splits is not None:
        splits = list(splits)
    return {name: cross_validate(name, p, dates, y, n_splits, expanding, splits)
            for name, p in variants.items()}


def compare(
    variants: dict[str, Predictor],
    dates: pd.Series,
    y: np.ndarray,
    n_splits: int = 5,
    expanding: bool = False,
    splits: list[tuple[slice, slice]] | None = None,
) -> pd.DataFrame:
    """
    Evalue plusieurs variantes sous le meme protocole et classe les resultats.

    Le tri se fait sur le score hors echantillon, jamais sur l'ajustement.
    """
    results = compare_results(variants, dates, y, n_splits, expanding, splits)
    rows = [r.to_dict() for r in results.values()]
    return pd.DataFrame(rows).sort_values("r2_out", ascending=False).reset_index(drop=True)
