"""
References de comparaison, declarees avant toute variante de modele.

Ce module existe pour une raison precise : sans reference, un R2 de 0,03 se
raconte comme « faible mais positif » alors qu'il peut etre moins bon que
predire la moyenne. Les references sont donc fixees ici, une fois, et
n'importe quel modele doit les battre pour meriter d'etre presente.

Toutes suivent la meme interface :

    fit(dates_train, y_train) -> None
    predict(dates) -> np.ndarray

Aucune ne regarde l'avenir : elles ne recoivent que le passe du pli courant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class Baseline:
    """Interface commune. Une reference qui n'a rien vu predit NaN."""

    name: str = "baseline"

    def fit(self, dates: pd.Series, y: np.ndarray) -> Baseline:
        raise NotImplementedError

    def predict(self, dates: pd.Series) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, dates_tr, y_tr, dates_te) -> np.ndarray:
        return self.fit(dates_tr, y_tr).predict(dates_te)


@dataclass
class MeanBaseline(Baseline):
    """Predire la moyenne de l'historique. La barre minimale absolue."""

    name: str = "moyenne"
    _value: float = field(default=float("nan"), repr=False)

    def fit(self, dates, y):
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        self._value = float(y.mean()) if y.size else float("nan")
        return self

    def predict(self, dates):
        return np.full(len(dates), self._value)


@dataclass
class LastValueBaseline(Baseline):
    """
    Predire la derniere valeur observee — la persistance.

    C'est la reference serieuse pour une serie temporelle. Un modele qui ne
    la bat pas n'apprend rien que l'inertie ne donne deja.
    """

    name: str = "persistance"
    _value: float = field(default=float("nan"), repr=False)

    def fit(self, dates, y):
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        self._value = float(y[-1]) if y.size else float("nan")
        return self

    def predict(self, dates):
        return np.full(len(dates), self._value)


@dataclass
class EwmaBaseline(Baseline):
    """Moyenne mobile exponentielle des performances passees."""

    halflife: float = 5.0
    name: str = "ewma"
    _value: float = field(default=float("nan"), repr=False)

    def fit(self, dates, y):
        s = pd.Series(np.asarray(y, dtype=float)).dropna()
        self._value = float(s.ewm(halflife=self.halflife).mean().iloc[-1]) if len(s) else float("nan")
        return self

    def predict(self, dates):
        return np.full(len(dates), self._value)


@dataclass
class TrendBaseline(Baseline):
    """
    Regression lineaire sur le temps seul.

    Reference exigeante : beaucoup de « progressions » s'expliquent par une
    simple derive temporelle, sans qu'aucune physiologie n'intervienne.
    """

    name: str = "tendance"
    _coef: tuple[float, float] = field(default=(float("nan"),) * 2, repr=False)
    _origin: pd.Timestamp | None = field(default=None, repr=False)

    def fit(self, dates, y):
        d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 3:
            return self
        self._origin = d[ok].iloc[0]
        t = (d[ok] - self._origin).dt.days.to_numpy(dtype=float)
        slope, intercept = np.polyfit(t, y[ok], 1)
        self._coef = (float(slope), float(intercept))
        return self

    def predict(self, dates):
        if self._origin is None or not np.isfinite(self._coef[0]):
            return np.full(len(dates), float("nan"))
        t = (pd.to_datetime(pd.Series(dates)) - self._origin).dt.days.to_numpy(dtype=float)
        return self._coef[0] * t + self._coef[1]


def default_baselines() -> list[Baseline]:
    """
    Le jeu de references du projet.

    Fige ici pour qu'on ne puisse pas, apres coup, retirer celle qui gagne.
    """
    return [MeanBaseline(), LastValueBaseline(), EwmaBaseline(), TrendBaseline()]
