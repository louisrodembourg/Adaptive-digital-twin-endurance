"""
Enveloppe de performance : ce dont l'athlete est capable, et non ce qu'il a
choisi de faire.

Le projet mesurait la performance par le meilleur effort de chaque seance.
C'est un mauvais estimande, et la mesure le montre sans ambiguite :

    autocorrelation au retard 1 de l'effort ponctuel   0,15
    autocorrelation au retard 1 de l'enveloppe 42 j    0,97

Un meilleur effort de seance decrit surtout ce qu'etait la seance —
fractionne ou recuperation — pas la capacite de l'athlete. Deux sorties
faciles consecutives donnent deux « performances » basses sans qu'aucune
forme n'ait bouge. La serie obtenue est du bruit autour d'un niveau, et
aucun modele ne peut en tirer quoi que ce soit : c'est ce qui explique qu'ici
aucune variante de Banister ne batte le fait de predire la moyenne.

L'enveloppe repond a une autre question : « quel est le meilleur dont cet
athlete ait ete capable recemment ? » Elle est monotone par construction sur
sa fenetre, donc lisse, donc predictible — et c'est la grandeur que le modele
de charge est cense expliquer.

Contrepartie a ne pas cacher : une enveloppe glissante est fortement
autocorrelee, donc facile a predire par simple persistance. Tout modele qui
s'y compare doit battre cette persistance, pas seulement obtenir un R2
flatteur. `src/evaluation/baselines.py` la fournit pour cette raison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_WINDOW_DAYS = 42


@dataclass(frozen=True)
class EnvelopeSpec:
    """Definition d'une enveloppe : quelle colonne, quelle fenetre."""

    column: str
    window_days: int = DEFAULT_WINDOW_DAYS
    min_periods: int = 3
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or f"{self.column}_env{self.window_days}"


def rolling_envelope(
    observations: pd.DataFrame,
    spec: EnvelopeSpec,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Meilleure valeur atteinte sur une fenetre glissante se terminant ce jour.

    La fenetre est exprimee en JOURS, pas en nombre de seances : une periode
    creuse doit faire retomber l'enveloppe, ce qu'un decompte de lignes ne
    ferait pas.

    Ne regarde jamais l'avenir — `rolling` sur un index temporel est
    strictement causal, ce qui rend la sortie utilisable telle quelle comme
    variable a expliquer dans une validation temporelle.
    """
    if observations.empty or spec.column not in observations.columns:
        return pd.DataFrame(columns=[date_column, spec.label, "n_in_window"])

    df = observations[[date_column, spec.column]].copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.normalize()
    df = df.dropna(subset=[spec.column]).sort_values(date_column)
    if df.empty:
        return pd.DataFrame(columns=[date_column, spec.label, "n_in_window"])

    # Une seule valeur par jour : le meilleur effort du jour.
    daily = df.groupby(date_column)[spec.column].max()
    window = f"{spec.window_days}D"
    env = daily.rolling(window, min_periods=1).max()
    count = daily.rolling(window, min_periods=1).count()

    out = pd.DataFrame({
        date_column: env.index,
        spec.label: env.to_numpy(),
        "n_in_window": count.to_numpy().astype(int),
    })
    # Une enveloppe adossee a une seule seance n'est qu'une seance.
    return out[out["n_in_window"] >= spec.min_periods].reset_index(drop=True)


def envelope_series(
    sessions: pd.DataFrame,
    column: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    sport_filter: str | None = None,
    min_periods: int = 3,
) -> pd.DataFrame:
    """
    Enveloppe construite directement depuis sessions.parquet.

    `sport_filter` s'applique sur le debut du nom de discipline, en
    minuscules : "run" retient Run, "bike" retient Bike mais pas VirtualRide,
    qui est une discipline distincte et doit etre demandee explicitement.
    """
    df = sessions
    if sport_filter:
        df = df[df["sport"].str.lower().str.startswith(sport_filter.lower())]
    spec = EnvelopeSpec(column=column, window_days=window_days,
                        min_periods=min_periods,
                        name=f"{column}_env{window_days}")
    return rolling_envelope(df, spec)


def describe_predictability(series: pd.Series | np.ndarray, lag: int = 1) -> dict:
    """
    Caracterise une serie avant de tenter de la modeliser.

    Une autocorrelation proche de zero signale du bruit autour d'un niveau :
    inutile d'ajuster quoi que ce soit dessus, le probleme est l'observable.
    """
    from ..evaluation.metrics import autocorrelation

    x = pd.Series(np.asarray(series, dtype=float)).dropna()
    return {
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else float("nan"),
        "std": float(x.std()) if x.size else float("nan"),
        "autocorr_lag1": autocorrelation(x, lag),
        "coefficient_of_variation": float(x.std() / x.mean()) if x.size and x.mean() else float("nan"),
    }
