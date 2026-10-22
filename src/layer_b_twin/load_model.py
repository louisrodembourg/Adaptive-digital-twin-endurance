"""
Modele de charge : fitness, fatigue et fraicheur.

Separe du modele de performance, dont la confusion produisait le defaut
central du projet. `compute_fitness_fatigue` renvoyait des sommes
d'impulsions brutes :

    fitness[t] = fitness[t-1] * exp(-1/tau) + trimp[t]

Il manque le facteur (1 - exp(-1/tau)). Sans lui la sortie n'est pas une
moyenne mobile exponentielle mais une somme cumulee amortie, qui croit avec
le volume d'entrainement au lieu de se stabiliser. Chez l'athlete du depot
elle atteignait 4871, et `perf = p0 + k1*fitness - k2*fatigue` donnait 941,57
la ou les consommateurs attendaient une valeur autour de 1,0 :

    estimated_time = record / 941,57            -> "0h00:05"
    prob_success   = 1 - cdf(1,05 ; mu=941,57)  -> 1.0, toujours

Avec le facteur, la sortie est une moyenne ponderee des charges passees,
exprimee dans la meme unite qu'elles (TRIMP par jour) et bornee par le niveau
d'entrainement habituel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter


@dataclass(frozen=True)
class LoadCurves:
    """Fitness, fatigue et fraicheur, un point par jour."""

    dates: pd.DatetimeIndex
    fitness: np.ndarray
    fatigue: np.ndarray
    tau1: float
    tau2: float

    @property
    def form(self) -> np.ndarray:
        """
        Fraicheur du jour : fitness et fatigue de la veille.

        On decale d'un jour parce que la charge du jour meme n'a pas encore
        produit son effet au moment ou l'on decide de s'entrainer.
        """
        return np.concatenate([[0.0], self.fitness[:-1] - self.fatigue[:-1]])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": self.dates,
            "fitness": self.fitness,
            "fatigue": self.fatigue,
            "form": self.form,
        })


def exponential_load(loads: np.ndarray, tau: float) -> np.ndarray:
    """
    Moyenne mobile exponentielle de la charge, de constante de temps tau.

        y[t] = y[t-1] * a + (1 - a) * x[t],  a = exp(-1/tau)

    Le facteur (1 - a) est ce qui manquait : il normalise la reponse, de sorte
    qu'une charge constante c converge vers c plutot que vers c/(1-a).

    Implemente par filtre recursif et non par boucle Python : cette fonction
    est appelee une fois par echantillon posterieur lors de la reconstruction
    des courbes, soit des centaines de milliers d'iterations.
    """
    if tau <= 0:
        raise ValueError(f"La constante de temps doit etre positive (recu {tau}).")
    x = np.asarray(loads, dtype=float)
    if x.size == 0:
        return x
    a = float(np.exp(-1.0 / tau))
    return lfilter([1.0 - a], [1.0, -a], x)


def compute_load_curves(
    daily_load: pd.DataFrame,
    tau1: float = 42.0,
    tau2: float = 12.0,
    trimp_column: str = "trimp",
) -> LoadCurves:
    """Fitness (tau long) et fatigue (tau court) depuis la charge quotidienne."""
    if daily_load.empty:
        return LoadCurves(pd.DatetimeIndex([]), np.array([]), np.array([]), tau1, tau2)
    if tau1 <= tau2:
        raise ValueError(
            f"La constante de fitness ({tau1}) doit depasser celle de fatigue "
            f"({tau2}) : sans cet ordre les deux courbes sont interchangeables."
        )

    dates = pd.DatetimeIndex(pd.to_datetime(daily_load["date"]))
    loads = pd.to_numeric(daily_load[trimp_column], errors="coerce").fillna(0.0).to_numpy()
    return LoadCurves(dates, exponential_load(loads, tau1),
                      exponential_load(loads, tau2), tau1, tau2)


def form_thresholds(form: np.ndarray, low_pct: float = 5.0,
                    high_pct: float = 95.0) -> tuple[float, float]:
    """
    Seuils de surmenage et de desentrainement, propres a l'athlete.

    Les valeurs -30 et +25 codees en dur sont des conventions TrainingPeaks
    pour le TSS, pas pour le TRIMP : les emprunter reviendrait a ajouter un
    troisieme nombre magique. On lit plutot les percentiles de la fraicheur
    observee chez cet athlete, ce qui donne un seuil interpretable pour
    n'importe qui.
    """
    finite = np.asarray(form, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 30:
        raise ValueError("Historique trop court pour calibrer des seuils de fraicheur.")
    return float(np.percentile(finite, low_pct)), float(np.percentile(finite, high_pct))


def acute_chronic_ratio(daily_load: pd.DataFrame, acute_days: int = 7,
                        chronic_days: int = 28, trimp_column: str = "trimp") -> pd.Series:
    """
    Rapport charge aigue sur charge chronique.

    Sert de regle de securite reellement declenchable : contrairement au
    rMSSD et au DFA, elle ne demande que la charge, qui existe toujours.
    """
    loads = pd.to_numeric(daily_load[trimp_column], errors="coerce").fillna(0.0)
    acute = loads.rolling(acute_days, min_periods=acute_days).mean()
    chronic = loads.rolling(chronic_days, min_periods=chronic_days).mean()
    return (acute / chronic.replace(0.0, np.nan)).rename("acwr")


# ── decomposition de la charge par discipline ─────────────────────────────
#
# Pourquoi decomposer : la cible du projet est un rapport de vitesses EN
# COURSE A PIED, et sur la fenetre d'evaluation 68,5 % du TRIMP vient du velo
# (64,7 % Bike + 3,8 % VirtualRide) contre 30,2 % de la course. Un modele a
# coefficient unique doit donc trouver un k1 qui vaille a la fois pour un
# kilojoule de velo et pour un kilometre de course. Le transfert velo ->
# course est partiel et de constante de temps differente, alors que la
# fatigue se cumule pleinement : un coefficient unique moyenne un transfert
# positif et un transfert quasi nul, et sort zero. C'est une explication
# mesurable du resultat nul du depot, distincte de « la charge n'explique
# rien ».
#
# Le total `trimp` reste la colonne de reference et n'est pas touche : les
# six lignes publiees dans docs/resultats.md en dependent et doivent rester
# reproductibles au chiffre pres.

SPORT_FAMILIES: dict[str, tuple[str, ...]] = {
    "run": ("run",),
    # VirtualRide est une discipline distincte dans GoldenCheetah mais la meme
    # charge physiologique qu'une sortie velo : on la regroupe ici, alors que
    # `envelope.envelope_series` la laisse separee parce qu'une VITESSE sur
    # home-trainer n'est pas comparable a une vitesse sur route.
    "bike": ("bike", "virtualride", "ride"),
}
OTHER_FAMILY = "other"

#: Colonnes de charge produites par `build_daily_load`, dans un ordre stable.
LOAD_COLUMNS: tuple[str, ...] = ("trimp_run", "trimp_bike", "trimp_other")


def sport_family(sport: object) -> str:
    """
    Range une discipline dans une famille de charge.

    La natation tombe dans `other` et non dans une famille propre : son TRIMP
    vaut 0,0 % du total parce que les fichiers de natation du jeu de donnees
    n'ont pas de canal de frequence cardiaque. Une colonne identiquement
    nulle n'apporte aucune colonne de regression, elle en casse une.
    """
    name = str(sport or "").strip().lower()
    for family, prefixes in SPORT_FAMILIES.items():
        if any(name.startswith(p) for p in prefixes):
            return family
    return OTHER_FAMILY


def compute_load_curves_by_sport(
    daily_load: pd.DataFrame,
    tau1: float = 42.0,
    tau2: float = 12.0,
    columns: tuple[str, ...] = LOAD_COLUMNS,
) -> pd.DataFrame:
    """
    Fitness et fatigue par discipline, sur le meme calendrier.

    Les constantes de temps sont partagees entre disciplines : les separer
    aussi ajouterait quatre parametres a estimer sur 90 observations, alors
    que la question posee ici est celle des COEFFICIENTS (un kilojoule de
    velo vaut-il autant qu'un kilometre de course ?), pas celle des delais.
    """
    missing = [c for c in columns if c not in daily_load.columns]
    if missing:
        raise ValueError(
            f"Colonnes de charge absentes : {missing}. "
            "Relancer `cli ingest` : `build_daily_load` les produit."
        )
    out = pd.DataFrame({"date": pd.DatetimeIndex(pd.to_datetime(daily_load["date"]))})
    for column in columns:
        family = column.removeprefix("trimp_")
        curves = compute_load_curves(daily_load, tau1, tau2, trimp_column=column)
        out[f"fitness_{family}"] = curves.fitness
        out[f"fatigue_{family}"] = curves.fatigue
    return out
