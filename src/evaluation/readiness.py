"""
Maturite : ce que le systeme peut honnetement affirmer aujourd'hui.

Le probleme
-----------
Le verdict de dose-reponse demande 180 jours de contexte puis huit plis de
28 jours, soit plus de quatre cents jours d'historique. Avant cela
`assess_dose_response` rend « indeterminable », ce qui est exact et inutile :
un nouvel utilisateur n'entend rien pendant plus d'un an, sans savoir ni
pourquoi, ni combien de temps encore.

Or les capacites du systeme n'apparaissent pas toutes en meme temps. La charge
existe des la premiere seance ; la borne aigu/chronique demande 28 jours ; les
seuils de fraicheur, 30 ; l'enveloppe de performance, trois mesures dans une
fenetre de 42 jours ; une premiere prevision evaluable, un pli ; un verdict,
huit. Ce module rend cette echelle, avec pour chaque barreau ce qui manque
exprime en jours ou en mesures.

Aucun seuil n'est invente ici
-----------------------------
Chacun est celui qu'une fonction impose vraiment, et les comptages passent par
ces fonctions plutot que de les reimplementer : `horizon_splits` pour les
plis, `rolling_envelope` pour l'enveloppe, les valeurs par defaut de
`assess_dose_response` pour le verdict. Si un seuil change la, il change ici.

Ce que le systeme peut dire au premier jour
-------------------------------------------
Rien sur l'athlete — mais quelque chose de mesure sur la population : la
charge explique la puissance chez quatre athletes du cache sur treize, et pas
chez les neuf autres. Ce prior se lit dans l'artefact de cohorte quand il
existe, et n'est jamais recopie en dur : un chiffre de population recopie
devient faux des que la cohorte grandit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..layer_a_signal.envelope import EnvelopeSpec, rolling_envelope
from .validation import horizon_splits

#: Jours de calendrier complets exiges par `optimizer._chronic_daily_load`
#: pour que le denominateur du rapport aigu/chronique existe.
CHRONIC_DAYS = 28

#: Points de fraicheur exiges par `load_model.form_thresholds` avant de
#: calibrer des seuils propres a l'athlete.
FORM_DAYS = 30

#: Constante de fitness par defaut : avant ce delai la moyenne mobile
#: exponentielle n'a pas converge, donc la « forme » affichee depend encore
#: surtout de sa valeur initiale.
FITNESS_SETTLING_DAYS = 42


@dataclass(frozen=True)
class Capability:
    """Une capacite du systeme, et ce qui lui manque le cas echeant."""

    name: str
    available: bool
    detail: str
    missing: str = ""

    def describe(self) -> str:
        mark = "oui" if self.available else "non"
        line = f"  [{mark:>3s}] {self.name:34s} {self.detail}"
        return line if self.available or not self.missing else f"{line}\n{'':42s}manque : {self.missing}"


@dataclass(frozen=True)
class Readiness:
    """L'echelle complete, a une date donnee."""

    days: int
    observations: int
    envelope_days: int
    folds: int
    capabilities: tuple[Capability, ...]
    prior: str | None = None

    @property
    def available(self) -> tuple[Capability, ...]:
        return tuple(c for c in self.capabilities if c.available)

    def capability(self, name: str) -> Capability | None:
        return next((c for c in self.capabilities if c.name == name), None)


def _population_prior(cohort: pd.DataFrame | Path | str | None) -> str | None:
    """
    Ce que la cohorte dit, quand on ne peut encore rien dire de l'athlete.

    Lu dans l'artefact plutot que recopie : un chiffre de population en dur
    devient faux des que la cohorte grandit, et personne ne s'en apercoit.
    """
    if cohort is None:
        return None
    if isinstance(cohort, (str, Path)):
        path = Path(cohort)
        if not path.exists():
            return None
        try:
            cohort = pd.read_parquet(path)
        except Exception:
            return None
    if cohort.empty or "reference" not in cohort.columns:
        return None
    against_mean = cohort[cohort["reference"] == "moyenne"]
    if against_mean.empty:
        return None
    detected = int((against_mean["ci_low"] > 0).sum())
    total = len(against_mean)
    return (
        f"Chez {detected} athlete(s) sur {total} de la cohorte, la charge "
        "explique la performance avec un intervalle qui exclut zero ; chez "
        f"{total - detected}, non. Rien dans les donnees disponibles a "
        "l'inscription ne dit lequel vous etes."
    )


def assess_readiness(
    daily_load: pd.DataFrame,
    observations: pd.DataFrame,
    value_column: str = "pi",
    window_days: int = 42,
    min_periods: int = 3,
    horizon_days: int = 28,
    min_train_days: int = 180,
    min_folds: int = 8,
    cohort: pd.DataFrame | Path | str | None = None,
) -> Readiness:
    """
    Dresse l'echelle des capacites a partir des donnees reellement presentes.

    Ne leve pas sur des donnees vides : un systeme qui demarre doit pouvoir
    decrire son propre vide.
    """
    days = 0 if daily_load.empty else len(daily_load)
    source = pd.DataFrame(columns=["date", value_column])
    if not observations.empty and value_column in observations.columns:
        source = (observations[["date", value_column]].dropna()
                  .rename(columns={value_column: "pi"}))
    n_obs = len(source)

    envelope = rolling_envelope(
        source, EnvelopeSpec("pi", window_days, min_periods=min_periods),
        calendar="daily") if n_obs else pd.DataFrame()
    envelope_days = len(envelope)

    folds = 0
    if envelope_days:
        folds = len(list(horizon_splits(
            envelope["date"], horizon_days=horizon_days,
            min_train_days=min_train_days, step_days=horizon_days)))

    def days_short(needed: int) -> str:
        return f"{needed - days} jour(s) de calendrier" if days < needed else ""

    capabilities = (
        Capability(
            "charge (fitness, fatigue)",
            available=days >= 1,
            detail=(f"{days} jour(s) de calendrier"
                    + (f", fitness stabilisee depuis {days - FITNESS_SETTLING_DAYS} j"
                       if days >= FITNESS_SETTLING_DAYS
                       else f", pas encore stabilisee (t1 = {FITNESS_SETTLING_DAYS} j)")),
            missing=days_short(1),
        ),
        Capability(
            "borne aigu/chronique",
            available=days >= CHRONIC_DAYS,
            detail=f"exige {CHRONIC_DAYS} jours pleins de charge chronique",
            missing=days_short(CHRONIC_DAYS),
        ),
        Capability(
            "seuils de fraicheur propres",
            available=days >= FORM_DAYS,
            detail=f"exige {FORM_DAYS} points de fraicheur",
            missing=days_short(FORM_DAYS),
        ),
        Capability(
            "enveloppe de performance",
            available=envelope_days > 0,
            detail=(f"{envelope_days} jour(s) definis sur {n_obs} mesure(s)"
                    if envelope_days else f"{n_obs} mesure(s)"),
            missing=("" if envelope_days else
                     f"{max(0, min_periods - n_obs)} mesure(s) dans une fenetre "
                     f"de {window_days} jours"),
        ),
        Capability(
            "prevision evaluable",
            available=folds >= 1,
            detail=f"{folds} pli(s) d'horizon {horizon_days} j",
            missing=("" if folds >= 1 else
                     f"{max(0, min_train_days + horizon_days - envelope_days)} "
                     "jour(s) d'enveloppe definie"),
        ),
        Capability(
            "verdict de dose-reponse",
            available=folds >= min_folds,
            detail=f"exige {min_folds} plis, {folds} disponible(s)",
            missing=("" if folds >= min_folds else
                     f"{(min_folds - folds) * horizon_days} jour(s) "
                     f"({min_folds - folds} pli(s))"),
        ),
    )
    return Readiness(
        days=days,
        observations=n_obs,
        envelope_days=envelope_days,
        folds=folds,
        capabilities=capabilities,
        prior=_population_prior(cohort),
    )


def format_readiness(readiness: Readiness) -> str:
    """Rend l'echelle lisible, du disponible vers ce qui manque."""
    lines = [
        f"Maturite : {len(readiness.available)}/{len(readiness.capabilities)} "
        "capacites disponibles",
        "",
    ]
    lines += [c.describe() for c in readiness.capabilities]

    verdict = readiness.capability("verdict de dose-reponse")
    if verdict is not None and not verdict.available and readiness.prior:
        # C'est ici que le prior de population sert : l'athlete n'a pas encore
        # son propre verdict, et se taire reviendrait a laisser croire qu'il
        # n'y a rien a dire.
        lines += ["", "  En attendant son propre verdict :",
                  "    " + readiness.prior.replace(" ; ", " ;\n    ")]
    return "\n".join(lines)
