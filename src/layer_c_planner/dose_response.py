"""
Verdict de dose-reponse : ce planificateur sait-il quelque chose de CET athlete ?

Pourquoi ce module existe
-------------------------
La mesure a etabli trois choses chez les treize athletes du cache. La charge
d'entrainement explique la performance chez quatre d'entre eux, avec un
intervalle qui exclut zero. Elle n'explique rien de mesurable chez les neuf
autres, dont les estimations ponctuelles sont centrees sur -0,054 — nulles, et
non positives-mais-noyees. Et aucune variable disponible a l'inscription ne
separe les deux groupes.

Un planificateur qui suppose une dose-reponse se trompe donc chez environ deux
athletes sur trois, sans que rien ne l'avertisse. La seule reponse honnete est
de MESURER chez l'athlete qu'on a en face, et de ne revendiquer un effet de
charge que si son propre intervalle exclut zero.

Trois verdicts, pas deux
------------------------
« Non detectee » et « indeterminable » ne disent pas la meme chose et ne
s'adressent pas au meme interlocuteur. Le premier signifie qu'on a regarde
avec assez de donnees et qu'il n'y a rien a voir ; le second, qu'il n'y en a
pas assez pour regarder. Les confondre reviendrait a presenter une absence de
mesure comme une absence d'effet, ce qui est exactement l'erreur que ce depot
s'emploie a ne pas commettre.

Aucun des trois n'autorise a maximiser une performance predite. Seul le
premier autorise a en parler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..evaluation.foundation import build_forecast_context
from ..evaluation.metrics import paired_block_bootstrap
from ..evaluation.report import evaluate_results, specific_variant
from ..evaluation.validation import horizon_splits

DETECTED = "detectee"
ABSENT = "non detectee"
UNDETERMINED = "indeterminable"


@dataclass(frozen=True)
class DoseResponse:
    """
    Ce que la mesure dit de la relation charge -> performance chez un athlete.

    `skill` est la competence appariee du modele de charge contre la
    reference, et l'intervalle est celui d'un bootstrap par blocs sur les
    plis — le meme que publie le reste du depot. C'est l'intervalle qui
    decide, pas le point.
    """

    verdict: str
    skill: float = math.nan
    ci_low: float = math.nan
    ci_high: float = math.nan
    folds: int = 0
    reference: str = "moyenne"
    load_column: str = ""
    reason: str = ""

    @property
    def detected(self) -> bool:
        return self.verdict == DETECTED

    @property
    def may_claim_performance_effect(self) -> bool:
        """
        Le planificateur a-t-il le droit de parler d'un effet de charge ?

        Un seul verdict l'autorise. Ni « non detectee » ni « indeterminable »
        ne le font, et pour des raisons differentes qu'il faut dire a
        l'utilisateur plutot que de lui livrer un plan qui tranche a sa place.
        """
        return self.detected

    def describe(self) -> str:
        if self.verdict == UNDETERMINED:
            return f"Dose-reponse indeterminable — {self.reason}"
        interval = f"[{self.ci_low:+.3f} · {self.ci_high:+.3f}]"
        head = ("Dose-reponse detectee" if self.detected
                else "Dose-reponse non detectee")
        return (f"{head} chez cet athlete : competence {self.skill:+.3f} "
                f"contre « {self.reference} », IC 95 % {interval}, "
                f"{self.folds} plis")


def assess_dose_response(
    observations: pd.DataFrame,
    daily_load: pd.DataFrame,
    value_column: str = "pi",
    load_column: str = "trimp_bike",
    reference: str = "moyenne",
    tau1: float = 42.0,
    tau2: float = 12.0,
    horizon_days: int = 28,
    min_train_days: int = 180,
    window_days: int = 42,
    min_periods: int = 3,
    min_folds: int = 8,
    seed: int = 0,
) -> DoseResponse:
    """
    Mesure la dose-reponse chez un athlete, par le protocole deja publie.

    Rien n'est reimplemente ici : l'enveloppe, le decoupage a horizon
    calendaire, les variantes et le bootstrap apparie sont ceux de
    `cli evaluate`. Si ce verdict et le tableau publie divergeaient, l'un des
    deux mentirait.

    Ne leve pas quand les donnees manquent : rend `indeterminable` avec sa
    raison. Un planificateur doit pouvoir repondre « je ne sais pas » sans
    s'interrompre.
    """
    if observations.empty or value_column not in observations.columns:
        return DoseResponse(UNDETERMINED, load_column=load_column,
                            reason="aucune observation de performance")
    if load_column not in daily_load.columns:
        return DoseResponse(
            UNDETERMINED, load_column=load_column,
            reason=f"colonne de charge {load_column!r} absente — relancer `cli ingest`")

    source = (observations[["date", value_column]].dropna()
              .rename(columns={value_column: "pi"}))
    if len(source) < 3:
        return DoseResponse(UNDETERMINED, load_column=load_column,
                            reason=f"{len(source)} observations, il en faut au moins trois")

    context = build_forecast_context(source, daily_load, "pi",
                                     window_days=window_days,
                                     min_periods=min_periods)
    dense = (context.frame[[context.target_column]].dropna()
             .rename(columns={context.target_column: "perf"}).reset_index())
    splits = list(horizon_splits(dense["date"], horizon_days=horizon_days,
                                 min_train_days=min_train_days,
                                 step_days=horizon_days))
    if len(splits) < min_folds:
        # Compter les jours qui manquent, et pas seulement les plis :
        # « indeterminable » ne dit rien a l'athlete, « il manque 224 jours »
        # lui dit quoi attendre.
        missing_days = (min_folds - len(splits)) * horizon_days
        return DoseResponse(
            UNDETERMINED, load_column=load_column, folds=len(splits),
            reason=(f"{len(splits)} pli(s) pour un minimum de {min_folds} — "
                    f"il manque environ {missing_days} jour(s) d'historique"))

    try:
        variant = specific_variant(load_column, tau1, tau2)
    except ValueError as exc:
        return DoseResponse(UNDETERMINED, load_column=load_column, reason=str(exc))

    results = evaluate_results(dense, daily_load, target="perf",
                               taus=[(tau1, tau2)], splits=splits)
    model, other = results.get(variant), results.get(reference)
    if model is None or other is None or model.y_true.size == 0:
        return DoseResponse(
            UNDETERMINED, load_column=load_column, reference=reference,
            reason=f"la variante « {variant} » n'a rien pu evaluer")
    if model.y_true.shape != other.y_true.shape:
        return DoseResponse(
            UNDETERMINED, load_column=load_column, reference=reference,
            reason="variante et reference n'ont pas couvert les memes points")

    stats = paired_block_bootstrap(model.y_true, model.y_pred, other.y_pred,
                                   model.blocks, seed=seed)
    # C'est l'intervalle qui decide. Un gain ponctuel positif avec un
    # intervalle qui contient zero ne vaut pas mieux qu'un gain nul : c'est
    # la regle que le reste du depot applique, et elle vaut ici aussi.
    verdict = DETECTED if stats["ci_low"] > 0.0 else ABSENT
    return DoseResponse(
        verdict=verdict,
        skill=float(stats["skill"]),
        ci_low=float(stats["ci_low"]),
        ci_high=float(stats["ci_high"]),
        folds=model.n_folds,
        reference=reference,
        load_column=load_column,
    )
