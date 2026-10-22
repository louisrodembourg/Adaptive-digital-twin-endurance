"""
Plancher de detectabilite : quel effet ce protocole SAIT voir.

Le depot conclut a l'absence d'effet de la charge sur la performance. Cette
conclusion est incomplete tant qu'une question reste sans reponse : **de
quelle taille un effet devrait-il etre pour que ce protocole le remarque ?**
Sans ce nombre, « on n'a rien trouve » et « il n'y a rien » sont
indistinguables, et c'est le premier reproche qu'un lecteur attentif fera au
resultat nul.

La methode est une etude de sensibilite : on ajoute a l'enveloppe REELLE un
effet de charge SYNTHETIQUE d'amplitude connue, de la forme exacte que le
modele postule, puis on repasse le protocole inchange. Le plus petit delta
dont l'intervalle apparie exclut zero est le plancher. On peut alors ecrire
« ce protocole exclut tout effet superieur a delta* », ce qui est une
affirmation falsifiable, la ou « rien trouve » n'en est pas une.

Ce que ce module n'est pas
--------------------------
Il ne fabrique aucune donnee d'entrainement et n'ecrit rien dans `data/`.
L'enveloppe perturbee n'existe qu'en memoire, le temps d'un calcul, et
l'interface ne rend que des statistiques — jamais la serie elle-meme, pour
qu'aucune sortie ne puisse etre confondue avec une mesure. La regle du depot
(« aucune donnee inventee ne doit pouvoir passer pour mesuree ») porte sur
les seances et les observations ; un plancher de detectabilite est une
propriete de l'estimateur, pas une donnee de l'athlete.

Pourquoi le plancher est OPTIMISTE
----------------------------------
Le signal injecte a precisement la forme que `banister_predictor` ajuste :
une difference de deux moyennes mobiles exponentielles aux memes constantes
de temps. C'est le cas le plus favorable imaginable. Un effet reel de forme
differente devrait etre PLUS grand pour etre vu. Le plancher est donc une
borne inferieure sur ce qu'il faudrait, et non une garantie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..layer_b_twin.load_model import compute_load_curves_by_sport
from .metrics import paired_block_bootstrap
from .report import evaluate_results, specific_variant
from .validation import horizon_splits

#: Amplitudes testees, en ecarts-types de l'enveloppe. delta = 1 signifie un
#: effet de charge aussi dispersé que toute l'enveloppe observee : c'est
#: enorme, et le fait de devoir monter si haut est en soi le resultat.
DEFAULT_DELTAS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class Floor:
    """Plancher atteint pour une reference donnee."""

    reference: str
    delta: float | None
    deltas_tested: tuple[float, ...]

    @property
    def found(self) -> bool:
        return self.delta is not None

    @property
    def detected_without_injection(self) -> bool:
        """
        Vrai quand l'effet est deja vu a delta = 0.

        C'est un cas a distinguer : il ne dit pas que le protocole est
        sensible, il dit qu'il a trouve quelque chose dans les donnees
        REELLES, sans qu'on ait rien ajoute.
        """
        return self.delta == 0.0

    def describe(self) -> str:
        if not self.found:
            return (f"contre {self.reference} : aucun plancher atteint jusqu'a "
                    f"delta = {max(self.deltas_tested):.2f} ecart-type")
        if self.detected_without_injection:
            return (f"contre {self.reference} : effet deja detecte SANS injection "
                    "— la charge reelle suffit")
        return (f"contre {self.reference} : plancher a delta = {self.delta:.2f} "
                "ecart-type d'enveloppe")


def injected_signal(daily_load: pd.DataFrame, dates: pd.Series,
                    tau1: float = 42.0, tau2: float = 12.0,
                    column: str = "trimp_run") -> np.ndarray:
    """
    Le signal de charge standardise, de la forme que le modele postule.

    `column` choisit la discipline : elle doit etre celle que consomme le
    modele dont on mesure le plancher.

    `fitness - fatigue` aux memes constantes de temps que le prediteur : on
    donne au modele exactement ce qu'il cherche. Standardise pour que
    `delta` s'exprime en ecarts-types d'enveloppe et reste comparable d'une
    duree a l'autre.

    La charge de COURSE est prise par defaut, pas la charge totale : c'est la
    seule dont on puisse supposer qu'elle agit sur une vitesse en course a
    pied, et le protocole doit donc etre teste sur sa capacite a la voir.
    """
    curves = compute_load_curves_by_sport(daily_load, tau1, tau2, (column,))
    family = column.removeprefix("trimp_")
    idx = curves.set_index("date")
    rows = idx.reindex(pd.to_datetime(pd.Series(dates)).dt.normalize())
    signal = (rows[f"fitness_{family}"] - rows[f"fatigue_{family}"]).to_numpy(dtype=float)
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        raise ValueError(f"Signal de charge vide pour {column!r}.")
    centred = signal - finite.mean()
    scale = finite.std() or 1.0
    return centred / scale


def detectability_curve(
    dense: pd.DataFrame,
    daily_load: pd.DataFrame,
    deltas: tuple[float, ...] = DEFAULT_DELTAS,
    target: str = "perf",
    tau1: float = 42.0,
    tau2: float = 12.0,
    references: tuple[str, ...] = ("moyenne", "persistance"),
    horizon_days: int = 28,
    min_train_days: int = 180,
    column: str = "trimp_run",
    variant: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Repasse le protocole sur l'enveloppe additionnee d'un effet connu.

    Le decoupage, les variantes et le bootstrap sont ceux du protocole reel :
    seule la cible change. C'est ce qui rend le plancher interpretable — il
    mesure le protocole tel qu'il est publie, pas une version simplifiee.
    """
    frame = dense.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    y = frame[target].to_numpy(dtype=float)
    sigma = float(np.nanstd(y)) or 1.0
    signal = injected_signal(daily_load, frame["date"], tau1, tau2, column)
    splits = list(horizon_splits(frame["date"], horizon_days=horizon_days,
                                 min_train_days=min_train_days,
                                 step_days=horizon_days))
    # Le modele teste doit consommer la MEME charge que celle injectee :
    # mesurer le plancher d'un modele nourri de course quand l'effet est
    # injecte depuis le velo ne dirait rien du protocole.
    if variant is None:
        try:
            variant = specific_variant(column, tau1, tau2)
        except ValueError as exc:
            raise ValueError(f"{exc} Passer `variant` explicitement.") from None
    specific = variant

    rows = []
    for delta in deltas:
        perturbed = frame.copy()
        # L'effet injecte s'ajoute a l'enveloppe reelle ; `delta` est en
        # ecarts-types d'enveloppe pour que le nombre publie soit lisible
        # sans connaitre l'unite de l'index de performance.
        perturbed[target] = y + delta * sigma * signal
        results = evaluate_results(perturbed, daily_load, target=target,
                                   taus=[(tau1, tau2)], splits=splits)
        model = results.get(specific)
        if model is None or model.y_true.size == 0:
            continue
        for reference in references:
            other = results.get(reference)
            if other is None or other.y_true.shape != model.y_true.shape:
                continue
            stats = paired_block_bootstrap(model.y_true, model.y_pred,
                                           other.y_pred, model.blocks, seed=seed)
            rows.append({
                "delta": float(delta),
                "reference": reference,
                "r2_modele": model.r2_out_of_sample,
                "r2_reference": other.r2_out_of_sample,
                **stats,
                "detecte": bool(stats["ci_low"] > 0.0),
            })
    return pd.DataFrame(rows)


def detection_floor(table: pd.DataFrame, reference: str) -> Floor:
    """Le plus petit delta dont l'intervalle apparie exclut zero."""
    subset = table[table["reference"] == reference].sort_values("delta")
    tested = tuple(float(d) for d in subset["delta"])
    hits = subset[subset["detecte"]]
    delta = float(hits.iloc[0]["delta"]) if not hits.empty else None
    return Floor(reference=reference, delta=delta, deltas_tested=tested)


def format_detectability(table: pd.DataFrame) -> str:
    """Rend la courbe de detectabilite lisible en terminal."""
    if table.empty:
        return "Aucune amplitude evaluable."
    lines = [
        f"{'delta':>7s} {'contre':12s} {'R2 modele':>10s} {'R2 ref.':>9s} "
        f"{'gain':>8s} {'IC 95 %':>20s} {'P(gain)':>8s} {'vu':>4s}",
        "-" * 84,
    ]
    for _, r in table.iterrows():
        interval = f"[{r['ci_low']:+.4f} · {r['ci_high']:+.4f}]"
        lines.append(
            f"{r['delta']:7.2f} {r['reference']:12s} {r['r2_modele']:+10.4f} "
            f"{r['r2_reference']:+9.4f} {r['skill']:+8.4f} {interval:>20s} "
            f"{r['p_a_better']:8.3f} {'oui' if r['detecte'] else 'non':>4s}"
        )
    lines.append("")
    for reference in table["reference"].unique():
        lines.append("  " + detection_floor(table, reference).describe())
    return "\n".join(lines)
