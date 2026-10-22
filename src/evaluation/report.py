"""
Rapport d'evaluation : confronter chaque variante aux memes references.

Le tableau produit place cote a cote le score d'ajustement et le score hors
echantillon. C'est l'ecart entre les deux colonnes qui informe : un modele
dont le R2 d'ajustement est honorable et le R2 hors echantillon negatif n'a
rien appris, il a memorise.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from ..layer_b_twin.load_model import compute_load_curves
from .baselines import default_baselines
from .foundation import ForecastContext, Timing, build_foundation_variants
from .metrics import paired_block_bootstrap
from .validation import Predictor, ValidationResult, compare_results


def banister_predictor(daily_load: pd.DataFrame, tau1: float = 42.0,
                       tau2: float = 12.0, with_fatigue: bool = True) -> Predictor:
    """
    Fabrique une variante Banister ajustable par moindres carres.

    Les courbes de charge se calculent une fois sur tout l'historique — ce
    n'est pas une fuite : elles ne dependent que de l'entrainement passe, pas
    des performances a predire. Seuls p0, k1 et k2 sont ajustes par pli.
    """
    curves = compute_load_curves(daily_load, tau1, tau2).to_frame()
    f_mean, f_std = curves.fitness.mean(), curves.fitness.std() or 1.0
    g_mean, g_std = curves.fatigue.mean(), curves.fatigue.std() or 1.0
    idx = curves.set_index("date")

    def design(dates: pd.Series) -> np.ndarray:
        rows = idx.reindex(pd.to_datetime(pd.Series(dates)).dt.normalize())
        cols = [np.ones(len(rows)), ((rows.fitness - f_mean) / f_std).to_numpy()]
        if with_fatigue:
            cols.append(-((rows.fatigue - g_mean) / g_std).to_numpy())
        return np.column_stack(cols)

    def predict(dates_tr, y_tr, dates_te):
        X, y = design(dates_tr), np.asarray(y_tr, dtype=float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < X.shape[1] + 2:
            return np.full(len(dates_te), np.nan)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        return design(dates_te) @ beta

    return predict


def _run_baseline(baseline, dates_tr, y_tr, dates_te) -> np.ndarray:
    return baseline.fit_predict(dates_tr, y_tr, dates_te)


def build_variants(daily_load: pd.DataFrame,
                   taus: list[tuple[float, float]] | None = None,
                   context: ForecastContext | None = None,
                   models: tuple[str, ...] = (),
                   device: str = "cpu",
                   timings: dict[str, Timing] | None = None) -> dict[str, Predictor]:
    """
    Assemble references et variantes de modele sous une interface commune.

    Les references viennent en premier et ne sont jamais retirees : c'est ce
    qui empeche de presenter comme un succes un modele qui perd contre
    « predire la moyenne ».

    `context` ouvre les variantes a modele de fondation. Sans lui le tableau
    est exactement celui d'avant, au chiffre pres : les modeles de fondation
    ont besoin d'une grille journaliere que l'enveloppe rapportee aux seules
    dates d'observation ne fournit pas.
    """
    def _as_predictor(baseline) -> Predictor:
        # functools.partial lie la reference explicitement ; une lambda
        # capturerait la variable de boucle, donc la derniere reference.
        return functools.partial(_run_baseline, baseline)

    variants: dict[str, Predictor] = {
        b.name: _as_predictor(b) for b in default_baselines()
    }
    for tau1, tau2 in taus or [(42.0, 12.0)]:
        variants[f"banister t{tau1:.0f}/{tau2:.0f}"] = banister_predictor(daily_load, tau1, tau2)
        variants[f"fitness seul t{tau1:.0f}"] = banister_predictor(
            daily_load, tau1, tau2, with_fatigue=False)
    if context is not None and models:
        variants.update(build_foundation_variants(context, models, device, timings))
    return variants


def evaluate_results(observations: pd.DataFrame, daily_load: pd.DataFrame,
             target: str = "perf_index", n_splits: int = 5,
             taus: list[tuple[float, float]] | None = None,
             context: ForecastContext | None = None,
             models: tuple[str, ...] = (),
             splits: list | None = None,
             device: str = "cpu",
             timings: dict[str, Timing] | None = None) -> pd.DataFrame:
    """
    Passe toutes les variantes sous le meme protocole.

    `splits` impose un decoupage construit a l'avance — `horizon_splits` pour
    le protocole a horizon calendaire. Il est passe tel quel a toutes les
    variantes : c'est la seule facon de garder les lignes comparables.
    """
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    obs = obs.sort_values("date").dropna(subset=[target]).reset_index(drop=True)
    variants = build_variants(daily_load, taus, context, models, device, timings)
    return compare_results(variants, obs["date"], obs[target].to_numpy(),
                           n_splits=n_splits, splits=splits)


def results_table(results: dict[str, ValidationResult]) -> pd.DataFrame:
    """Range les resultats en tableau, classe sur le score hors echantillon."""
    rows = [r.to_dict() for r in results.values()]
    return pd.DataFrame(rows).sort_values("r2_out", ascending=False).reset_index(drop=True)


def evaluate(observations: pd.DataFrame, daily_load: pd.DataFrame,
             target: str = "perf_index", n_splits: int = 5,
             taus: list[tuple[float, float]] | None = None,
             context: ForecastContext | None = None,
             models: tuple[str, ...] = (),
             splits: list | None = None,
             device: str = "cpu",
             timings: dict[str, Timing] | None = None) -> pd.DataFrame:
    """Le tableau seul, pour les appelants qui n'ont pas besoin des plis."""
    return results_table(evaluate_results(
        observations, daily_load, target, n_splits, taus, context, models,
        splits, device, timings))


def paired_table(results: dict[str, ValidationResult],
                 pairs: list[tuple[str, str]], seed: int = 0) -> pd.DataFrame:
    """
    Compare des variantes deux a deux sur les memes points.

    Soustraire deux R2 agreges ne dit pas si l'ecart depasse le bruit. Ici
    chaque paire passe par un bootstrap apparie par blocs, et l'intervalle
    rendu est ce qui autorise — ou interdit — d'affirmer un gain.
    """
    rows = []
    for a, b in pairs:
        ra, rb = results.get(a), results.get(b)
        if ra is None or rb is None or ra.y_true.size == 0 or rb.y_true.size == 0:
            continue
        if ra.y_true.shape != rb.y_true.shape:
            raise ValueError(
                f"{a} et {b} n'ont pas ete evalues sur les memes points "
                "— la comparaison appariee n'a pas de sens."
            )
        stats = paired_block_bootstrap(ra.y_true, ra.y_pred, rb.y_pred, ra.blocks, seed=seed)
        rows.append({"variante": a, "reference": b, "delta_r2": ra.r2_out_of_sample
                     - rb.r2_out_of_sample, **stats})
    return pd.DataFrame(rows)


def format_pairs(table: pd.DataFrame) -> str:
    """Rend les comparaisons appariees lisibles en terminal."""
    if table.empty:
        return "Aucune comparaison appariee."
    lines = [
        f"{'variante':24s} {'contre':24s} {'d R2':>8s} {'gain':>8s} "
        f"{'IC 95 %':>20s} {'P(gain)':>8s}",
        "-" * 96,
    ]
    for _, r in table.iterrows():
        interval = f"[{r['ci_low']:+.4f} · {r['ci_high']:+.4f}]"
        lines.append(
            f"{r['variante']:24s} {r['reference']:24s} {r['delta_r2']:+8.4f} "
            f"{r['skill']:+8.4f} {interval:>20s} {r['p_a_better']:8.3f}"
        )
    return "\n".join(lines)


def format_report(table: pd.DataFrame) -> str:
    """Rend le tableau lisible en terminal, trie par score hors echantillon."""
    lines = [
        f"{'variante':24s} {'R2 ajust.':>10s} {'R2 h.-ech.':>11s} "
        f"{'ecart':>8s} {'RMSE':>8s} {'gain/moy.':>10s} {'plis':>5s}",
        "-" * 82,
    ]
    for _, r in table.iterrows():
        # Le nombre de plis est affiche parce qu'il n'est pas toujours le meme :
        # un modele dont la tete est entrainee sur un horizon court echoue sur
        # les plis les plus longs, et sa ligne porte alors sur moins de points
        # que les autres. Sans cette colonne la comparaison paraitrait a armes
        # egales.
        lines.append(
            f"{r['name']:24s} {r['r2_in']:+10.4f} {r['r2_out']:+11.4f} "
            f"{r['gap']:+8.4f} {r['rmse_out']:8.5f} {r['skill_vs_mean']:+10.4f} "
            f"{int(r['folds']):5d}"
        )
    winners = table[table["r2_out"] > 0]
    lines.append("")
    if winners.empty:
        lines.append("Aucune variante ne fait mieux que predire la moyenne.")
    else:
        lines.append(f"Meilleure variante hors echantillon : {winners.iloc[0]['name']}")
    return "\n".join(lines)
