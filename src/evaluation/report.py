"""
Rapport d'evaluation : confronter chaque variante aux memes references.

Le tableau produit place cote a cote le score d'ajustement et le score hors
echantillon. C'est l'ecart entre les deux colonnes qui informe : un modele
dont le R2 d'ajustement est honorable et le R2 hors echantillon negatif n'a
rien appris, il a memorise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..layer_b_twin.load_model import compute_load_curves
from .baselines import default_baselines
from .validation import Predictor, compare


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


def build_variants(daily_load: pd.DataFrame,
                   taus: list[tuple[float, float]] | None = None) -> dict[str, Predictor]:
    """
    Assemble references et variantes de modele sous une interface commune.

    Les references viennent en premier et ne sont jamais retirees : c'est ce
    qui empeche de presenter comme un succes un modele qui perd contre
    « predire la moyenne ».
    """
    variants: dict[str, Predictor] = {
        b.name: (lambda b: lambda dtr, ytr, dte: b.fit_predict(dtr, ytr, dte))(b)
        for b in default_baselines()
    }
    for tau1, tau2 in taus or [(42.0, 12.0)]:
        variants[f"banister t{tau1:.0f}/{tau2:.0f}"] = banister_predictor(daily_load, tau1, tau2)
        variants[f"fitness seul t{tau1:.0f}"] = banister_predictor(
            daily_load, tau1, tau2, with_fatigue=False)
    return variants


def evaluate(observations: pd.DataFrame, daily_load: pd.DataFrame,
             target: str = "perf_index", n_splits: int = 5,
             taus: list[tuple[float, float]] | None = None) -> pd.DataFrame:
    """Passe toutes les variantes sous le meme protocole."""
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    obs = obs.sort_values("date").dropna(subset=[target]).reset_index(drop=True)
    return compare(build_variants(daily_load, taus), obs["date"],
                   obs[target].to_numpy(), n_splits=n_splits)


def format_report(table: pd.DataFrame) -> str:
    """Rend le tableau lisible en terminal, trie par score hors echantillon."""
    lines = [
        f"{'variante':24s} {'R2 ajust.':>10s} {'R2 h.-ech.':>11s} "
        f"{'ecart':>8s} {'RMSE':>8s} {'gain/moy.':>10s}",
        "-" * 76,
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{r['name']:24s} {r['r2_in']:+10.4f} {r['r2_out']:+11.4f} "
            f"{r['gap']:+8.4f} {r['rmse_out']:8.5f} {r['skill_vs_mean']:+10.4f}"
        )
    winners = table[table["r2_out"] > 0]
    lines.append("")
    if winners.empty:
        lines.append("Aucune variante ne fait mieux que predire la moyenne.")
    else:
        lines.append(f"Meilleure variante hors echantillon : {winners.iloc[0]['name']}")
    return "\n".join(lines)
