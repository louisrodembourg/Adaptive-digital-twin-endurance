"""
Metriques d'evaluation.

Volontairement separees des modeles : une metrique qui vit dans le module
qu'elle note finit par etre choisie pour le flatter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def rmse(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")


def mae(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def r2(y_true, y_pred, baseline_mean: float | None = None) -> float:
    """
    Coefficient de determination.

    `baseline_mean` doit valoir la moyenne du jeu d'ENTRAINEMENT quand on
    evalue hors echantillon. Utiliser celle du jeu de test revient a donner
    au modele une information qu'il n'avait pas, et gonfle le score.
    """
    a, b = _clean(y_true, y_pred)
    if a.size < 2:
        return float("nan")
    ref = a.mean() if baseline_mean is None else float(baseline_mean)
    ss_tot = float(((a - ref) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(((a - b) ** 2).sum()) / ss_tot


def skill_score(y_true, y_pred, y_baseline) -> float:
    """
    Gain relatif par rapport a une reference, en termes d'erreur quadratique.

    Positif : le modele fait mieux que la reference. Negatif : moins bien.
    C'est la lecture qui compte ici, davantage que le R2 : un modele peut
    avoir un R2 honorable et rester battu par « predire la derniere valeur ».
    """
    a, b = _clean(y_true, y_pred)
    _, c = _clean(y_true, y_baseline)
    if a.size < 2 or c.size != a.size:
        return float("nan")
    mse_base = float(((a - c) ** 2).mean())
    if mse_base <= 0:
        return float("nan")
    return 1.0 - float(((a - b) ** 2).mean()) / mse_base


def interval_coverage(y_true, lower, upper) -> float:
    """Part des observations tombant dans l'intervalle annonce."""
    a = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    ok = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    return float(((a >= lo) & (a <= hi))[ok].mean()) if ok.any() else float("nan")


def autocorrelation(series, lag: int = 1) -> float:
    """
    Autocorrelation d'une serie.

    Sert a juger si une variable est seulement predictible : une serie dont
    l'autocorrelation est nulle est du bruit autour d'un niveau, et aucun
    modele ne pourra en tirer quoi que ce soit.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def paired_block_bootstrap(y_true, y_a, y_b, blocks, n_boot: int = 10000,
                           seed: int = 0) -> dict:
    """
    Compare deux predictions sur les memes points, par bootstrap apparie.

    Deux variantes evaluees sous le meme decoupage ne sont pas deux
    echantillons independants : elles predisent exactement les memes points.
    Comparer leurs R2 agreges ne dit donc pas si l'ecart depasse le bruit, et
    un ecart de 0,003 de R2 ne veut rien dire tant qu'on ne l'a pas borne.

    Le reechantillonnage porte sur des BLOCS — les plis de validation — et non
    sur des points. Les erreurs d'un meme pli sont fortement autocorrelees :
    tirer des jours independamment supposerait 500 observations independantes
    la ou il y en a 18, et donnerait un intervalle faussement etroit.

    Renvoie le gain de `a` sur `b` au sens du score de competence
    (positif : `a` fait moins d'erreur), son intervalle a 95 %, et la part des
    reechantillonnages ou `a` gagne.
    """
    a = np.asarray(y_true, dtype=float)
    pa = np.asarray(y_a, dtype=float)
    pb = np.asarray(y_b, dtype=float)
    blk = np.asarray(blocks)
    ok = np.isfinite(a) & np.isfinite(pa) & np.isfinite(pb)
    a, pa, pb, blk = a[ok], pa[ok], pb[ok], blk[ok]
    if a.size < 2:
        return {"skill": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_a_better": float("nan"), "n_blocks": 0}

    groups = [np.flatnonzero(blk == b) for b in pd.unique(blk)]
    rng = np.random.default_rng(seed)

    def skill(idx: np.ndarray) -> float:
        mse_b = float(((a[idx] - pb[idx]) ** 2).mean())
        if mse_b <= 0:
            return float("nan")
        return 1.0 - float(((a[idx] - pa[idx]) ** 2).mean()) / mse_b

    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picked = rng.integers(0, len(groups), len(groups))
        draws[i] = skill(np.concatenate([groups[j] for j in picked]))

    finite = draws[np.isfinite(draws)]
    return {
        "skill": skill(np.arange(a.size)),
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
        "p_a_better": float((finite > 0).mean()) if finite.size else float("nan"),
        "n_blocks": len(groups),
    }
