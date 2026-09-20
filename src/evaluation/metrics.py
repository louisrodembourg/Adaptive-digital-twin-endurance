"""
Metriques d'evaluation.

Volontairement separees des modeles : une metrique qui vit dans le module
qu'elle note finit par etre choisie pour le flatter.
"""

from __future__ import annotations

import numpy as np


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
