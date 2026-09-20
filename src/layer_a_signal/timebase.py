"""
Base de temps d'une seance.

Tout Layer A supposait implicitement une ligne par seconde, sans trou. Cette
hypothese tient a peu pres sur un export .fit continu, mais elle est fausse
des qu'il y a une pause, une mise en pause automatique ou une coupure de
capteur — ce qui est le cas courant dans GoldenCheetah OpenData, dont les CSV
portent une colonne `secs` explicite.

Consequence si on l'ignore : `len(df)` devient une duree, une seance de
90 minutes avec 20 minutes de pause est comptee comme 70 minutes de charge, et
une fenetre glissante enjambe un trou sans le savoir.

On calcule donc une fois pour toutes le pas de temps de chaque echantillon, et
les metriques s'expriment en integrales ponderees par ce pas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DT_COLUMN = "dt_s"
ELAPSED_COLUMN = "secs"


def attach_timebase(df: pd.DataFrame, max_gap_s: float = 30.0) -> pd.DataFrame:
    """
    Ajoute la colonne `dt_s` : duree representee par chaque echantillon.

    La source du temps, par ordre de preference :
      1. une colonne `secs` (secondes ecoulees, format OpenData) ;
      2. un index ou une colonne temporelle (`timestamp`) ;
      3. a defaut, l'hypothese 1 Hz — explicitement, pas par accident.

    Les ecarts superieurs a `max_gap_s` sont ramenes a cette borne : une pause
    de vingt minutes ne doit pas compter comme vingt minutes d'effort, mais la
    laisser a zero fausserait la continuite de la serie.
    """
    out = df.copy()
    elapsed = _elapsed_seconds(out)

    if elapsed is None:
        out[DT_COLUMN] = 1.0
        return out

    dt = np.diff(elapsed, prepend=elapsed[0] - 1.0)
    dt = np.where(np.isfinite(dt), dt, 1.0)
    dt = np.clip(dt, 0.0, max_gap_s)
    # Un pas nul (horodatages dupliques) ne porte aucune charge.
    out[DT_COLUMN] = dt
    return out


def _elapsed_seconds(df: pd.DataFrame) -> np.ndarray | None:
    if ELAPSED_COLUMN in df.columns:
        return pd.to_numeric(df[ELAPSED_COLUMN], errors="coerce").to_numpy(dtype=float)
    for col in ("timestamp", "time"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=True)
            if ts.notna().any():
                return (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    if isinstance(df.index, pd.DatetimeIndex):
        return (df.index - df.index[0]).total_seconds().to_numpy(dtype=float)
    return None


def dt_seconds(df: pd.DataFrame, max_gap_s: float = 30.0) -> pd.Series:
    """Pas de temps par echantillon, calcule a la volee si absent."""
    if DT_COLUMN in df.columns:
        return pd.to_numeric(df[DT_COLUMN], errors="coerce").fillna(0.0)
    return attach_timebase(df, max_gap_s)[DT_COLUMN]


def active_duration_s(df: pd.DataFrame, mask: pd.Series | None = None,
                      max_gap_s: float = 30.0) -> float:
    """Duree effective, en secondes — jamais `len(df)`."""
    dt = dt_seconds(df, max_gap_s)
    if mask is not None:
        dt = dt.where(mask.reindex(dt.index, fill_value=False), 0.0)
    return float(dt.sum())


def longest_contiguous_s(mask: pd.Series, dt: pd.Series) -> float:
    """
    Duree du plus long segment continu ou `mask` est vraie.

    `specific_ef` exigeait 60 secondes a l'allure cible mais comptait des
    lignes : soixante secondes eparpillees sur une heure et demie comptaient
    autant qu'une minute soutenue. Pour un marqueur cense mesurer la capacite
    a tenir une allure, la difference est tout le propos.
    """
    if mask.empty:
        return 0.0
    m = mask.fillna(False).to_numpy(dtype=bool)
    d = dt.reindex(mask.index).fillna(0.0).to_numpy(dtype=float)
    best = current = 0.0
    for flag, step in zip(m, d):
        current = current + step if flag else 0.0
        best = max(best, current)
    return float(best)
