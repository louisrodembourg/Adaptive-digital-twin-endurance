"""
Index de performance : transformer une seance en une observation exploitable.

Le modele de Banister apprend en confrontant une charge accumulee a des
performances mesurees. Le projet n'avait jamais produit une seule observation
de performance — ce qui explique que le chemin PyMC soit reste du code mort :
`build_banister_model` attend `perf_observations`, et rien ne savait en
fabriquer.

Convention retenue
------------------
Un rapport de vitesses a duree egale, contre la meilleure courbe de l'athlete :

    pi = max sur d de  v_observee(d) / v_reference(d)

L'alternative naturelle — rapport de chronos sur une distance fixe — n'est
definie que les jours de course, soit une poignee d'observations par an, et
zero sur un jeu de donnees d'entrainement. Le rapport a duree egale est defini
pour chaque seance, ce qui donne une vraisemblance reellement alimentee.

pi = 1.0 signifie « au niveau de reference de l'athlete ». La courbe de
reference est ajustee une fois sur tout l'historique puis figee : la
recalculer a chaque execution ferait deriver le sens meme de la valeur 1.0.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .timebase import dt_seconds

DEFAULT_DURATIONS_S: tuple[int, ...] = (300, 600, 1200, 1800, 2700, 3600)

# Plafond de vraisemblance pour la course a pied, decroissant avec la duree.
# Un record du monde tient environ 7,5 m/s sur cinq minutes et 5,6 m/s sur une
# heure ; on borne sur cette enveloppe, avec la meme loi de Riegel que le
# reste du module.
#
# Le besoin est reel : dans le jeu OpenData, des seances etiquetees Run sont
# en fait des enchainements velo-course, ou portent un canal distance
# incoherent avec leur propre resume (26 km dans le CSV contre 10 km annonces
# par le meme fichier). Un plafond constant laissait encore passer 7,2 m/s sur
# quarante-cinq minutes, et prendre le maximum absolu suffisait a empoisonner
# toute la courbe — donc l'echelle entiere de l'index de performance.
MAX_PLAUSIBLE_RUN_SPEED_MS = 7.5
MAX_SPEED_REFERENCE_S = 300.0
MAX_SPEED_DECAY = 0.07


def plausible_speed_ceiling(duration_s: float) -> float:
    """Vitesse au-dela de laquelle la donnee, pas l'athlete, est en cause."""
    if duration_s <= 0:
        return MAX_PLAUSIBLE_RUN_SPEED_MS
    return MAX_PLAUSIBLE_RUN_SPEED_MS * (duration_s / MAX_SPEED_REFERENCE_S) ** (
        -MAX_SPEED_DECAY
    )


class PerformanceError(ValueError):
    """Courbe de reference inexploitable."""


# ── vitesse maximale moyenne ─────────────────────────────────────────────

def mean_max_speed(
    df: pd.DataFrame,
    durations_s: Sequence[int] = DEFAULT_DURATIONS_S,
    max_gap_s: float = 30.0,
    max_plausible_ms: float | None = None,
) -> dict[int, float]:
    """
    Meilleure vitesse moyenne soutenue, pour chaque duree demandee.

    La serie est reechantillonnee sur une grille reguliere d'une seconde, et
    toute fenetre chevauchant un trou est ecartee : sans cette precaution une
    pause de dix minutes au milieu d'une sortie fabriquerait un record, la
    distance continuant d'avancer pendant que le temps est arrete.

    Renvoie NaN pour les durees depassant le plus long segment continu.
    """
    grid = _uniform_grid(df, max_gap_s)
    if grid is None:
        return {int(d): float("nan") for d in durations_s}
    distance_m, valid = grid

    out: dict[int, float] = {}
    for d in durations_s:
        d = int(d)
        if d <= 0 or len(distance_m) <= d:
            out[d] = float("nan")
            continue
        # Distance parcourue sur chaque fenetre glissante de d secondes.
        delta = distance_m[d:] - distance_m[:-d]
        # Une fenetre n'est retenue que si elle ne contient aucun trou.
        window_ok = _all_true_windows(valid, d)
        delta = np.where(window_ok, delta, -np.inf)
        best = delta.max() if delta.size else -np.inf
        speed = float(best / d) if np.isfinite(best) and best > 0 else float("nan")
        # Une vitesse invraisemblable ne dit rien de l'athlete : elle signale
        # un canal de distance a jeter, pas un record a enregistrer.
        ceiling = max_plausible_ms if max_plausible_ms else plausible_speed_ceiling(d)
        out[d] = speed if np.isfinite(speed) and speed <= ceiling else float("nan")
    return out


def _uniform_grid(df: pd.DataFrame, max_gap_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Distance cumulee sur une grille d'une seconde, plus un masque de validite."""
    if df.empty:
        return None

    dt = dt_seconds(df, max_gap_s).to_numpy(dtype=float)
    elapsed = np.cumsum(dt)

    distance = _cumulative_distance(df, dt)
    if distance is None:
        return None

    total = math.floor(elapsed[-1])
    if total < 2:
        return None

    grid = np.arange(total + 1, dtype=float)
    on_grid = np.interp(grid, elapsed, distance)

    # Un point de grille est valide s'il ne tombe pas dans un trou. Les trous
    # sont les echantillons dont le pas depasse largement la seconde.
    valid = np.ones(total + 1, dtype=bool)
    holes = np.flatnonzero(dt > 1.5)
    for i in holes:
        start = elapsed[i] - dt[i]
        lo = max(0, math.floor(start))
        hi = min(total, math.ceil(elapsed[i]))
        valid[lo : hi + 1] = False
    return on_grid, valid


def _cumulative_distance(df: pd.DataFrame, dt: np.ndarray) -> np.ndarray | None:
    """Distance cumulee en metres, depuis la colonne disponible."""
    for col, scale in (("km", 1000.0), ("distance_m", 1.0), ("distance", 1.0)):
        if col in df.columns:
            d = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) * scale
            d = pd.Series(d).ffill().bfill().to_numpy()
            if np.isfinite(d).all():
                # La distance doit croitre : un capteur qui recule est corrige.
                return np.maximum.accumulate(d)
    for col in ("speed_ms", "speed"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            return np.cumsum(np.clip(v, 0.0, None) * dt)
    return None


def _all_true_windows(valid: np.ndarray, width: int) -> np.ndarray:
    """Pour chaque fenetre de `width`+1 points, vrai si aucun point invalide."""
    bad = np.cumsum(~valid)
    # nombre de points invalides dans [i, i+width]
    inside = bad[width:] - bad[:-width] + (~valid[:-width]).astype(int)
    return inside == 0


# ── courbe de reference ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ReferenceCurve:
    """
    Loi de Riegel ajustee sur les meilleurs efforts de l'athlete :

        v(t) = v_ref * (t / t_ref) ** (-b)

    `b` vaut typiquement 0,06 a 0,08 chez un coureur entraine ; une valeur
    hors de cette plage signale un ajustement douteux.
    """

    v_ref_ms: float
    t_ref_s: float
    exponent: float
    n_points: int
    r_squared: float
    durations_s: tuple[int, ...] = field(default=DEFAULT_DURATIONS_S)

    def speed_at(self, duration_s: float) -> float:
        if duration_s <= 0:
            return float("nan")
        return self.v_ref_ms * (duration_s / self.t_ref_s) ** (-self.exponent)

    def time_for_distance(self, distance_m: float) -> float:
        """Chrono predit sur une distance, par resolution numerique."""
        # La distance couverte croit avec la duree : si la duree testee ne
        # suffit pas, c'est la borne basse qu'il faut relever.
        lo, hi = 60.0, 60_000.0
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            if self.speed_at(mid) * mid < distance_m:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def to_dict(self) -> dict:
        return {
            "v_ref_ms": self.v_ref_ms,
            "t_ref_s": self.t_ref_s,
            "exponent": self.exponent,
            "n_points": self.n_points,
            "r_squared": self.r_squared,
            "durations_s": list(self.durations_s),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ReferenceCurve:
        return cls(
            v_ref_ms=float(d["v_ref_ms"]),
            t_ref_s=float(d["t_ref_s"]),
            exponent=float(d["exponent"]),
            n_points=int(d["n_points"]),
            r_squared=float(d["r_squared"]),
            durations_s=tuple(d.get("durations_s", DEFAULT_DURATIONS_S)),
        )


def fit_reference_curve(
    mmp_history: Iterable[dict[int, float]],
    durations_s: Sequence[int] = DEFAULT_DURATIONS_S,
    t_ref_s: float = 1800.0,
    percentile: float = 98.0,
    min_sessions_per_duration: int = 20,
) -> ReferenceCurve:
    """
    Ajuste la loi de Riegel sur les meilleurs efforts de chaque duree.

    On ne prend pas le maximum absolu mais un percentile eleve : sur des
    donnees de terrain, le record d'une duree est souvent un artefact de
    capteur plutot qu'une performance, et une seule seance aberrante
    deplacerait toute la courbe — donc l'echelle entiere de l'index.

    Une duree n'est retenue que si assez de seances l'alimentent. Chez
    l'athlete principal, l'heure pleine n'apparait que dans quinze seances :
    un percentile calcule sur si peu de valeurs redevient un maximum, et
    l'exposant ajuste sortait negatif, c'est-a-dire une vitesse qui augmente
    avec la duree.

    Cette courbe definit ce que signifie « 1.0 ». Elle est ajustee une fois
    sur tout l'historique puis figee dans athlete_calibration.json : la
    recalculer a chaque execution ferait deriver le sens de la valeur 1.0.
    """
    samples: dict[int, list[float]] = {int(d): [] for d in durations_s}
    for mmp in mmp_history:
        for d, v in mmp.items():
            d = int(d)
            if d not in samples or v is None or not np.isfinite(v) or v <= 0:
                continue
            # Le plafond est reapplique ici : cette fonction peut recevoir des
            # MMP calcules ailleurs, et un percentile sur un petit echantillon
            # redevient sensible a une seule valeur aberrante.
            if v > plausible_speed_ceiling(d):
                continue
            samples[d].append(float(v))

    bests = {
        d: float(np.percentile(vals, percentile))
        for d, vals in samples.items()
        if len(vals) >= min_sessions_per_duration
    }

    if len(bests) < 3:
        raise PerformanceError(
            f"Seulement {len(bests)} duree(s) exploitable(s) (il en faut trois), "
            f"chacune devant compter au moins {min_sessions_per_duration} seances."
        )

    ds = np.array(sorted(bests), dtype=float)
    vs = np.array([bests[int(d)] for d in ds], dtype=float)

    slope, intercept = np.polyfit(np.log(ds), np.log(vs), 1)
    exponent = float(-slope)

    predicted = intercept + slope * np.log(ds)
    residual = np.log(vs) - predicted
    ss_tot = float(((np.log(vs) - np.log(vs).mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / ss_tot if ss_tot > 0 else 1.0

    v_ref = float(math.exp(intercept + slope * math.log(t_ref_s)))
    return ReferenceCurve(v_ref, float(t_ref_s), exponent, len(bests), r2, tuple(durations_s))


# ── observation de performance ────────────────────────────────────────────

@dataclass(frozen=True)
class PerformanceObservation:
    session_id: str
    date: pd.Timestamp
    index: float
    duration_s: int
    speed_ms: float
    quality_flags: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        return not self.quality_flags


def performance_index(
    df: pd.DataFrame,
    reference: ReferenceCurve,
    *,
    session_id: str,
    date: pd.Timestamp,
    hr_max: float | None = None,
    elevation_gain_m: float | None = None,
    distance_km: float | None = None,
    min_duration_s: int = 300,
    min_index: float = 0.90,
    min_hr_frac_of_max: float = 0.90,
    max_elevation_m_per_km: float = 15.0,
    min_hr_coverage: float = 0.90,
    max_gap_s: float = 30.0,
) -> PerformanceObservation | None:
    """
    Transforme une seance en observation, ou renvoie None si l'effort ne
    ressemble pas a une performance.

    Les garde-fous comptent autant que le calcul : alimenter la vraisemblance
    de Banister avec un footing en endurance etiquete « performance » est le
    moyen le plus sur d'obtenir un posterior denue de sens. On exige donc un
    effort proche du maximal, corrobore par la frequence cardiaque, sur un
    terrain qui ne fausse pas la vitesse.
    """
    durations = [d for d in reference.durations_s if d >= min_duration_s]
    mmp = mean_max_speed(df, durations, max_gap_s)

    best_index, best_d, best_v = float("-inf"), None, float("nan")
    for d, v in mmp.items():
        if not np.isfinite(v):
            continue
        ref_v = reference.speed_at(d)
        if not np.isfinite(ref_v) or ref_v <= 0:
            continue
        ratio = v / ref_v
        if ratio > best_index:
            best_index, best_d, best_v = ratio, d, v

    if best_d is None or not np.isfinite(best_index):
        return None
    if best_index < min_index:
        return None

    flags: list[str] = []

    if "hr" in df.columns:
        hr = pd.to_numeric(df["hr"], errors="coerce")
        coverage = float(hr.notna().mean())
        if coverage < min_hr_coverage:
            flags.append("hr_coverage")
        if hr_max and hr.notna().any() and float(hr.max()) < min_hr_frac_of_max * hr_max:
            # Vitesse elevee sans cout cardiaque : descente, vent, ou capteur.
            flags.append("hr_not_maximal")
    else:
        flags.append("no_hr")

    if elevation_gain_m is not None and distance_km and (
        elevation_gain_m / distance_km > max_elevation_m_per_km
    ):
        flags.append("hilly")

    return PerformanceObservation(
        session_id=session_id,
        date=pd.Timestamp(date),
        index=float(best_index),
        duration_s=int(best_d),
        speed_ms=float(best_v),
        quality_flags=tuple(flags),
    )


def target_index(reference: ReferenceCurve, distance_m: float, target_time_s: float) -> float:
    """
    Traduit un objectif (distance + chrono) en index de performance vise.

    C'est ce qui remplace les deux constantes 4920 et 4680 codees en dur : la
    cible se derive de la courbe de l'athlete, donc le systeme accepte
    n'importe quel athlete et n'importe quel objectif.
    """
    if target_time_s <= 0:
        raise PerformanceError("Le chrono vise doit etre strictement positif.")
    v_target = distance_m / target_time_s
    v_ref = reference.speed_at(target_time_s)
    if not np.isfinite(v_ref) or v_ref <= 0:
        raise PerformanceError("Courbe de reference inexploitable a cette duree.")
    return float(v_target / v_ref)
