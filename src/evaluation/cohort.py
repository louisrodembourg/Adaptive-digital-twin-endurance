"""
Replication sur une cohorte : un resultat sur un athlete est une anecdote.

Ce depot a mesure, chez son athlete de reference, le premier effet de charge
qui survive a son intervalle : apparier la discipline de la charge a celle de
la cible vaut +0,631 de competence (P = 1,000), et la charge velo bat
« predire la moyenne » de +0,339 avec un intervalle [+0,055 · +0,524]. Mais
38 plis chez une personne ne distinguent pas un fait d'une particularite.

Ce module rejoue l'experience sur les athletes du cache OpenData et rend la
DISTRIBUTION des resultats, pas leur moyenne : combien d'athletes voient
l'effet, chez combien l'intervalle exclut zero, et dans quel sens se trompent
les autres. Un effet reel doit se retrouver chez une majorite nette ; un
artefact se repartit autour de zero.

Pourquoi les resumes suffisent
------------------------------
Tout ce dont l'experience a besoin est dans le `<uuid>.json` de chaque
archive : la date, la discipline, la puissance critique de vingt minutes et un
TRIMP. Decoder les CSV de 62 athletes couterait des heures pour la meme
reponse. La contrepartie est declaree et verifiee : le TRIMP employe ici est
celui de GoldenCheetah (`trimp_points`), pas celui que `features.py` calcule.
Chez l'athlete de reference les deux correlent a 0,982 (0,984 sur le velo), le
TRIMP du depot valant 0,938 fois celui de GoldenCheetah en mediane. Pour
savoir si une relation charge-performance se replique, un substitut correle a
0,98 convient ; pour publier un coefficient, non.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..ingestion.opendata import AthleteSummary, read_summary
from .foundation import build_forecast_context
from .metrics import paired_block_bootstrap
from .report import BIKE_SPORTS, evaluate_results, specific_variant
from .validation import horizon_splits

log = logging.getLogger(__name__)

#: Metriques de charge du resume, par ordre de preference. `trimp_points` est
#: l'equivalent le plus direct de ce que calcule le depot ; le zonal couvre
#: quelques seances de plus et sert de repli.
TRIMP_KEYS: tuple[str, ...] = ("trimp_points", "trimp_zonal_points")

#: La puissance critique de vingt minutes, telle que GoldenCheetah la nomme.
POWER_KEY = "20m_critical_power"

#: Comparaisons rejouees chez chaque athlete. La premiere est le resultat a
#: repliquer, la deuxieme est le controle de validite discriminante — la
#: charge de COURSE ne doit pas predire une puissance en velo.
PAIRS: tuple[tuple[str, str], ...] = (
    ("velo seul", "moyenne"),
    ("velo seul", "course seule"),
    ("velo seul", "persistance"),
)


@dataclass(frozen=True)
class CohortCriteria:
    """
    Ce qu'on exige d'un athlete pour qu'il porte l'experience.

    Les seuils decoulent du protocole, pas d'un gout : il faut `min_train_days`
    de contexte plus plusieurs plis d'horizon pour que la validation dise
    quelque chose, et une enveloppe glissante a besoin d'observations assez
    denses pour ne pas retomber a chaque trou.
    """

    min_power_observations: int = 60
    min_span_days: int = 540
    min_rides_with_load: int = 150
    min_folds: int = 8


@dataclass(frozen=True)
class AthleteFrames:
    """Les deux tableaux dont l'experience a besoin, et rien de plus."""

    uuid: str
    sessions: pd.DataFrame
    power: pd.DataFrame
    trimp_key: str

    @property
    def span_days(self) -> int:
        if self.sessions.empty:
            return 0
        dates = pd.to_datetime(self.sessions["date"])
        return int((dates.max() - dates.min()).days)


def summary_to_frames(summary: AthleteSummary) -> AthleteFrames | None:
    """
    Transforme un resume en seances et en observations de puissance.

    Les seances portent les colonnes que `build_daily_load` lit reellement :
    la ventilation par discipline vient donc de la meme fonction que pour
    l'athlete de reference, sans second chemin a maintenir.
    """
    trimp_key = next(
        (k for k in TRIMP_KEYS
         if any(r.metric(k) is not None for r in summary.rides)),
        None,
    )
    if trimp_key is None:
        return None

    rows, power = [], []
    for i, ride in enumerate(summary.rides):
        if ride.date is None:
            continue
        date = pd.Timestamp(ride.date).normalize()
        load = ride.metric(trimp_key)
        rows.append({
            "session_id": f"{summary.uuid}:{i}",
            "date": date,
            "sport": ride.sport or "Unknown",
            "trimp": float(load) if load is not None else 0.0,
            "source": "opendata_summary",
        })
        if str(ride.sport or "").lower().startswith(BIKE_SPORTS):
            watts = ride.metric(POWER_KEY)
            # Une puissance nulle ou absurde signale une seance sans capteur,
            # pas un athlete faible : GoldenCheetah ecrit 0 dans ce cas.
            if watts is not None and 50.0 < watts < 600.0:
                power.append({"date": date, "pi": float(watts)})

    if not rows:
        return None
    return AthleteFrames(summary.uuid, pd.DataFrame(rows),
                         pd.DataFrame(power, columns=["date", "pi"]), trimp_key)


def screen_cache(cache_dir: Path | str,
                 criteria: CohortCriteria | None = None) -> list[AthleteFrames]:
    """
    Retient les athletes du cache qui peuvent porter l'experience.

    Un athlete ecarte est trace, pas efface : savoir combien d'athletes et
    pourquoi est la seule facon de lire ensuite la distribution des
    resultats sans se tromper sur la population.
    """
    criteria = criteria or CohortCriteria()
    kept: list[AthleteFrames] = []
    for path in sorted(Path(cache_dir).glob("*.zip")):
        try:
            frames = summary_to_frames(read_summary(path))
        except Exception as exc:                    # archive illisible
            log.info("%s ecarte : %s", path.name, exc)
            continue
        if frames is None:
            log.info("%s ecarte : aucune metrique de charge", path.name)
            continue
        if len(frames.power) < criteria.min_power_observations:
            log.info("%s ecarte : %d observations de puissance",
                     path.name, len(frames.power))
            continue
        if frames.span_days < criteria.min_span_days:
            log.info("%s ecarte : %d jours d'historique", path.name, frames.span_days)
            continue
        with_load = int((frames.sessions["trimp"] > 0).sum())
        if with_load < criteria.min_rides_with_load:
            log.info("%s ecarte : %d seances avec charge", path.name, with_load)
            continue
        kept.append(frames)
    return kept


#: Nom court de `PAIRS` vers le canal de charge correspondant.
_SHORT_TO_COLUMN = {"velo seul": "trimp_bike", "course seule": "trimp_run"}


def _variant(short: str, tau1: float, tau2: float) -> str:
    """Traduit un nom court de `PAIRS` en nom de variante enregistree."""
    column = _SHORT_TO_COLUMN.get(short)
    if column is None:
        return short                      # une reference, pas une variante
    return specific_variant(column, tau1, tau2)


def evaluate_athlete(
    frames: AthleteFrames,
    tau1: float = 42.0,
    tau2: float = 12.0,
    horizon_days: int = 28,
    min_train_days: int = 180,
    window_days: int = 42,
    min_periods: int = 3,
    criteria: CohortCriteria | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Rejoue l'experience chez un athlete et rend ses comparaisons appariees.

    Le protocole est celui de la cible velo de `cli evaluate`, sans un
    parametre de plus : si les deux divergeaient, la replication ne
    repliquerait rien.
    """
    from ..pipeline.orchestrator import build_daily_load

    criteria = criteria or CohortCriteria()
    daily = build_daily_load(frames.sessions)
    context = build_forecast_context(frames.power, daily, "pi",
                                     window_days=window_days,
                                     min_periods=min_periods)
    dense = (context.frame[[context.target_column]].dropna()
             .rename(columns={context.target_column: "perf"}).reset_index())
    splits = list(horizon_splits(dense["date"], horizon_days=horizon_days,
                                 min_train_days=min_train_days,
                                 step_days=horizon_days))
    if len(splits) < criteria.min_folds:
        return pd.DataFrame()

    results = evaluate_results(dense, daily, target="perf",
                               taus=[(tau1, tau2)], splits=splits)
    rows = []
    for short_a, short_b in PAIRS:
        a, b = _variant(short_a, tau1, tau2), _variant(short_b, tau1, tau2)
        ra, rb = results.get(a), results.get(b)
        if ra is None or rb is None or ra.y_true.size == 0:
            continue
        if ra.y_true.shape != rb.y_true.shape:
            continue
        stats = paired_block_bootstrap(ra.y_true, ra.y_pred, rb.y_pred,
                                       ra.blocks, seed=seed)
        rows.append({
            "uuid": frames.uuid,
            "variante": short_a,
            "reference": short_b,
            "r2_variante": ra.r2_out_of_sample,
            "r2_reference": rb.r2_out_of_sample,
            "n_jours": len(dense),
            "n_puissances": len(frames.power),
            "plis": ra.n_folds,
            **stats,
        })
    return pd.DataFrame(rows)


def run_cohort(cache_dir: Path | str, criteria: CohortCriteria | None = None,
               **kwargs) -> pd.DataFrame:
    """Crible le cache puis rejoue l'experience chez chaque athlete retenu."""
    criteria = criteria or CohortCriteria()
    tables = []
    for frames in screen_cache(cache_dir, criteria):
        try:
            table = evaluate_athlete(frames, criteria=criteria, **kwargs)
        except Exception as exc:
            # Un athlete qui echoue est trace et saute : ne pas laisser une
            # archive bancale emporter toute la cohorte.
            log.warning("%s : %s", frames.uuid[:8], exc)
            continue
        if not table.empty:
            tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def summarise_cohort(table: pd.DataFrame) -> pd.DataFrame:
    """
    Resume la DISTRIBUTION, pas la moyenne.

    Une moyenne de competences sur des athletes d'historiques tres inegaux
    cacherait justement ce qu'on cherche. Ce qui compte est combien
    d'athletes voient l'effet, chez combien l'intervalle exclut zero, et dans
    quel sens se trompent les autres : un effet reel se retrouve chez une
    majorite nette, un artefact se repartit autour de zero.

    `p_signe` est un test binomial bilateral sur le nombre d'athletes au gain
    positif, sous l'hypothese nulle d'un effet absent (une chance sur deux).
    """
    if table.empty:
        return pd.DataFrame()
    from scipy.stats import binomtest

    rows = []
    for (variante, reference), g in table.groupby(["variante", "reference"], sort=False):
        positive = int((g["skill"] > 0).sum())
        rows.append({
            "variante": variante,
            "reference": reference,
            "athletes": len(g),
            "gain_median": float(g["skill"].median()),
            "gain_q1": float(g["skill"].quantile(0.25)),
            "gain_q3": float(g["skill"].quantile(0.75)),
            "gain_positif": positive,
            "ic_exclut_zero_pour": int((g["ci_low"] > 0).sum()),
            "ic_exclut_zero_contre": int((g["ci_high"] < 0).sum()),
            "p_signe": float(binomtest(positive, len(g), 0.5).pvalue),
        })
    return pd.DataFrame(rows)


def format_cohort(table: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Rend la cohorte lisible : le resume d'abord, les athletes ensuite."""
    if table.empty:
        return "Aucun athlete n'a pu porter l'experience."
    lines = [
        f"{'variante':12s} {'contre':12s} {'athl.':>6s} {'gain med.':>10s} "
        f"{'Q1 · Q3':>20s} {'gain > 0':>9s} {'IC > 0':>7s} {'IC < 0':>7s} {'p signe':>8s}",
        "-" * 100,
    ]
    for _, r in summary.iterrows():
        quartiles = f"[{r['gain_q1']:+.3f} · {r['gain_q3']:+.3f}]"
        lines.append(
            f"{r['variante']:12s} {r['reference']:12s} {int(r['athletes']):6d} "
            f"{r['gain_median']:+10.4f} {quartiles:>20s} "
            f"{int(r['gain_positif']):4d}/{int(r['athletes']):<4d} "
            f"{int(r['ic_exclut_zero_pour']):7d} {int(r['ic_exclut_zero_contre']):7d} "
            f"{r['p_signe']:8.4f}"
        )
    lines += ["", "Par athlete, contre « predire la moyenne » :", ""]
    detail = table[table["reference"] == "moyenne"].sort_values("skill", ascending=False)
    lines.append(f"  {'athlete':10s} {'jours':>6s} {'puiss.':>7s} {'plis':>5s} "
                 f"{'R2 charge':>10s} {'R2 moy.':>9s} {'gain':>8s} {'IC 95 %':>20s}")
    for _, r in detail.iterrows():
        interval = f"[{r['ci_low']:+.3f} · {r['ci_high']:+.3f}]"
        flag = " *" if r["ci_low"] > 0 else ""
        lines.append(
            f"  {r['uuid'][:8]:10s} {int(r['n_jours']):6d} {int(r['n_puissances']):7d} "
            f"{int(r['plis']):5d} {r['r2_variante']:+10.4f} {r['r2_reference']:+9.4f} "
            f"{r['skill']:+8.4f} {interval:>20s}{flag}"
        )
    lines += ["", "  * l'intervalle exclut zero chez cet athlete"]
    return "\n".join(lines)


def detection_contrast(table: pd.DataFrame, reference: str = "moyenne") -> pd.DataFrame:
    """
    Distingue une heterogeneite reelle d'un simple defaut de mesure.

    Quand une minorite d'athletes montre un effet et le reste rien, deux
    lectures s'opposent et elles ne mènent pas au meme modele :

    - l'effet est universel mais noye chez les athletes mal mesures. Dans ce
      cas leurs ESTIMATIONS PONCTUELLES restent positives, seulement avec des
      intervalles trop larges pour conclure ;
    - l'effet n'existe que chez certains. Alors les estimations des autres se
      repartissent autour de zero, independamment de leur precision.

    La colonne `gain_median` tranche, et la largeur d'intervalle dit a quel
    point la premiere lecture etait plausible. Mesure sur le cache : +0,331
    chez les detectes contre -0,054 chez les autres, dont 2 positifs sur 9 —
    c'est la seconde lecture.
    """
    if table.empty:
        return pd.DataFrame()
    subset = table[table["reference"] == reference].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["largeur_ic"] = subset["ci_high"] - subset["ci_low"]
    subset["detecte"] = subset["ci_low"] > 0

    rows = []
    for detected, g in subset.groupby("detecte"):
        rows.append({
            "groupe": "detectes" if detected else "non detectes",
            "athletes": len(g),
            "gain_median": float(g["skill"].median()),
            "gain_positif": int((g["skill"] > 0).sum()),
            "largeur_ic_median": float(g["largeur_ic"].median()),
            "puissances_median": float(g["n_puissances"].median()),
            "plis_median": float(g["plis"].median()),
        })
    return pd.DataFrame(rows)


def format_contrast(contrast: pd.DataFrame) -> str:
    """Rend le contraste lisible, avec sa conclusion explicite."""
    if contrast.empty:
        return "Pas de contraste calculable."
    lines = [
        f"{'groupe':14s} {'athl.':>6s} {'gain med.':>10s} {'gain > 0':>9s} "
        f"{'largeur IC':>11s} {'puiss. med.':>12s} {'plis med.':>10s}",
        "-" * 78,
    ]
    for _, r in contrast.iterrows():
        lines.append(
            f"{r['groupe']:14s} {int(r['athletes']):6d} {r['gain_median']:+10.4f} "
            f"{int(r['gain_positif']):4d}/{int(r['athletes']):<4d} "
            f"{r['largeur_ic_median']:11.3f} {r['puissances_median']:12.0f} "
            f"{r['plis_median']:10.0f}"
        )
    non = contrast[contrast["groupe"] == "non detectes"]
    if not non.empty:
        median = float(non.iloc[0]["gain_median"])
        lines += ["", (
            "  Les non detectes sont centres sur "
            f"{median:+.3f} : leurs estimations ponctuelles ne sont pas"
            "\n  positives-mais-noyees, elles sont nulles. L'heterogeneite entre"
            "\n  athletes est donc reelle, et non un defaut de mesure."
        )]
    return "\n".join(lines)
