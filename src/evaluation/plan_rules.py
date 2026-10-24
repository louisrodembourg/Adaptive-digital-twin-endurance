"""
Confronter les regles du planificateur a ce que les athletes ont fait.

Le planificateur impose des bornes — rapport charge aigue sur charge chronique
sous `acwr_upper_bound`, un plancher de retention, au plus
`max_consecutive_training_days` jours d'entrainement de suite. Elles viennent
de la litterature et de l'usage, ce qui est une raison de les ecrire et pas
une mesure. Aucune n'avait jamais ete confrontee a des donnees, alors que le
depot exige une validation hors echantillon de tout le reste.

Protocole
---------
Chaque semaine calendaire d'un athlete porte ses caracteristiques (charge
hebdomadaire, ACWR en fin de semaine, plus longue serie de jours consecutifs)
et son issue : la variation du meilleur effort entre une fenetre de reference
AVANT la semaine et une fenetre d'issue APRES. On compare ensuite, A
L'INTERIEUR de chaque athlete, les semaines qui respectent une borne a celles
qui la violent, puis on agrege par un test du signe sur les athletes — comme
`cohort.py`, et jamais par une moyenne, qui melangerait des historiques tres
inegaux.

Trois pieges, tous rencontres avant d'arriver a cette forme
-----------------------------------------------------------
**L'enveloppe ne peut pas servir d'issue.** C'est un maximum glissant sur
42 jours : elle ne bouge que par paliers, et sa variation MEDIANE sur 28 jours
vaut exactement zero chez les soixante-six athletes. Le test tournait alors sur
des asymetries de bruit.

**La semaine evaluee ne doit toucher ni la reference ni l'issue.** Un maximum
glissant qui regarde 42 jours en arriere englobe la semaine qu'on juge :
l'effort dur de cette semaine EST une observation de performance et entre
mecaniquement dans les deux bornes de la difference. D'ou le tampon de
`BUFFER_DAYS` jours de part et d'autre.

**Un maximum sur plus de tirages est plus grand.** Les semaines a ACWR eleve
sont suivies de plus de sorties — +5,2 observations entre les deux fenetres
dans la tranche au-dela de 2,0 — donc d'un meilleur maximum sans qu'aucune
performance n'ait progresse. Chaque regle est jugee sur DEUX statistiques : le
meilleur effort, et l'effort MEDIAN de la fenetre, insensible au nombre de
tirages. Une regle qui ne tient que sur la premiere ne tient pas.

Et un confond qui ne se controle pas, il s'exclut
-------------------------------------------------
Un ACWR eleve signifie souvent que la charge chronique etait basse, donc que
l'athlete revenait d'un arret : la fenetre de reference est alors elle-meme
deprimee, et la progression mesuree est un retour a la normale. Sans ce
controle, la borne haute d'ACWR ressort VIOLEMMENT nuisible
(-0,333 ecart-type, p < 0,0001) et le depot aurait publie la refutation d'une
regle de securite. En exigeant `MIN_BASELINE_OBS` observations dans la fenetre
de reference — donc un entrainement regulier avant la semaine jugee — l'ecart
tombe a -0,042 et le test ne conclut plus.

Ce que ce protocole ne fait pas
-------------------------------
Il ne reconstruit pas les TYPES de seances : les resumes publics ne les
portent pas, donc les plafonds par type ne sont pas testables ici. Il n'etablit
aucune causalite. Et le plancher de retention du planificateur porte sur 80 %
de la charge d'une semaine de reference, quantite qui n'existe pas dans un
historique observe : `ACWR >= 0,8` en est l'analogue le plus proche, et c'est
une approximation declaree, pas la regle elle-meme.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

# Fenetres. Le tampon isole la semaine jugee ; la portee doit couvrir assez de
# sorties pour qu'un maximum ait un sens sans diluer l'effet sur un trimestre.
BUFFER_DAYS = 7
REACH_DAYS = 42

# Une fenetre de reference pauvre en observations signale une coupure, et c'est
# le confond qui renverse le resultat. Le seuil est la mediane observee sur la
# cohorte : la moitie des semaines reste comparable.
MIN_BASELINE_OBS = 15

# Un athlete qui n'a pas assez de semaines des deux cotes d'une borne ne
# compare rien : il sort, et son absence est comptee.
MIN_WEEKS_PER_SIDE = 6

# Amplitudes injectees pour le plancher de detectabilite, en ecarts-types de la
# puissance de l'athlete. Meme logique que `sensitivity.py`.
DEFAULT_DELTAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)

STATISTICS = ("meilleur", "median")

SUPPORTED = "soutenue"
CONTRADICTED = "contredite"
UNDETERMINED = "indeterminable"


@dataclass(frozen=True)
class RuleVerdict:
    """Ce qu'une regle du planificateur vaut face aux donnees."""

    rule: str
    statistic: str
    n_athletes: int
    mean_gap: float
    median_gap: float
    in_favour: int
    p_value: float
    floor: float | None

    @property
    def verdict(self) -> str:
        """
        Trois issues, jamais deux.

        Confondre « la regle n'est pas soutenue » avec « la regle est
        contredite » est l'erreur que ce depot s'emploie a ne pas commettre :
        un test qui ne conclut pas rend `indeterminable`, et son plancher dit
        quel effet il aurait su voir.
        """
        if self.n_athletes == 0 or not np.isfinite(self.p_value):
            return UNDETERMINED
        if self.p_value >= 0.05:
            return UNDETERMINED
        return SUPPORTED if self.in_favour > self.n_athletes / 2 else CONTRADICTED

    def describe(self) -> str:
        if self.n_athletes == 0:
            return f"{self.rule} : aucun athlete comparable."
        plancher = ("deja detecte sans injection" if self.floor == 0.0
                    else f"plancher {self.floor:.2f} sd" if self.floor is not None
                    else "plancher non atteint")
        return (f"{self.rule} ({self.statistic}) : {self.verdict} — "
                f"{self.mean_gap:+.3f} sd, {self.in_favour}/{self.n_athletes} "
                f"en faveur, p = {self.p_value:.4f}, {plancher}")


def longest_streak(active: np.ndarray) -> int:
    """Plus longue suite de jours d'entrainement consecutifs."""
    best = current = 0
    for day in active:
        current = current + 1 if day else 0
        best = max(best, current)
    return best


def weekly_frame(uuid: str, daily_load: pd.DataFrame, power: pd.DataFrame,
                 load_column: str = "trimp_bike", value_column: str = "pi",
                 buffer_days: int = BUFFER_DAYS,
                 reach_days: int = REACH_DAYS) -> pd.DataFrame:
    """
    Une ligne par semaine calendaire : ce qui a ete fait, et ce qui a suivi.

    La charge est reindexee sur un calendrier journalier complet avant tout
    comptage : un jour sans seance est une charge nulle explicite, sinon
    « trois jours consecutifs » compterait des seances et non des jours.
    """
    if daily_load.empty or load_column not in daily_load.columns:
        return pd.DataFrame()
    if power.empty or value_column not in power.columns:
        return pd.DataFrame()

    daily = daily_load.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").sort_index()
    if daily.empty:
        return pd.DataFrame()
    daily = daily.reindex(
        pd.date_range(daily.index.min(), daily.index.max(), freq="D"),
        fill_value=0.0)
    load = pd.to_numeric(daily[load_column], errors="coerce").fillna(0.0)

    obs = power[["date", value_column]].dropna().copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.sort_values("date")
    if obs.empty:
        return pd.DataFrame()
    spread = float(obs[value_column].std())
    if not np.isfinite(spread) or spread == 0.0:
        return pd.DataFrame()
    days = obs["date"].to_numpy()
    watts = pd.to_numeric(obs[value_column], errors="coerce").to_numpy(dtype=float)

    def window(start, stop) -> tuple[float, float, int]:
        """Meilleur effort, effort median, et nombre d'observations."""
        mask = (days >= np.datetime64(start)) & (days <= np.datetime64(stop))
        if not mask.any():
            return np.nan, np.nan, 0
        return float(watts[mask].max()), float(np.median(watts[mask])), int(mask.sum())

    acute = load.rolling(7).mean()
    chronic = load.rolling(28).mean()
    rows = []
    for week_end, block in load.groupby(pd.Grouper(freq="W-SUN")):
        if len(block) < 7:
            continue
        chron, acu = chronic.get(week_end, np.nan), acute.get(week_end, np.nan)
        if not np.isfinite(chron) or chron <= 0.0 or not np.isfinite(acu):
            continue
        before_max, before_med, n_before = window(
            week_end - pd.Timedelta(days=reach_days),
            week_end - pd.Timedelta(days=buffer_days))
        after_max, after_med, n_after = window(
            week_end + pd.Timedelta(days=buffer_days),
            week_end + pd.Timedelta(days=reach_days))
        if not np.isfinite(before_max) or not np.isfinite(after_max):
            continue
        rows.append({
            "uuid": uuid,
            "week_end": week_end,
            "load": float(block.sum()),
            "acwr": float(acu / chron),
            "consecutive_days": longest_streak((block > 0).to_numpy()),
            "active_days": int((block > 0).sum()),
            "meilleur": (after_max - before_max) / spread,
            "median": (after_med - before_med) / spread,
            "n_before": n_before,
            "n_after": n_after,
        })
    return pd.DataFrame(rows)


def sign_test(values: pd.Series) -> tuple[int, int, float]:
    """Test du signe bilateral exact. Rend (en faveur, total, p)."""
    clean = pd.Series(values).dropna()
    n = len(clean)
    if n == 0:
        return 0, 0, float("nan")
    favour = int((clean > 0).sum())
    k = min(favour, n - favour)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return favour, n, p


def _per_athlete_gaps(weeks: pd.DataFrame, respects: pd.Series, statistic: str,
                       min_weeks: int = MIN_WEEKS_PER_SIDE) -> pd.Series:
    """Ecart d'issue respect / violation, un nombre par athlete."""
    gaps = []
    for _, group in weeks.groupby("uuid"):
        mask = respects.loc[group.index]
        kept, broken = group[mask], group[~mask]
        if len(kept) < min_weeks or len(broken) < min_weeks:
            continue
        gaps.append(float(kept[statistic].mean() - broken[statistic].mean()))
    return pd.Series(gaps, dtype=float)


def rule_floor(weeks: pd.DataFrame, respects: pd.Series, statistic: str,
               deltas: tuple[float, ...] = DEFAULT_DELTAS,
               min_weeks: int = MIN_WEEKS_PER_SIDE) -> float | None:
    """
    Plus petit effet que ce protocole distinguerait du hasard, ou None.

    L'effet est ajoute a l'issue des semaines qui RESPECTENT la regle, et le
    test est repasse inchange. Ce plancher est optimiste : l'effet injecte est
    constant et de meme signe chez tous les athletes, ce qu'un vrai effet n'est
    jamais.
    """
    base = _per_athlete_gaps(weeks, respects, statistic, min_weeks)
    if base.empty:
        return None
    for delta in sorted(deltas):
        favour, n, p = sign_test(base + delta)
        if n and p < 0.05 and favour > n / 2:
            return float(delta)
    return None


def evaluate_rule(weeks: pd.DataFrame, respects: pd.Series, name: str,
                  statistic: str = "median",
                  min_weeks: int = MIN_WEEKS_PER_SIDE) -> RuleVerdict:
    """Verdict d'une regle sur une statistique d'issue."""
    gaps = _per_athlete_gaps(weeks, respects, statistic, min_weeks)
    favour, n, p = sign_test(gaps)
    return RuleVerdict(
        rule=name, statistic=statistic, n_athletes=n,
        mean_gap=float(gaps.mean()) if n else float("nan"),
        median_gap=float(gaps.median()) if n else float("nan"),
        in_favour=favour, p_value=p,
        floor=rule_floor(weeks, respects, statistic, min_weeks=min_weeks))


def rule_masks(weeks: pd.DataFrame, acwr_upper: float,
               acwr_lower: float, max_consecutive: int) -> dict[str, pd.Series]:
    """
    Les regles testables, nommees comme le planificateur les nomme.

    Les seuils arrivent en parametres et ne sont jamais ecrits ici : un seuil
    recopie derive le jour ou la configuration change, et la mesure se met
    alors a valider une regle que le planificateur n'applique plus.
    """
    return {
        f"ACWR <= {acwr_upper:g} (borne haute)": weeks["acwr"] <= acwr_upper,
        f"ACWR >= {acwr_lower:g} (plancher de retention)": weeks["acwr"] >= acwr_lower,
        f"au plus {max_consecutive} jours consecutifs":
            weeks["consecutive_days"] <= max_consecutive,
    }


def evaluate_plan_rules(weeks: pd.DataFrame, acwr_upper: float = 1.5,
                        acwr_lower: float = 0.8, max_consecutive: int = 3,
                        min_baseline_obs: int = MIN_BASELINE_OBS) -> pd.DataFrame:
    """
    Chaque regle, sur les deux statistiques, avec et sans le controle de reprise.

    Les quatre lignes par regle sont publiees ensemble parce que c'est leur
    ecart qui porte l'information : une regle qui ne tient que sur le meilleur
    effort tient sur un artefact d'echantillonnage, et une regle qui ne tient
    que sans le controle de reprise tient sur des retours de coupure.
    """
    if weeks.empty:
        return pd.DataFrame()
    rows = []
    regimes = (("toutes les semaines", weeks),
               (f"reference >= {min_baseline_obs} obs",
                weeks[weeks["n_before"] >= min_baseline_obs]))
    for regime, subset in regimes:
        if subset.empty:
            continue
        masks = rule_masks(subset, acwr_upper, acwr_lower, max_consecutive)
        for name, mask in masks.items():
            for statistic in STATISTICS:
                v = evaluate_rule(subset, mask, name, statistic)
                rows.append({
                    "regle": name, "regime": regime, "statistique": statistic,
                    "athletes": v.n_athletes, "ecart_moyen": v.mean_gap,
                    "ecart_median": v.median_gap, "en_faveur": v.in_favour,
                    "p": v.p_value, "plancher": v.floor, "verdict": v.verdict,
                })
    return pd.DataFrame(rows)


def format_plan_rules(table: pd.DataFrame) -> str:
    """Une ligne par regle, regime et statistique."""
    if table.empty:
        return "Aucune semaine exploitable."
    header = (f"{'regle':38s} {'regime':24s} {'stat':9s} {'ath':>4s} "
              f"{'ecart':>7s} {'faveur':>7s} {'p':>7s} {'planch':>7s} verdict")
    lines = [header, "-" * len(header)]
    for _, r in table.iterrows():
        plancher = ("-" if r["plancher"] is None or pd.isna(r["plancher"])
                    else f"{r['plancher']:.2f}")
        lines.append(
            f"{r['regle'][:38]:38s} {r['regime'][:24]:24s} "
            f"{r['statistique']:9s} {int(r['athletes']):4d} "
            f"{r['ecart_moyen']:+7.3f} "
            f"{int(r['en_faveur'])}/{int(r['athletes']):<5d} "
            f"{r['p']:7.4f} {plancher:>7s} {r['verdict']}")
    return "\n".join(lines)
