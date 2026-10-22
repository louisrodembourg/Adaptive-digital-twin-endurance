"""
Layer C — Planner : Optimizer OR-Tools (CP-SAT)

Génère le plan d'entraînement hebdomadaire optimal en respectant
les contraintes dures (Safety Valve) et douces (LLM Context Parser).

Approche Neuro-Symbolique :
  - Symbolique (déterministe) : OR-Tools CP-SAT résout le problème
    d'optimisation sous contraintes → résultats reproductibles et fiables
  - Neuro (LLM) : Claude traduit le contexte naturel en contraintes mathématiques
    → flexibilité et compréhension du langage humain

Objectif : la charge MINIMALE qui délivre le stimulus prescrit, sous
           contraintes de sécurité et de progression.

Pourquoi minimiser et non maximiser
-----------------------------------
L'objectif historique maximisait la spécificité sous un plafond de volume, ce
qui revient à prescrire « entraîne-toi autant que la soupape l'autorise ».
C'est défendable si l'on connaît la dose-réponse de l'athlète. La mesure dit
qu'on ne la connaît pas : sur les treize athlètes du cache, la charge explique
la performance chez quatre et chez neuf non, et aucune variable disponible à
l'inscription ne les sépare (voir `dose_response.py`).

Sous une dose-réponse inconnue, maximiser un proxy est le mauvais côté de
l'incertitude : le coût d'une charge inutile est réel — blessure, surmenage,
temps — alors que son bénéfice est, chez deux athlètes sur trois, non
mesurable. On cherche donc la charge la plus faible qui délivre le stimulus
demandé, et non la plus forte que les contraintes tolèrent.

Variables de décision : session_type[jour] pour 7 jours (lundi → dimanche)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# OR-Tools import conditionnel
try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    warnings.warn(
        "OR-Tools non installé. L'optimizer fonctionnera en mode greedy.\n"
        "pip install ortools",
        ImportWarning, stacklevel=2
    )

from ..config import load_params as _shared_load_params
from .dose_response import DoseResponse
from .llm_context_parser import SoftConstraints
from .safety_valve import (
    SESSION_TYPES,
    HardConstraint,
    get_allowed_sessions,
    get_forced_session,
    get_volume_multiplier,
)


def _load_params(config_path: str | Path | None = None) -> dict:
    """Parametres bruts — delegue au chargeur unique et mis en cache."""
    return _shared_load_params(config_path)


# ═══════════════════════════════════════════════════════════════
#  DATACLASS — Plan Hebdomadaire
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StimulusTarget:
    """
    Le stimulus a delivrer, que le plan doit atteindre au moindre cout.

    C'est ce qui remplace l'objectif de maximisation : on ne demande plus
    « le plus de specificite possible » mais « ces seances-la, pour le moins
    de charge totale possible ».

    `min_retention_pct` empeche l'optimiseur de prescrire le desentrainement.
    Minimiser la charge sans plancher donnerait deux seances cles et cinq
    jours de repos, ce qui est un minimum au sens litteral et une regression
    au sens physiologique. Le plancher se lit dans `params.yaml` plutot que
    d'etre code ici.
    """

    min_key_sessions: int = 2
    key_specificity: float = 0.7
    min_total_trimp: float | None = None
    max_consecutive_training_days: int = 3


@dataclass
class DayPlan:
    """Séance planifiée pour un jour donné."""
    date: date
    day_name: str           # "Lundi", "Mardi", etc.
    session_type: str       # ex: "ENDURANCE"
    trimp_estimated: float
    description: str        # description humaine
    rationale: str          # explication XAI (pourquoi ce choix)
    is_blocked: bool = False
    block_reason: str = ""


@dataclass
class WeeklyPlan:
    """Plan complet pour une semaine."""
    week_start: date
    days: list[DayPlan] = field(default_factory=list)
    total_trimp: float = 0.0
    total_sessions: int = 0
    specificity_score: float = 0.0
    hard_constraints_applied: list[str] = field(default_factory=list)
    soft_constraints_applied: str = ""
    objective: str = "min_load"
    dose_response: DoseResponse | None = None
    #: Ce que le solveur a du abandonner pour rendre un plan. Vide est la
    #: situation normale ; une entree ici doit apparaitre dans le rapport,
    #: sans quoi un plan degrade se lit comme un plan optimal.
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Plan semaine du {self.week_start.strftime('%d/%m/%Y')}",
            f"TRIMP total : {self.total_trimp:.0f} | "
            f"Séances : {self.total_sessions} | "
            f"Score spécificité : {self.specificity_score:.2f}",
        ]
        if self.hard_constraints_applied:
            lines.append(f"Contraintes dures : {', '.join(self.hard_constraints_applied)}")
        for day in self.days:
            icon = "🔴" if day.is_blocked else ("💤" if day.session_type == "REST" else "✅")
            lines.append(
                f"  {icon} {day.day_name:9s} : {day.session_type:12s} "
                f"(TRIMP≈{day.trimp_estimated:.0f}) — {day.description}"
            )
        return "\n".join(lines)

    def xai_report(self) -> str:
        """
        Rapport d'explication (XAI) des choix du planner.

        Commence par dire ce que ce plan optimise — et surtout ce qu'il
        n'optimise pas. Un plan qui ne l'annonce pas laisse croire qu'il
        maximise une performance prédite, alors qu'aucun effet de charge n'est
        démontré chez la plupart des athlètes.
        """
        lines = ["=== Explication du Plan (XAI) ===", "", self.objective_statement(), ""]
        for day in self.days:
            if day.rationale:
                lines.append(f"{day.day_name} [{day.session_type}] : {day.rationale}")
        if self.soft_constraints_applied:
            lines.append(f"\nContexte LLM appliqué : {self.soft_constraints_applied}")
        if self.notes:
            lines.append("\nCe que le plan n'a pas pu appliquer :")
            lines += [f"  — {note}" for note in self.notes]
        return "\n".join(lines)

    def objective_statement(self) -> str:
        """Ce que ce plan cherche, en une phrase honnête."""
        if self.objective == "min_load":
            head = ("Ce plan cherche la charge la PLUS FAIBLE qui délivre le "
                    "stimulus demandé,\nsous les contraintes de sécurité.")
        else:
            head = ("Ce plan MAXIMISE la spécificité sous un plafond de volume "
                    "(comportement\nhistorique).")
        if self.dose_response is None:
            tail = ("Aucune mesure de dose-réponse n'a été fournie : il "
                    "n'optimise donc aucune\nperformance prédite.")
        elif self.dose_response.may_claim_performance_effect:
            tail = (f"{self.dose_response.describe()}.\nUn effet de charge est "
                    "donc démontré chez cet athlète — ce plan ne le maximise "
                    "pas\npour autant, il délivre un stimulus au moindre coût.")
        else:
            tail = (f"{self.dose_response.describe()}.\nCe plan n'optimise "
                    "aucune performance prédite, et ne peut pas en promettre.")
        return f"{head}\n{tail}"


# ═══════════════════════════════════════════════════════════════
#  PARAMÈTRES SÉANCES
# ═══════════════════════════════════════════════════════════════

# TRIMP de référence par type de séance (base = athlète 70 TRIMP/semaine moyen)
SESSION_TRIMP_BASE = {
    "REST":        0,
    "RECOVERY":    18,
    "ENDURANCE":   55,
    "THRESHOLD":   75,
    "VMA":         85,
    "SPECIFIC":    65,
    "STRENGTH":    35,
    "CALIBRATION": 50,
}

SESSION_DESCRIPTIONS = {
    "REST":        "Repos complet ou marche légère",
    "RECOVERY":    "Footing très léger 30-40min Z1 (< 65% FCmax)",
    "ENDURANCE":   "Sortie longue 60-90min Z2 (65-75% FCmax)",
    "THRESHOLD":   "Intervalles au seuil 2×20min Z4 (80-88% FCmax)",
    "VMA":         "VMA court 8×400m ou 6×800m Z5 (> 90% FCmax)",
    "SPECIFIC":    "Allure spécifique 3'40\"/km — 5-8km",
    "STRENGTH":    "Renforcement musculaire : PPG, gainage, plyométrie",
    "CALIBRATION": "Séance test 5km à 4'00\"/km — mesure EF_spec",
}

DAY_NAMES = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


# ═══════════════════════════════════════════════════════════════
#  OPTIMISEUR OR-TOOLS CP-SAT
# ═══════════════════════════════════════════════════════════════

def optimize_week(
    hard_constraints: list[HardConstraint],
    soft_constraints: SoftConstraints | None = None,
    previous_week_trimp: float = 250.0,
    week_start: date | None = None,
    params: dict | None = None,
    race_date: date | None = None,
    objective: str = "min_load",
    stimulus: StimulusTarget | None = None,
    daily_load: pd.DataFrame | None = None,
    dose_response: DoseResponse | None = None,
) -> WeeklyPlan:
    """
    Génère le plan de la semaine via OR-Tools CP-SAT.

    Parameters
    ----------
    hard_constraints : contraintes dures (Safety Valve)
    soft_constraints : contraintes douces (LLM Context Parser)
    previous_week_trimp : TRIMP total de la semaine précédente
    week_start : lundi de la semaine (None = lundi prochain)
    params : configuration
    race_date : date de la compétition (pour taper)
    objective : "min_load" (défaut) ou "max_specificity" (comportement
        historique, conservé pour que les deux soient comparables)
    stimulus : le stimulus à délivrer au moindre coût
    daily_load : historique de charge, pour borner le rapport aigu/chronique.
        Sans lui cette borne — la seule règle de sécurité du dépôt qui soit
        réellement déclenchable — reste inactive, et le plan le dit.
    dose_response : verdict de mesure chez cet athlète. N'entre pas dans
        l'optimisation : il n'y a rien à optimiser tant qu'aucun effet n'est
        démontré. Il est rapporté, pour que le plan ne laisse pas croire
        qu'il maximise une performance prédite.

    Returns
    -------
    WeeklyPlan avec les 7 jours planifiés
    """
    if objective not in ("min_load", "max_specificity"):
        raise ValueError(
            f"objectif inconnu : {objective!r} "
            "(attendus : 'min_load', 'max_specificity')"
        )
    if stimulus is None:
        stimulus = StimulusTarget()
    if params is None:
        params = _load_params()
    if soft_constraints is None:
        soft_constraints = SoftConstraints()
    if week_start is None:
        today = date.today()
        week_start = today - timedelta(days=today.weekday())  # lundi courant

    planner = params.get('planner', {})
    max_progression = 1 + planner.get('max_weekly_progression_pct', 10) / 100  # 1.10
    retention = planner.get('min_weekly_retention_pct', 80) / 100
    acwr_upper = planner.get('acwr_upper_bound', 1.5)

    if stimulus.min_total_trimp is None:
        # Plancher pour ne pas prescrire le desentrainement : minimiser la
        # charge sans borne basse donnerait deux seances cles et cinq jours de
        # repos, ce qui est un minimum litteral et une regression reelle.
        stimulus = replace(stimulus, min_total_trimp=previous_week_trimp * retention)

    # Volume max de la semaine
    volume_multiplier = get_volume_multiplier(hard_constraints)
    if soft_constraints:
        volume_multiplier = min(volume_multiplier, soft_constraints.max_volume_pct)

    max_trimp_week = previous_week_trimp * max_progression * volume_multiplier

    # Sessions autorisées
    allowed = get_allowed_sessions(hard_constraints)
    if soft_constraints and soft_constraints.no_session_types:
        allowed -= set(soft_constraints.no_session_types)

    # Forcer une séance si contrainte dure l'impose
    forced_type = get_forced_session(hard_constraints)

    chronic = _chronic_daily_load(daily_load)
    if HAS_ORTOOLS:
        return _optimize_ortools(
            allowed=allowed,
            forced_type=forced_type,
            max_trimp_week=max_trimp_week,
            soft_constraints=soft_constraints,
            hard_constraints=hard_constraints,
            week_start=week_start,
            params=params,
            race_date=race_date,
            objective=objective,
            stimulus=stimulus,
            chronic_daily_load=chronic,
            acwr_upper=acwr_upper,
            dose_response=dose_response,
        )
    else:
        return _optimize_greedy(
            allowed=allowed,
            forced_type=forced_type,
            max_trimp_week=max_trimp_week,
            soft_constraints=soft_constraints,
            hard_constraints=hard_constraints,
            week_start=week_start,
        )


def _chronic_daily_load(daily_load: pd.DataFrame | None,
                        chronic_days: int = 28) -> float | None:
    """
    Charge quotidienne moyenne des 28 derniers jours, ou None.

    C'est le denominateur du rapport aigu/chronique. On le calcule ici plutot
    que d'appeler `acute_chronic_ratio` : celle-ci rend une serie du PASSE,
    alors qu'il faut le denominateur pour borner une semaine A VENIR, qui
    n'existe pas encore.
    """
    if daily_load is None or len(daily_load) == 0:
        return None
    if "trimp" not in daily_load.columns or "date" not in daily_load.columns:
        return None
    recent = daily_load.sort_values("date").tail(chronic_days)
    if len(recent) < chronic_days:
        # Une moyenne sur quinze jours appelee « chronique » serait un abus :
        # mieux vaut ne pas poser la borne que la poser sur du sable.
        return None
    mean = float(recent["trimp"].fillna(0.0).mean())
    return mean if mean > 0 else None


def _optimize_ortools(
    allowed: set[str],
    forced_type: str | None,
    max_trimp_week: float,
    soft_constraints: SoftConstraints,
    hard_constraints: list[HardConstraint],
    week_start: date,
    params: dict,
    race_date: date | None,
    objective: str = "min_load",
    stimulus: StimulusTarget | None = None,
    chronic_daily_load: float | None = None,
    acwr_upper: float = 1.5,
    dose_response: DoseResponse | None = None,
) -> WeeklyPlan:
    """
    Résolution CP-SAT avec OR-Tools.

    Le modèle est construit deux fois au plus. La première passe impose le
    stimulus demandé ; si elle est infaisable, la seconde le retire et le
    note. C'est voulu : la version précédente retombait silencieusement sur
    l'heuristique dès qu'une contrainte rendait le problème infaisable, et un
    plan dégradé se lisait alors comme un plan optimal.
    """
    if stimulus is None:
        stimulus = StimulusTarget()
    notes: list[str] = []

    session_list = sorted(allowed)
    n_types = len(session_list)
    type_index = {s: i for i, s in enumerate(session_list)}

    # Facteur d'échelle (CP-SAT travaille en entiers)
    SCALE = 100

    # La limite etait codee en dur a 2 alors que la cle existe dans
    # params.yaml : une constante physiologique n'a pas sa place ici.
    planner_cfg = params.get('planner', {})
    max_high = planner_cfg.get('max_high_intensity_per_week', 2)
    per_type_caps = planner_cfg.get('max_per_type_per_week', {}) or {}
    default_cap = planner_cfg.get('max_same_type_per_week', 3)

    # Valeurs TRIMP et spécificité entiers
    trimp_vals = [int(SESSION_TRIMP_BASE.get(s, 0) * SCALE) for s in session_list]
    spec_vals  = [int(SESSION_TYPES.get(s, {}).get('specificity', 0) * SCALE * 10)
                  for s in session_list]

    # ── Plafond effectif, et plancher rabaisse a sa hauteur ───
    #
    # Le plafond est le plus contraignant de la progression hebdomadaire et de
    # la borne aigu/chronique. Quand le plancher de retention le depasse — un
    # athlete dont la charge chronique s'est effondree — on RABAISSE le
    # plancher au plafond au lieu de l'abandonner. Abandonner le plancher
    # ferait rendre a un objectif de minimisation un plan a zero TRIMP, ce qui
    # est le minimum litteral et un conseil inutile ; le rabaisser rend « le
    # maximum que tu peux faire sans danger », qui est le conseil demande.
    ceiling = max_trimp_week
    if chronic_daily_load:
        ceiling = min(ceiling, 7 * acwr_upper * chronic_daily_load)
    floor = stimulus.min_total_trimp or 0.0
    if floor > ceiling:
        notes.append(
            f"Plancher de charge rabaisse de {floor:.0f} a {ceiling:.0f} TRIMP : "
            "la progression hebdomadaire ou la borne aigu/chronique l'interdit "
            "a cette hauteur."
        )
        floor = ceiling

    # ── Construction du modele, repetable ─────────────────────
    #
    # Trois exigences, par ordre de priorite decroissante. Les contraintes de
    # securite et les indisponibilites declarees sont posees dans toutes les
    # passes : les premieres parce qu'elles protegent l'athlete, les secondes
    # parce qu'elles sont PHYSIQUES — il ne peut pas s'entrainer ce jour-la, et
    # aucun objectif ne rend un creneau disponible. Le plancher de retention et
    # le stimulus specifique sont, eux, des preferences, et cedent dans cet
    # ordre : mieux vaut une semaine legere qu'une semaine impossible.
    def build(with_key_sessions: bool, with_floor: bool = True):
        """Monte le modele ; les deux drapeaux relachent les preferences."""
        model = cp_model.CpModel()

        # Pour chaque jour j et type t -> un booleen x[j][t].
        x = [[model.NewBoolVar(f"x_{j}_{t}") for t in range(n_types)]
             for j in range(7)]
        for j in range(7):
            model.AddExactlyOne(x[j])

        total_trimp_expr = sum(
            x[j][t] * trimp_vals[t] for j in range(7) for t in range(n_types)
        )

        if forced_type and forced_type in type_index:
            fi = type_index[forced_type]
            model.AddAtLeastOne([x[j][fi] for j in range(7)])

        if "REST" in type_index:
            ri = type_index["REST"]
            # Au moins un jour de repos complet. L'indice « dimanche = repos »
            # qui figurait ici a ete retire : c'etait une preference sans
            # fondement, et un `AddHint` que les contraintes rendent
            # infaisable fait ecrire a CP-SAT des pages d'avertissements sur
            # la sortie standard.
            model.Add(sum(x[j][ri] for j in range(7)) >= 1)

        # Bornes par type. Sans elles l'optimiseur empile le type le moins
        # couteux qui satisfait ses contraintes — quatre CALIBRATION, qui est
        # un test. C'est aussi le controle de monotonie le plus direct qu'un
        # modele lineaire permette.
        for t, name in enumerate(session_list):
            if name == "REST":
                continue            # le repos n'a pas a etre rationne
            cap = per_type_caps.get(name, default_cap)
            if cap is not None and cap < 7:
                model.Add(sum(x[j][t] for j in range(7)) <= int(cap))

        high = [type_index[s] for s in ("VMA", "THRESHOLD", "SPECIFIC")
                if s in type_index]
        if high:
            model.Add(sum(x[j][t] for j in range(7) for t in high) <= max_high)
            # Pas deux seances dures consecutives.
            for j in range(6):
                for t1 in high:
                    for t2 in high:
                        model.Add(x[j][t1] + x[j + 1][t2] <= 1)

        if soft_constraints.max_sessions_this_week is not None and "REST" in type_index:
            non_rest = sum(x[j][t] for j in range(7) for t in range(n_types)
                           if session_list[t] != "REST")
            model.Add(non_rest <= soft_constraints.max_sessions_this_week)

        model.Add(total_trimp_expr <= int(max_trimp_week * SCALE))

        # Borne aigu/chronique : la charge de la semaine a venir, ramenee au
        # jour, ne doit pas depasser `acwr_upper` fois la charge chronique.
        # `acute_chronic_ratio` existait dans load_model.py et sa docstring la
        # presentait comme « la seule regle de securite reellement
        # declenchable » — elle n'etait branchee sur rien.
        if chronic_daily_load:
            model.Add(total_trimp_expr
                      <= int(7 * acwr_upper * chronic_daily_load * SCALE))

        # Controle de monotonie, en version lineaire. La monotonie de Foster
        # est un rapport moyenne/ecart-type, donc non lineaire dans les
        # variables de decision. Son contenu pratique — ne pas enchainer
        # indefiniment des journees semblables — se pose en exigeant une
        # journee legere dans toute fenetre glissante.
        light = [i for i, name in enumerate(session_list)
                 if SESSION_TYPES.get(name, {}).get("intensity", 0) <= 1]
        span = stimulus.max_consecutive_training_days
        if light and span > 0:
            for first in range(7 - span):
                model.Add(sum(x[first + k][t]
                              for k in range(span + 1) for t in light) >= 1)

        # Les jours a eviter viennent de l'athlete lui-meme — « je ne peux pas
        # m'entrainer ce jour-la ». Poses en `AddHint`, ils etaient
        # silencieusement ignorables par le solveur : une indisponibilite
        # declaree se retrouvait planifiee. Ce sont des contraintes, et elles
        # passent avant le stimulus prescrit : c'est l'athlete qui sait son
        # emploi du temps.
        if soft_constraints.avoid_days and "REST" in type_index:
            ri = type_index["REST"]
            for d in soft_constraints.avoid_days:
                if 0 <= d < 7:
                    model.Add(x[d][ri] == 1)

        if with_floor and floor > 0:
            model.Add(total_trimp_expr >= int(floor * SCALE))

        if with_key_sessions:
            key = [i for i, name in enumerate(session_list)
                   if SESSION_TYPES.get(name, {}).get("specificity", 0.0)
                   >= stimulus.key_specificity]
            if key and stimulus.min_key_sessions > 0:
                model.Add(sum(x[j][t] for j in range(7) for t in key)
                          >= stimulus.min_key_sessions)

        if objective == "min_load":
            # La charge la plus faible qui delivre le stimulus, et non la plus
            # forte que les contraintes tolerent.
            model.Minimize(total_trimp_expr)
        else:
            model.Maximize(sum(x[j][t] * spec_vals[t]
                               for j in range(7) for t in range(n_types)))
        return model, x

    # ── Resolution, avec puis sans le stimulus ────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    model, x = build(with_key_sessions=True)
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Le dire, et non retomber silencieusement sur l'heuristique : un plan
        # degrade qui ne s'annonce pas se lit comme un plan optimal.
        notes.append(
            f"Stimulus specifique abandonne ({stimulus.min_key_sessions} seance(s) "
            f"a specificite >= {stimulus.key_specificity:.1f}) : infaisable sous "
            "les contraintes de securite et les disponibilites declarees."
        )
        model, x = build(with_key_sessions=False)
        status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE) and floor > 0:
        # Le plancher cede a son tour. Il protege contre le desentrainement,
        # ce qui ne vaut rien face a une semaine sans creneau disponible.
        notes.append(
            f"Plancher de charge abandonne ({floor:.0f} TRIMP) : les "
            "disponibilites declarees ne permettent pas de l'atteindre."
        )
        model, x = build(with_key_sessions=False, with_floor=False)
        status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Infaisable meme sans le stimulus : l'heuristique rend quelque chose,
        # mais le plan doit porter la trace de ce qui n'a pas ete applique.
        plan = _optimize_greedy(
            allowed=allowed,
            forced_type=forced_type,
            max_trimp_week=max_trimp_week,
            soft_constraints=soft_constraints,
            hard_constraints=hard_constraints,
            week_start=week_start,
        )
        plan.objective = objective
        plan.dose_response = dose_response
        plan.notes = [*notes, (
            "Modele infaisable meme sans plancher ni stimulus : plan "
            "heuristique. Ni la charge minimale, ni la borne aigu/chronique, "
            "ni les disponibilites declarees ne sont appliquees."
        )]
        return plan

    # ── Construction du plan ───────────────────────────────────
    plan_days = []
    total_trimp = 0.0
    total_spec = 0.0
    n_sessions = 0

    for j in range(7):
        chosen_type = "REST"
        for t in range(n_types):
            if solver.Value(x[j][t]) == 1:
                chosen_type = session_list[t]
                break

        trimp_est = SESSION_TRIMP_BASE.get(chosen_type, 0)
        spec = SESSION_TYPES.get(chosen_type, {}).get('specificity', 0)
        total_trimp += trimp_est
        total_spec += spec
        if chosen_type != "REST":
            n_sessions += 1

        plan_days.append(DayPlan(
            date=week_start + timedelta(days=j),
            day_name=DAY_NAMES[j],
            session_type=chosen_type,
            trimp_estimated=trimp_est,
            description=SESSION_DESCRIPTIONS.get(chosen_type, ""),
            rationale=_build_rationale(chosen_type, j, hard_constraints, soft_constraints),
        ))

    if chronic_daily_load is None:
        notes.append(
            "Borne aigu/chronique inactive : aucun historique de charge fourni "
            "(passer `daily_load`, ou 28 jours au moins)."
        )

    return WeeklyPlan(
        week_start=week_start,
        days=plan_days,
        total_trimp=round(total_trimp, 1),
        total_sessions=n_sessions,
        specificity_score=round(total_spec / 7, 3),
        hard_constraints_applied=[c.code for c in hard_constraints],
        soft_constraints_applied=soft_constraints.reason if soft_constraints.is_restrictive() else "",
        objective=objective,
        dose_response=dose_response,
        notes=notes,
    )


def _optimize_greedy(
    allowed: set[str],
    forced_type: str | None,
    max_trimp_week: float,
    soft_constraints: SoftConstraints,
    hard_constraints: list[HardConstraint],
    week_start: date,
) -> WeeklyPlan:
    """
    Fallback greedy quand OR-Tools n'est pas disponible.
    Applique un template hebdomadaire heuristique adapté aux contraintes.
    """
    # Template de base : L=END, M=VMA, M=END, J=REST, V=THRESHOLD, S=LONG, D=REST
    base_template = ["ENDURANCE", "VMA", "RECOVERY", "REST",
                     "THRESHOLD", "ENDURANCE", "REST"]

    # Adapter selon ce qui est autorisé
    adjusted = []
    for s in base_template:
        if s in allowed:
            adjusted.append(s)
        elif "RECOVERY" in allowed:
            adjusted.append("RECOVERY")
        else:
            adjusted.append("REST")

    # Si séance forcée
    if forced_type:
        adjusted = [forced_type] * 7

    plan_days = []
    total_trimp = 0.0
    total_spec = 0.0
    n_sessions = 0

    for j, stype in enumerate(adjusted):
        trimp_est = SESSION_TRIMP_BASE.get(stype, 0)
        spec = SESSION_TYPES.get(stype, {}).get('specificity', 0)

        # Respecter le volume max
        if total_trimp + trimp_est > max_trimp_week and stype != "REST":
            stype = "REST"
            trimp_est = 0
            spec = 0

        total_trimp += trimp_est
        total_spec += spec
        if stype != "REST":
            n_sessions += 1

        plan_days.append(DayPlan(
            date=week_start + timedelta(days=j),
            day_name=DAY_NAMES[j],
            session_type=stype,
            trimp_estimated=trimp_est,
            description=SESSION_DESCRIPTIONS.get(stype, ""),
            rationale=_build_rationale(stype, j, hard_constraints, soft_constraints),
        ))

    return WeeklyPlan(
        week_start=week_start,
        days=plan_days,
        total_trimp=round(total_trimp, 1),
        total_sessions=n_sessions,
        specificity_score=round(total_spec / 7, 3),
        hard_constraints_applied=[c.code for c in hard_constraints],
        soft_constraints_applied=soft_constraints.reason if soft_constraints.is_restrictive() else "",
    )


def _build_rationale(
    session_type: str,
    day_idx: int,
    hard_constraints: list[HardConstraint],
    soft_constraints: SoftConstraints,
) -> str:
    """Génère une explication XAI pour le choix d'une séance."""
    reasons = []

    # Contraintes dures appliquées
    for c in hard_constraints:
        if session_type in c.blocked_sessions:
            reasons.append(f"Bloqué par {c.code} : {c.reason}")
        elif c.force_session_type == session_type:
            reasons.append(f"Imposé par {c.code} : {c.reason}")

    # Contexte LLM
    if soft_constraints.is_restrictive() and soft_constraints.reason:
        if session_type in soft_constraints.no_session_types:
            reasons.append(f"Évité par contexte : {soft_constraints.reason}")
        elif soft_constraints.max_volume_pct < 1.0 and session_type != "REST":
            reasons.append(f"Volume réduit ({int(soft_constraints.max_volume_pct*100)}%) : {soft_constraints.reason}")

    # Raisons positives
    if not reasons:
        spec = SESSION_TYPES.get(session_type, {}).get('specificity', 0)
        if spec >= 0.7:
            reasons.append(f"Haute spécificité 1h18 (score={spec:.1f})")
        elif session_type == "REST":
            reasons.append("Récupération planifiée")
        else:
            reasons.append("Volume aérobie de base")

    return " | ".join(reasons)


# ═══════════════════════════════════════════════════════════════
#  MODULE TAPERING
# ═══════════════════════════════════════════════════════════════

def generate_taper_plan(
    race_date: date,
    current_fitness: float,
    week_start: date | None = None,
    params: dict | None = None,
) -> WeeklyPlan:
    """
    Génère un plan tapering optimisé pour maximiser le TSB au jour J
    tout en maintenant la spécificité (intensité préservée, volume réduit).

    Règles Tapering :
      J-14 à J-8  : volume -20%, 1 séance SPECIFIC maintenue
      J-7 à J-4   : volume -40%, séances légères + 1 SPECIFIC finale
      J-3 à J-1   : repos actif (RECOVERY uniquement)
      J            : RACE DAY

    Parameters
    ----------
    race_date : date de la compétition
    current_fitness : fitness actuelle (pour calibrer le volume résiduel)
    week_start : lundi de la semaine (None = lundi prochain)
    params : configuration

    Returns
    -------
    WeeklyPlan tapering
    """
    if params is None:
        params = _load_params()
    if week_start is None:
        today = date.today()
        week_start = today - timedelta(days=today.weekday())

    days_to_race_start = (race_date - week_start).days

    plan_days = []
    total_trimp = 0.0
    total_spec = 0.0

    for j in range(7):
        day = week_start + timedelta(days=j)
        dtr = (race_date - day).days  # jours jusqu'à la course

        if dtr < 0:
            stype = "REST"
            rationale = "Après la course"
        elif dtr == 0:
            stype = "REST"
            rationale = "JOUR J — Course !"
        elif dtr <= 2:
            stype = "RECOVERY"
            rationale = f"J-{dtr} : récupération active pré-course"
        elif dtr <= 4:
            stype = "RECOVERY" if j % 2 == 0 else "REST"
            rationale = f"J-{dtr} : tapering final, repos prioritaire"
        elif dtr <= 7:
            stype = "SPECIFIC" if j == 1 else ("ENDURANCE" if j == 3 else "REST")
            rationale = f"J-{dtr} : maintien intensité spécifique"
        elif dtr <= 14:
            stype = ["ENDURANCE", "THRESHOLD", "RECOVERY", "REST",
                     "SPECIFIC", "ENDURANCE", "REST"][j]
            rationale = f"J-{dtr} : tapering progressif (-20% volume)"
        else:
            stype = "ENDURANCE"
            rationale = "Avant période tapering"

        trimp_est = SESSION_TRIMP_BASE.get(stype, 0)
        spec = SESSION_TYPES.get(stype, {}).get('specificity', 0)
        total_trimp += trimp_est
        total_spec += spec

        plan_days.append(DayPlan(
            date=day,
            day_name=DAY_NAMES[j],
            session_type=stype,
            trimp_estimated=trimp_est,
            description=SESSION_DESCRIPTIONS.get(stype, ""),
            rationale=rationale,
        ))

    return WeeklyPlan(
        week_start=week_start,
        days=plan_days,
        total_trimp=round(total_trimp, 1),
        total_sessions=sum(1 for d in plan_days if d.session_type != "REST"),
        specificity_score=round(total_spec / 7, 3),
        hard_constraints_applied=["TAPER_MODE"],
        soft_constraints_applied=f"Tapering J-{days_to_race_start}",
    )
