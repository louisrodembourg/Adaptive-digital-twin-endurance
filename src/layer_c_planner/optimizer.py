"""
Layer C — Planner : Optimizer OR-Tools (CP-SAT)

Génère le plan d'entraînement hebdomadaire optimal en respectant
les contraintes dures (Safety Valve) et douces (LLM Context Parser).

Approche Neuro-Symbolique :
  - Symbolique (déterministe) : OR-Tools CP-SAT résout le problème
    d'optimisation sous contraintes → résultats reproductibles et fiables
  - Neuro (LLM) : Claude traduit le contexte naturel en contraintes mathématiques
    → flexibilité et compréhension du langage humain

Objectif : maximiser la charge pondérée par la spécificité (allure 3'40"/km)
           sous contraintes de sécurité et de progression.

Variables de décision : session_type[jour] pour 7 jours (lundi → dimanche)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

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

from .safety_valve import (
    SESSION_TYPES,
    HardConstraint,
    get_allowed_sessions,
    get_volume_multiplier,
    get_forced_session,
)
from .llm_context_parser import SoftConstraints


def _load_params(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / 'config' / 'params.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
#  DATACLASS — Plan Hebdomadaire
# ═══════════════════════════════════════════════════════════════

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
        """Rapport d'explication (XAI) des choix du planner."""
        lines = ["=== Explication du Plan (XAI) ==="]
        for day in self.days:
            if day.rationale:
                lines.append(f"{day.day_name} [{day.session_type}] : {day.rationale}")
        if self.soft_constraints_applied:
            lines.append(f"\nContexte LLM appliqué : {self.soft_constraints_applied}")
        return "\n".join(lines)


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
    soft_constraints: Optional[SoftConstraints] = None,
    previous_week_trimp: float = 250.0,
    week_start: Optional[date] = None,
    params: dict | None = None,
    race_date: Optional[date] = None,
) -> WeeklyPlan:
    """
    Génère le plan optimal pour la semaine via OR-Tools CP-SAT.

    Parameters
    ----------
    hard_constraints : contraintes dures (Safety Valve)
    soft_constraints : contraintes douces (LLM Context Parser)
    previous_week_trimp : TRIMP total de la semaine précédente
    week_start : lundi de la semaine (None = lundi prochain)
    params : configuration
    race_date : date de la compétition (pour taper)

    Returns
    -------
    WeeklyPlan avec les 7 jours planifiés
    """
    if params is None:
        params = _load_params()
    if soft_constraints is None:
        soft_constraints = SoftConstraints()
    if week_start is None:
        today = date.today()
        week_start = today - timedelta(days=today.weekday())  # lundi courant

    planner = params.get('planner', {})
    max_progression = 1 + planner.get('max_weekly_progression_pct', 10) / 100  # 1.10

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


def _optimize_ortools(
    allowed: set[str],
    forced_type: str | None,
    max_trimp_week: float,
    soft_constraints: SoftConstraints,
    hard_constraints: list[HardConstraint],
    week_start: date,
    params: dict,
    race_date: date | None,
) -> WeeklyPlan:
    """Résolution CP-SAT avec OR-Tools."""
    model = cp_model.CpModel()

    session_list = sorted(allowed)
    n_types = len(session_list)
    type_index = {s: i for i, s in enumerate(session_list)}

    # Facteur d'échelle (CP-SAT travaille en entiers)
    SCALE = 100

    # Valeurs TRIMP et spécificité entiers
    trimp_vals = [int(SESSION_TRIMP_BASE.get(s, 0) * SCALE) for s in session_list]
    spec_vals  = [int(SESSION_TYPES.get(s, {}).get('specificity', 0) * SCALE * 10)
                  for s in session_list]

    # ── Variables de décision : session_type[jour] ─────────────
    # Représenté comme : pour chaque jour j et type t → bool x[j][t]
    x = [[model.NewBoolVar(f"x_{j}_{t}") for t in range(n_types)]
         for j in range(7)]

    # Contrainte : exactement 1 type par jour
    for j in range(7):
        model.AddExactlyOne(x[j])

    # ── Contrainte : séance imposée si forced_type ─────────────
    if forced_type and forced_type in type_index:
        fi = type_index[forced_type]
        # Au moins 1 jour doit avoir la séance forcée
        model.AddAtLeastOne([x[j][fi] for j in range(7)])

    # ── Contrainte : jours de repos (dimanche toujours REST si possible) ──
    if "REST" in type_index:
        ri = type_index["REST"]
        model.AddHint(x[6][ri], 1)  # Hint : dimanche = repos

    # ── Contrainte : max 2 séances haute intensité par semaine ─
    high_intensity_types = [
        type_index[s] for s in ["VMA", "THRESHOLD", "SPECIFIC"]
        if s in type_index
    ]
    if high_intensity_types:
        high_intensity_count = sum(
            x[j][t] for j in range(7) for t in high_intensity_types
        )
        model.Add(high_intensity_count <= 2)

    # ── Contrainte : au moins 1 jour de repos ─────────────────
    if "REST" in type_index:
        ri = type_index["REST"]
        rest_count = sum(x[j][ri] for j in range(7))
        model.Add(rest_count >= 1)

    # ── Contrainte : max sessions par semaine (soft → pénalité) ─
    if soft_constraints.max_sessions_this_week is not None:
        if "REST" in type_index:
            ri = type_index["REST"]
            non_rest_count = sum(
                x[j][t] for j in range(7) for t in range(n_types)
                if session_list[t] != "REST"
            )
            model.Add(non_rest_count <= soft_constraints.max_sessions_this_week)

    # ── Contrainte : volume total max ─────────────────────────
    total_trimp_expr = sum(
        x[j][t] * trimp_vals[t]
        for j in range(7) for t in range(n_types)
    )
    model.Add(total_trimp_expr <= int(max_trimp_week * SCALE))

    # ── Contrainte : pas 2 séances haute intensité consécutives ─
    for j in range(6):
        for t1 in high_intensity_types:
            for t2 in high_intensity_types:
                model.Add(x[j][t1] + x[j+1][t2] <= 1)

    # ── Contrainte : jours à éviter (soft) ────────────────────
    if soft_constraints.avoid_days and "REST" in type_index:
        ri = type_index["REST"]
        for d in soft_constraints.avoid_days:
            if 0 <= d < 7:
                model.AddHint(x[d][ri], 1)

    # ── Objectif : maximiser TRIMP × spécificité ──────────────
    obj = sum(
        x[j][t] * spec_vals[t]
        for j in range(7) for t in range(n_types)
    )
    model.Maximize(obj)

    # ── Résolution ────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Fallback si infaisable
        return _optimize_greedy(
            allowed, forced_type, max_trimp_week,
            soft_constraints, hard_constraints, week_start
        )

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

    return WeeklyPlan(
        week_start=week_start,
        days=plan_days,
        total_trimp=round(total_trimp, 1),
        total_sessions=n_sessions,
        specificity_score=round(total_spec / 7, 3),
        hard_constraints_applied=[c.code for c in hard_constraints],
        soft_constraints_applied=soft_constraints.reason if soft_constraints.is_restrictive() else "",
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
    week_start: Optional[date] = None,
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
