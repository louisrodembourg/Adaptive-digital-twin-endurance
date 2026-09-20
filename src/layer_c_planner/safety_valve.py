"""
Layer C — Planner : Safety Valve (Contraintes Dures)

Premier filtre du Planner : vérifie les contraintes physiologiques
AVANT que l'optimizer OR-Tools génère le plan.

Les contraintes dures sont des règles non-négociables basées sur des
signaux physiologiques. Violer ces règles expose l'athlète à un risque
de blessure ou de sur-entraînement.

Contraintes implémentées :
  - NO_HIGH_INTENSITY   : rMSSD < seuil → pas de VMA/Seuil
  - FORCED_REST         : dérive cardiaque hier > 5% → repos obligatoire
  - REDUCE_LOAD_30PCT   : TSB < -30 → réduction 30% du volume
  - CALIBRATION_SESSION : incertitude modèle trop élevée → séance test
  - TAPER_MODE          : J ≤ 14 avant la course → mode tapering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def _load_params(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / 'config' / 'params.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
#  TYPES DE CONTRAINTES DURES
# ═══════════════════════════════════════════════════════════════

# Sessions autorisées par type (pour le solver OR-Tools)
SESSION_TYPES = {
    "REST":       {"trimp_factor": 0.00, "specificity": 0.0, "intensity": 0},
    "RECOVERY":   {"trimp_factor": 0.25, "specificity": 0.1, "intensity": 1},
    "ENDURANCE":  {"trimp_factor": 0.75, "specificity": 0.3, "intensity": 2},
    "THRESHOLD":  {"trimp_factor": 1.00, "specificity": 0.7, "intensity": 3},
    "VMA":        {"trimp_factor": 1.10, "specificity": 0.8, "intensity": 4},
    "SPECIFIC":   {"trimp_factor": 0.90, "specificity": 1.0, "intensity": 3},
    "STRENGTH":   {"trimp_factor": 0.50, "specificity": 0.2, "intensity": 2},
    "CALIBRATION":{"trimp_factor": 0.60, "specificity": 0.5, "intensity": 2},
}

HIGH_INTENSITY_SESSIONS = {"VMA", "THRESHOLD", "SPECIFIC"}
BLOCKED_IF_FORCED_REST = {"VMA", "THRESHOLD", "SPECIFIC", "ENDURANCE", "STRENGTH"}


@dataclass
class HardConstraint:
    """Représente une contrainte dure avec son explication."""
    code: str
    reason: str
    blocked_sessions: set[str] = field(default_factory=set)
    volume_reduction_pct: float = 0.0
    force_session_type: Optional[str] = None

    @property
    def description(self) -> str:
        parts = [f"[{self.code}] {self.reason}"]
        if self.blocked_sessions:
            parts.append(f"  → Sessions bloquées : {', '.join(sorted(self.blocked_sessions))}")
        if self.volume_reduction_pct > 0:
            parts.append(f"  → Volume réduit de {int(self.volume_reduction_pct * 100)}%")
        if self.force_session_type:
            parts.append(f"  → Session imposée : {self.force_session_type}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  SAFETY VALVE PRINCIPALE
# ═══════════════════════════════════════════════════════════════

def check_safety_constraints(
    rmssd_ms: float | None = None,
    tsb: float | None = None,
    decoupling_yesterday: float | None = None,
    dfa_alpha1: float | None = None,
    needs_calibration: bool = False,
    race_date: date | None = None,
    today: date | None = None,
    params: dict | None = None,
) -> list[HardConstraint]:
    """
    Évalue toutes les contraintes dures et retourne la liste active.

    Parameters
    ----------
    rmssd_ms : RMSSD matinal en ms (None si non mesuré)
    tsb : Training Stress Balance courant (Fitness - Fatigue)
    decoupling_yesterday : dérive cardiaque de la sortie de la veille (%)
    dfa_alpha1 : DFA α₁ de la dernière séance (proxy intensité réelle)
    needs_calibration : flag du modèle bayésien
    race_date : date de la compétition cible (pour taper)
    today : date du jour (None = date.today())
    params : configuration

    Returns
    -------
    list[HardConstraint] : contraintes actives (vide = pas de restriction)
    """
    if params is None:
        params = _load_params()
    if today is None:
        today = date.today()

    planner = params.get('planner', {})
    rmssd_threshold = planner.get('rmssd_vma_threshold', 50.0)
    taper_days = planner.get('taper_start_days_before', 14)

    constraints: list[HardConstraint] = []

    # ── 1. HRV BAS → pas de haute intensité ──────────────────
    if rmssd_ms is not None and rmssd_threshold > 0 and rmssd_ms < rmssd_threshold:
        constraints.append(HardConstraint(
            code="NO_HIGH_INTENSITY",
            reason=f"rMSSD={rmssd_ms:.0f}ms < seuil {rmssd_threshold:.0f}ms — SNA fatigué",
            blocked_sessions=HIGH_INTENSITY_SESSIONS,
        ))

    # ── 2. DÉRIVE CARDIAQUE HIER > 5% → repos ou récup ───────
    if decoupling_yesterday is not None and decoupling_yesterday > 5.0:
        alert_type = "FORCED_REST" if decoupling_yesterday > 10.0 else "RECOVERY_ONLY"
        constraints.append(HardConstraint(
            code=alert_type,
            reason=f"Découpling hier={decoupling_yesterday:.1f}% > 5% — endurance insuffisante",
            blocked_sessions=BLOCKED_IF_FORCED_REST if decoupling_yesterday > 10.0
                             else HIGH_INTENSITY_SESSIONS,
            force_session_type="REST" if decoupling_yesterday > 10.0 else None,
        ))

    # ── 3. TSB < -30 → sur-entraînement → réduction volume ───
    if tsb is not None and tsb < -30:
        constraints.append(HardConstraint(
            code="REDUCE_LOAD_30PCT",
            reason=f"TSB={tsb:.1f} < -30 — risque sur-entraînement",
            blocked_sessions={"VMA"},
            volume_reduction_pct=0.30,
        ))

    # ── 4. DFA α₁ BAS → intensité réelle trop élevée ─────────
    if dfa_alpha1 is not None and dfa_alpha1 < 0.75:
        constraints.append(HardConstraint(
            code="DFA_LOW",
            reason=f"DFA α₁={dfa_alpha1:.2f} < 0.75 — système nerveux sollicité",
            blocked_sessions={"VMA"},
        ))

    # ── 5. INCERTITUDE MODÈLE → séance calibration ────────────
    if needs_calibration:
        constraints.append(HardConstraint(
            code="CALIBRATION_SESSION",
            reason="Incertitude modèle trop élevée — séance test requise",
            blocked_sessions={"VMA", "SPECIFIC"},
            force_session_type="CALIBRATION",
        ))

    # ── 6. MODE TAPERING (J-14 avant course) ─────────────────
    if race_date is not None:
        days_to_race = (race_date - today).days
        if 0 <= days_to_race <= taper_days:
            taper_constraint = _get_taper_constraint(days_to_race, taper_days)
            if taper_constraint:
                constraints.append(taper_constraint)

    return constraints


def _get_taper_constraint(days_to_race: int, taper_days: int = 14) -> HardConstraint | None:
    """Retourne la contrainte taper adaptée à la distance jusqu'à la course."""
    if days_to_race <= 0:
        return None

    if days_to_race <= 3:
        return HardConstraint(
            code="TAPER_FINAL",
            reason=f"J-{days_to_race} : phase finale — repos actif uniquement",
            blocked_sessions={"VMA", "THRESHOLD", "SPECIFIC", "ENDURANCE", "STRENGTH"},
            volume_reduction_pct=0.70,
        )
    elif days_to_race <= 7:
        return HardConstraint(
            code="TAPER_WEEK1",
            reason=f"J-{days_to_race} : tapering semaine -1 — volume -40%",
            blocked_sessions={"VMA", "STRENGTH"},
            volume_reduction_pct=0.40,
        )
    else:
        return HardConstraint(
            code="TAPER_WEEK2",
            reason=f"J-{days_to_race} : tapering semaine -2 — volume -20%",
            blocked_sessions={"VMA"},
            volume_reduction_pct=0.20,
        )


# ═══════════════════════════════════════════════════════════════
#  HELPERS — Interface avec OR-Tools
# ═══════════════════════════════════════════════════════════════

def get_allowed_sessions(constraints: list[HardConstraint]) -> set[str]:
    """
    Déduit les types de séances autorisés depuis les contraintes actives.

    Returns
    -------
    set de noms de sessions (subset de SESSION_TYPES.keys())
    """
    all_sessions = set(SESSION_TYPES.keys())
    blocked = set()
    for c in constraints:
        blocked |= c.blocked_sessions
    return all_sessions - blocked


def get_volume_multiplier(constraints: list[HardConstraint]) -> float:
    """
    Retourne le multiplicateur de volume à appliquer (1.0 = pas de réduction).

    Si plusieurs contraintes réduisent le volume, prend la plus restrictive.
    """
    reductions = [c.volume_reduction_pct for c in constraints if c.volume_reduction_pct > 0]
    if not reductions:
        return 1.0
    return 1.0 - max(reductions)


def get_forced_session(constraints: list[HardConstraint]) -> str | None:
    """Retourne le type de séance imposé par une contrainte dure, ou None."""
    for c in constraints:
        if c.force_session_type:
            return c.force_session_type
    return None


def format_constraints_summary(constraints: list[HardConstraint]) -> str:
    """Retourne un résumé textuel lisible des contraintes actives (pour XAI)."""
    if not constraints:
        return "Aucune contrainte dure — plan complet autorisé."
    lines = [f"⚠️  {len(constraints)} contrainte(s) active(s) :"]
    for c in constraints:
        lines.append(c.description)
    return "\n".join(lines)
