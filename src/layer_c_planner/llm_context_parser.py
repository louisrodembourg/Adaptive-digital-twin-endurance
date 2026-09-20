"""
Layer C — Planner : LLM Context Parser (Analyseur de Contexte)

Traduit le contexte naturel de l'athlète ("je suis stressé cette semaine")
en contraintes mathématiques pour l'optimizer OR-Tools.

Utilise l'API Anthropic (Claude) comme moteur de compréhension du langage.
Fonctionne aussi en mode offline avec des règles heuristiques simples.

Exemples de traductions :
  "Je suis stressé"         → max_volume_pct=0.80
  "Mauvais sommeil 5h"      → no_session_types=["VMA", "THRESHOLD"]
  "Jambes lourdes"          → max_volume_pct=0.70, no_session_types=["SPECIFIC"]
  "Je me sens en forme"     → max_volume_pct=1.0 (pas de restriction)
  "Grand week-end famille"  → max_sessions_this_week=3
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import asdict, dataclass, field

# Anthropic API — import conditionnel
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    warnings.warn(
        "Package 'anthropic' non installé. LLM context parser fonctionnera "
        "en mode heuristique.\npip install anthropic",
        ImportWarning, stacklevel=2
    )


# ═══════════════════════════════════════════════════════════════
#  DATACLASS — Contraintes Douces (Soft Constraints)
# ═══════════════════════════════════════════════════════════════

@dataclass
class SoftConstraints:
    """
    Contraintes douces issues du contexte LLM.
    Contrairement aux HardConstraints, l'optimizer peut les violer
    si nécessaire, mais avec un coût de pénalité.
    """
    max_volume_pct: float = 1.0          # 1.0 = pas de restriction
    no_session_types: list[str] = field(default_factory=list)
    max_sessions_this_week: int | None = None
    prefer_morning: bool = False
    avoid_days: list[int] = field(default_factory=list)  # 0=lundi...6=dimanche
    reason: str = ""
    confidence: float = 1.0             # confiance du LLM [0-1]

    def is_restrictive(self) -> bool:
        """True si des restrictions sont appliquées."""
        return (
            self.max_volume_pct < 1.0
            or bool(self.no_session_types)
            or self.max_sessions_this_week is not None
            or bool(self.avoid_days)
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def no_restriction(cls) -> SoftConstraints:
        return cls(reason="Aucun contexte restrictif détecté.")

    def summary(self) -> str:
        if not self.is_restrictive():
            return "Aucune restriction contextuelle."
        parts = [f"Contexte LLM : {self.reason}"]
        if self.max_volume_pct < 1.0:
            parts.append(f"  Volume max : {int(self.max_volume_pct * 100)}%")
        if self.no_session_types:
            parts.append(f"  Sessions évitées : {', '.join(self.no_session_types)}")
        if self.max_sessions_this_week:
            parts.append(f"  Séances max cette semaine : {self.max_sessions_this_week}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT pour Claude
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """
Tu es un assistant expert en planification d'entraînement sportif (semi-marathon).
Tu analyses le contexte de l'athlète et le traduis en contraintes JSON précises.

Contexte du projet : athlète visant 1h18 semi-marathon (actuellement 1h22).
Types de séances : REST, RECOVERY, ENDURANCE, THRESHOLD, VMA, SPECIFIC, STRENGTH.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte supplémentaire.
Format :
{
  "max_volume_pct": <float entre 0.5 et 1.0>,
  "no_session_types": <liste de types à éviter, peut être []>,
  "max_sessions_this_week": <entier ou null>,
  "avoid_days": <liste d'entiers 0-6 (lundi=0), peut être []>,
  "reason": "<explication courte en français>",
  "confidence": <float entre 0.0 et 1.0>
}

Règles d'interprétation :
- "stressé" / "stress au travail" → max_volume_pct=0.80, avoid VMA
- "mauvais sommeil" / "sommeil court" → no_session_types=["VMA","THRESHOLD"]
- "jambes lourdes" / "courbatures" → max_volume_pct=0.70, no SPECIFIC/VMA
- "fatigué" / "épuisé" → max_volume_pct=0.70, no high intensity
- "en forme" / "bien reposé" → max_volume_pct=1.0, pas de restriction
- "voyage" / "déplacement" → max_sessions_this_week=3
- "blessé" / "douleur" → max_volume_pct=0.0 (repos complet), confidence=0.9
- Si contexte ambigu → confidence < 0.7
"""


# ═══════════════════════════════════════════════════════════════
#  PARSEUR LLM
# ═══════════════════════════════════════════════════════════════

def parse_context_to_constraints(
    user_input: str,
    api_key: str | None = None,
    model: str = "claude-opus-4-6",
    use_heuristic_fallback: bool = True,
) -> SoftConstraints:
    """
    Traduit le texte libre de l'athlète en SoftConstraints.

    Essaie d'abord via l'API Anthropic (Claude). Si indisponible,
    utilise les règles heuristiques.

    Parameters
    ----------
    user_input : texte libre (ex: "je suis épuisé et stressé cette semaine")
    api_key : clé API Anthropic (si None, cherche ANTHROPIC_API_KEY env var)
    model : modèle Claude à utiliser
    use_heuristic_fallback : si True, utilise heuristiques en cas d'erreur API

    Returns
    -------
    SoftConstraints prêtes à passer à l'optimizer
    """
    if not user_input or not user_input.strip():
        return SoftConstraints.no_restriction()

    # Tentative via API Claude
    if HAS_ANTHROPIC:
        try:
            return _parse_via_claude(user_input, api_key, model)
        except Exception as e:
            if not use_heuristic_fallback:
                raise
            warnings.warn(f"LLM API error, fallback heuristique : {e}", UserWarning, stacklevel=2)

    # Fallback heuristique
    return _parse_heuristic(user_input)


def _parse_via_claude(
    user_input: str,
    api_key: str | None,
    model: str,
) -> SoftConstraints:
    """Appel API Anthropic pour analyser le contexte."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY non définie.")

    client = anthropic.Anthropic(api_key=key)

    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Contexte de l'athlète : {user_input}",
        }]
    )

    raw = message.content[0].text.strip()

    # Extraire le JSON si entouré de markdown
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    data = json.loads(raw)
    return _dict_to_soft_constraints(data)


def _parse_heuristic(user_input: str) -> SoftConstraints:
    """
    Règles heuristiques simples pour analyser le contexte sans LLM.
    Utilisé comme fallback quand l'API est indisponible.
    """
    text = user_input.lower()

    max_volume = 1.0
    no_sessions: list[str] = []
    max_sessions = None
    reason_parts = []

    # Mots-clés négatifs
    if any(w in text for w in ["blessé", "blessure", "douleur", "mal au"]):
        return SoftConstraints(
            max_volume_pct=0.0,
            no_session_types=["VMA", "THRESHOLD", "SPECIFIC", "ENDURANCE"],
            reason="Blessure/douleur détectée — repos recommandé",
            confidence=0.85,
        )

    if any(w in text for w in ["épuisé", "épuisement", "exhausted"]):
        max_volume = min(max_volume, 0.65)
        no_sessions += ["VMA", "THRESHOLD", "SPECIFIC"]
        reason_parts.append("épuisement")

    if any(w in text for w in ["fatigué", "fatigue", "tired", "crevé"]):
        max_volume = min(max_volume, 0.75)
        no_sessions += ["VMA"]
        reason_parts.append("fatigue")

    if any(w in text for w in ["stressé", "stress", "anxieux", "anxiété"]):
        max_volume = min(max_volume, 0.80)
        no_sessions += ["VMA"]
        reason_parts.append("stress")

    if any(w in text for w in ["mauvais sommeil", "mal dormi", "insomnie", "pas dormi"]):
        max_volume = min(max_volume, 0.80)
        no_sessions += ["VMA", "THRESHOLD"]
        reason_parts.append("sommeil insuffisant")

    if any(w in text for w in ["jambes lourdes", "courbatures", "doms"]):
        max_volume = min(max_volume, 0.75)
        no_sessions += ["SPECIFIC", "VMA"]
        reason_parts.append("jambes lourdes")

    if any(w in text for w in ["voyage", "déplacement", "travail intensif"]):
        max_sessions = 3
        max_volume = min(max_volume, 0.85)
        reason_parts.append("contraintes logistiques")

    # Mots-clés positifs
    if any(w in text for w in ["en forme", "bien reposé", "frais", "motivé"]):
        max_volume = min(max_volume, 1.0)
        no_sessions = []
        reason_parts = ["bonne condition détectée"]

    reason = ", ".join(reason_parts) if reason_parts else "aucune restriction détectée"
    confidence = 0.7 if reason_parts else 1.0

    return SoftConstraints(
        max_volume_pct=max_volume,
        no_session_types=list(set(no_sessions)),
        max_sessions_this_week=max_sessions,
        reason=reason,
        confidence=confidence,
    )


def _dict_to_soft_constraints(data: dict) -> SoftConstraints:
    """Convertit un dict (retourné par Claude) en SoftConstraints."""
    return SoftConstraints(
        max_volume_pct=float(data.get('max_volume_pct', 1.0)),
        no_session_types=list(data.get('no_session_types', [])),
        max_sessions_this_week=data.get('max_sessions_this_week'),
        avoid_days=list(data.get('avoid_days', [])),
        reason=str(data.get('reason', '')),
        confidence=float(data.get('confidence', 1.0)),
    )
