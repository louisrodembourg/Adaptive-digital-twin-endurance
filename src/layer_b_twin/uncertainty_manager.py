"""
Layer B — Digital Twin : Gestionnaire d'Incertitude

Surveille la largeur du cône d'incertitude du modèle bayésien.
Quand l'incertitude dépasse un seuil critique, déclenche une séance
test de calibration pour réduire l'incertitude et recalibrer le modèle.

Logique :
  - HDI_width(Fitness) = fitness_high - fitness_low (95% HDI)
  - Si HDI_width > threshold_uncertainty → signal CALIBRATION_SESSION
  - Séance test standard : 5 km à allure 4'00"/km (contrôlée)
    → produit une nouvelle observation de performance pour la posterior
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


def _default_uncertainty_threshold() -> float:
    """Seuil de largeur HDI Fitness au-delà duquel on déclenche une séance test."""
    return 15.0  # Unités TRIMP — empirique, à calibrer


@dataclass
class UncertaintyReport:
    """Résumé de l'état d'incertitude du modèle."""
    fitness_hdi_width: float
    fatigue_hdi_width: float
    tsb_hdi_width: float
    perf_hdi_width: float
    needs_calibration: bool
    calibration_reason: str = ""
    threshold: float = 15.0

    @property
    def uncertainty_level(self) -> str:
        """Niveau d'incertitude : 'low' | 'moderate' | 'high' | 'critical'"""
        ratio = self.fitness_hdi_width / self.threshold
        if ratio < 0.5:
            return "low"
        elif ratio < 1.0:
            return "moderate"
        elif ratio < 1.5:
            return "high"
        return "critical"

    def summary(self) -> str:
        lines = [
            f"Uncertainty Level : {self.uncertainty_level.upper()}",
            f"  Fitness  HDI width : {self.fitness_hdi_width:.1f} (seuil: {self.threshold})",
            f"  Fatigue  HDI width : {self.fatigue_hdi_width:.1f}",
            f"  TSB      HDI width : {self.tsb_hdi_width:.1f}",
        ]
        if self.needs_calibration:
            lines.append(f"  ⚠️  CALIBRATION REQUISE : {self.calibration_reason}")
        return "\n".join(lines)


def assess_uncertainty(
    curves_df: pd.DataFrame,
    threshold: float | None = None,
    window_days: int = 7,
) -> UncertaintyReport:
    """
    Évalue l'incertitude actuelle du modèle à partir des courbes posteriors.

    Parameters
    ----------
    curves_df : DataFrame retourné par extract_posterior_curves()
                Colonnes attendues : fitness_high, fitness_low, etc.
    threshold : seuil HDI Fitness (si None, utilise défaut 15.0)
    window_days : fenêtre pour calculer la largeur HDI (7 derniers jours)

    Returns
    -------
    UncertaintyReport
    """
    if threshold is None:
        threshold = _default_uncertainty_threshold()

    # Prendre les n derniers jours
    recent = curves_df.tail(window_days)

    def hdi_width(col_high: str, col_low: str) -> float:
        if col_high not in recent.columns or col_low not in recent.columns:
            return np.nan
        return float((recent[col_high] - recent[col_low]).mean())

    fit_width  = hdi_width('fitness_high',  'fitness_low')
    fat_width  = hdi_width('fatigue_high',  'fatigue_low')
    tsb_width  = hdi_width('tsb_high',      'tsb_low')
    perf_width = hdi_width('perf_high',     'perf_low')

    # Décision de calibration
    needs_calib = False
    reason = ""

    if np.isnan(fit_width):
        # Pas de données bayésiennes (mode déterministe)
        needs_calib = False
    elif fit_width > threshold * 1.5:
        needs_calib = True
        reason = f"Incertitude critique (HDI={fit_width:.1f} > {threshold*1.5:.1f})"
    elif fit_width > threshold:
        needs_calib = True
        reason = f"Incertitude élevée (HDI={fit_width:.1f} > {threshold:.1f})"

    return UncertaintyReport(
        fitness_hdi_width=fit_width if not np.isnan(fit_width) else 0.0,
        fatigue_hdi_width=fat_width if not np.isnan(fat_width) else 0.0,
        tsb_hdi_width=tsb_width if not np.isnan(tsb_width) else 0.0,
        perf_hdi_width=perf_width if not np.isnan(perf_width) else 0.0,
        needs_calibration=needs_calib,
        calibration_reason=reason,
        threshold=threshold,
    )


def generate_calibration_session() -> dict:
    """
    Retourne la description de la séance test standard de calibration.

    La séance test est conçue pour produire une observation de performance
    propre et réduire l'incertitude du modèle bayésien :
      - 5 km à allure constante et contrôlée (4'00"/km = 4.167 m/s)
      - Conditions : terrain plat, reposé, température neutre
      - Mesures : temps, FC à cette allure, EF_spec
      - Output : une nouvelle observation (day, perf_normalized) pour le modèle

    Returns
    -------
    dict décrivant la séance test
    """
    return {
        "type": "CALIBRATION",
        "description": "Séance test de calibration — réduction d'incertitude",
        "distance_m": 5000,
        "target_pace_min_km": 4.0,
        "target_speed_ms": 4.167,
        "warmup_min": 10,
        "cooldown_min": 10,
        "instructions": [
            "Terrain plat (dénivelé < 20m total)",
            "Allure strictement constante : 4'00\"/km ± 5s",
            "Enregistrer la FC à 4'00\"/km pour mise à jour EF_spec",
            "Ne pas forcer — c'est une mesure, pas une performance",
            "Conditions : reposé (TSB > 0), pas de vent fort",
        ],
        "expected_output": {
            "perf_normalized": "actual_5km_time_s / reference_5km_time_s",
            "ef_spec_calibration": "speed_ms / hr_at_target_pace",
        },
    }


def check_overreaching(tsb_series: np.ndarray, threshold: float = -30.0) -> dict:
    """
    Détecte le sur-entraînement (overreaching) à partir de la courbe TSB.

    TSB = Fitness - Fatigue
      TSB < -30 : sur-entraînement fonctionnel → risque blessure/baisse perf
      TSB > 25  : désentraînement → perte de condition

    Parameters
    ----------
    tsb_series : array TSB (np.ndarray ou colonne DataFrame)
    threshold : seuil sur-entraînement (défaut -30)

    Returns
    -------
    dict avec flags et jours concernés
    """
    tsb = np.asarray(tsb_series)
    current_tsb = float(tsb[-1]) if len(tsb) > 0 else 0.0
    recent_min = float(tsb[-7:].min()) if len(tsb) >= 7 else current_tsb

    is_overreaching = current_tsb < threshold
    is_detrained = current_tsb > 25.0
    approaching = -30.0 <= current_tsb < -20.0

    return {
        'current_tsb': round(current_tsb, 1),
        'recent_7d_min': round(recent_min, 1),
        'is_overreaching': is_overreaching,
        'is_detrained': is_detrained,
        'approaching_overreach': approaching,
        'recommendation': (
            "REPOS FORCÉ : TSB critique, risque sur-entraînement" if is_overreaching
            else "ATTENTION : Approche du seuil de sur-entraînement" if approaching
            else "REPRISE : TSB positif, condition optimale pour performance" if is_detrained
            else "OK"
        ),
    }
