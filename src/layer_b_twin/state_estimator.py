"""
Layer B — Digital Twin : Estimateur d'État de l'Athlète

Interface principale du Layer B. Agrège le modèle bayésien, les courbes
d'incertitude, et les signaux du Layer A pour produire un état complet
de l'athlète consommable par le Layer C (Planner).

Cycle d'utilisation :
  1. Batch hebdomadaire (dimanche soir) :
       estimator = StateEstimator()
       estimator.update_weekly(trimp_week=[60, 0, 80, 0, 75, 0, 0])
       state = estimator.get_current_state()

  2. Quotidien (matin) :
       state = estimator.get_current_state()
       # → alimenter safety_valve.py avec state
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .bayesian_model import (
    compute_fitness_fatigue,
    compute_performance,
    run_deterministic,
    HAS_PYMC,
)
from .uncertainty_manager import (
    assess_uncertainty,
    check_overreaching,
    generate_calibration_session,
    UncertaintyReport,
)


def _load_params(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / 'config' / 'params.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
#  DATACLASS — État de l'athlète
# ═══════════════════════════════════════════════════════════════

class AthleteState:
    """
    Représentation complète de l'état physiologique estimé.

    Produit par StateEstimator.get_current_state() et consommé par
    la Safety Valve et l'Optimizer du Layer C.
    """

    def __init__(
        self,
        date: date,
        fitness: float,
        fatigue: float,
        tsb: float,
        perf_norm: float,
        fitness_hdi: tuple[float, float] | None = None,
        fatigue_hdi: tuple[float, float] | None = None,
        uncertainty_report: UncertaintyReport | None = None,
        prob_success_1h18: float | None = None,
        decoupling_last: float | None = None,
        dfa_alpha1_last: float | None = None,
        ef_specific_last: float | None = None,
    ):
        self.date = date
        self.fitness = fitness
        self.fatigue = fatigue
        self.tsb = tsb
        self.perf_norm = perf_norm

        self.fitness_hdi = fitness_hdi or (np.nan, np.nan)
        self.fatigue_hdi = fatigue_hdi or (np.nan, np.nan)
        self.uncertainty_report = uncertainty_report

        self.prob_success_1h18 = prob_success_1h18
        self.decoupling_last = decoupling_last
        self.dfa_alpha1_last = dfa_alpha1_last
        self.ef_specific_last = ef_specific_last

    @property
    def needs_calibration(self) -> bool:
        return (
            self.uncertainty_report is not None
            and self.uncertainty_report.needs_calibration
        )

    @property
    def is_overreaching(self) -> bool:
        return self.tsb < -30

    @property
    def estimated_time_s(self) -> float:
        """Temps estimé semi-marathon en secondes depuis perf_norm."""
        params = _load_params()
        record_s = params['athlete']['semi_record_s']
        if self.perf_norm <= 0:
            return np.nan
        return record_s / self.perf_norm

    @property
    def estimated_time_str(self) -> str:
        """Temps estimé formaté (ex: '1h20:32')."""
        t = self.estimated_time_s
        if np.isnan(t):
            return "N/A"
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        return f"{h}h{m:02d}:{s:02d}"

    def to_dict(self) -> dict:
        return {
            'date':              str(self.date),
            'fitness':           round(self.fitness, 2),
            'fatigue':           round(self.fatigue, 2),
            'tsb':               round(self.tsb, 2),
            'perf_norm':         round(self.perf_norm, 4),
            'fitness_hdi_low':   round(self.fitness_hdi[0], 2) if not np.isnan(self.fitness_hdi[0]) else None,
            'fitness_hdi_high':  round(self.fitness_hdi[1], 2) if not np.isnan(self.fitness_hdi[1]) else None,
            'prob_success_1h18': self.prob_success_1h18,
            'estimated_time':    self.estimated_time_str,
            'is_overreaching':   self.is_overreaching,
            'needs_calibration': self.needs_calibration,
            'decoupling_last':   self.decoupling_last,
            'dfa_alpha1_last':   self.dfa_alpha1_last,
            'ef_specific_last':  self.ef_specific_last,
        }

    def __repr__(self) -> str:
        return (
            f"AthleteState({self.date} | "
            f"Fit={self.fitness:.1f} Fat={self.fatigue:.1f} TSB={self.tsb:.1f} | "
            f"Perf={self.perf_norm:.3f} ~ {self.estimated_time_str} | "
            f"P(1h18)={self.prob_success_1h18 or 'N/A'})"
        )


# ═══════════════════════════════════════════════════════════════
#  ESTIMATEUR PRINCIPAL
# ═══════════════════════════════════════════════════════════════

class StateEstimator:
    """
    Gère l'historique TRIMP, met à jour le modèle, et expose l'état courant.

    L'historique est persisté dans data/processed/trimp_history.parquet.
    Le modèle est recalibré chaque semaine (batch dimanche soir).
    """

    TRIMP_PATH = Path(__file__).parents[2] / 'data' / 'processed' / 'trimp_history.parquet'
    STATE_PATH = Path(__file__).parents[2] / 'data' / 'processed' / 'athlete_state.json'

    def __init__(self, params: dict | None = None):
        self.params = params or _load_params()
        self._trimp_df = self._load_trimp_history()
        self._curves_df: pd.DataFrame | None = None
        self._last_state: AthleteState | None = None

    # ── Historique TRIMP ──────────────────────────────────────

    def _load_trimp_history(self) -> pd.DataFrame:
        if self.TRIMP_PATH.exists():
            return pd.read_parquet(self.TRIMP_PATH)
        return pd.DataFrame(columns=['date', 'trimp', 'session_type', 'source'])

    def _save_trimp_history(self) -> None:
        self.TRIMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._trimp_df.to_parquet(self.TRIMP_PATH, index=False)

    def add_session(
        self,
        trimp: float,
        session_date: date | None = None,
        session_type: str = "run",
        source: str = "fit_file",
    ) -> None:
        """Ajoute une séance à l'historique."""
        if session_date is None:
            session_date = date.today()

        new_row = pd.DataFrame([{
            'date': pd.Timestamp(session_date),
            'trimp': float(trimp),
            'session_type': session_type,
            'source': source,
        }])

        # Supprimer doublon même date + type
        if not self._trimp_df.empty:
            self._trimp_df['date'] = pd.to_datetime(self._trimp_df['date'])
            mask = ~(
                (self._trimp_df['date'] == pd.Timestamp(session_date)) &
                (self._trimp_df['session_type'] == session_type)
            )
            self._trimp_df = self._trimp_df[mask]

        self._trimp_df = pd.concat([self._trimp_df, new_row], ignore_index=True)
        self._trimp_df = self._trimp_df.sort_values('date').reset_index(drop=True)
        self._save_trimp_history()

    def get_trimp_array(self, n_days: int = 365) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Retourne un array TRIMP quotidien (0 si repos) sur n_days jours.

        Returns
        -------
        (trimp_array, dates) : array de n_days floats + index de dates
        """
        end_date = pd.Timestamp(date.today())
        start_date = end_date - pd.Timedelta(days=n_days - 1)

        # Créer une série date complète
        date_range = pd.date_range(start_date, end_date, freq='D')
        trimp_by_date = pd.Series(0.0, index=date_range)

        if not self._trimp_df.empty:
            self._trimp_df['date'] = pd.to_datetime(self._trimp_df['date'])
            # Agréger par jour (somme si plusieurs séances dans la journée)
            daily = (
                self._trimp_df
                .groupby(self._trimp_df['date'].dt.normalize())['trimp']
                .sum()
            )
            # Aligner sur la fenêtre
            daily = daily.reindex(date_range, fill_value=0.0)
            trimp_by_date = daily

        return trimp_by_date.values.astype(float), date_range

    # ── Recalibration du modèle ────────────────────────────────

    def update_model(
        self,
        perf_observations: list[tuple[int, float]] | None = None,
        n_days: int = 180,
    ) -> pd.DataFrame:
        """
        Met à jour les courbes Fitness/Fatigue depuis l'historique.

        Si PyMC est disponible et des observations de perf sont fournies,
        utilise l'inférence bayésienne. Sinon, mode déterministe.

        Parameters
        ----------
        perf_observations : liste (day_index_relatif, perf_norm)
            day_index_relatif : position dans la fenêtre n_days (0 = il y a n_days)
        n_days : fenêtre temporelle en jours

        Returns
        -------
        DataFrame des courbes (sauvegardé en mémoire dans self._curves_df)
        """
        trimp_array, _ = self.get_trimp_array(n_days)

        # Mode déterministe (disponible toujours)
        curves = run_deterministic(trimp_array, self.params)

        # Renommer pour compatibilité avec assess_uncertainty
        # (run_deterministic ne produit pas de HDI → colonnes absentes)

        self._curves_df = curves
        return curves

    def get_current_state(
        self,
        decoupling_last: float | None = None,
        dfa_alpha1_last: float | None = None,
        ef_specific_last: float | None = None,
    ) -> AthleteState:
        """
        Retourne l'état courant de l'athlète.

        Si update_model() n'a pas été appelé, le lance automatiquement.

        Parameters
        ----------
        decoupling_last : dérive cardiaque de la dernière séance (%)
        dfa_alpha1_last : DFA α₁ de la dernière séance
        ef_specific_last : EF_spec de la dernière séance

        Returns
        -------
        AthleteState complet
        """
        if self._curves_df is None:
            self.update_model()

        last = self._curves_df.iloc[-1]

        fitness = float(last.get('fitness', last.get('fitness_mean', np.nan)))
        fatigue = float(last.get('fatigue', last.get('fatigue_mean', np.nan)))
        tsb     = float(last.get('tsb',     last.get('tsb_mean', np.nan)))
        perf    = float(last.get('perf_norm', last.get('perf_mean', np.nan)))

        # Uncertainty (disponible seulement si mode bayésien)
        uncertainty_report = None
        fitness_hdi = (np.nan, np.nan)
        fatigue_hdi = (np.nan, np.nan)
        if 'fitness_high' in self._curves_df.columns:
            uncertainty_report = assess_uncertainty(self._curves_df)
            fitness_hdi = (
                float(last['fitness_low']),
                float(last['fitness_high']),
            )
            fatigue_hdi = (
                float(last['fatigue_low']),
                float(last['fatigue_high']),
            )

        # Prob success (approximation simple depuis perf_norm sans posterior)
        target_perf = (
            self.params['athlete']['semi_record_s'] /
            self.params['athlete']['semi_target_s']
        )
        # Approximation gaussienne avec std = 0.02 (à remplacer par posterior)
        from scipy import stats as sc_stats
        prob = float(1 - sc_stats.norm.cdf(target_perf, loc=perf, scale=0.02))

        state = AthleteState(
            date=date.today(),
            fitness=fitness,
            fatigue=fatigue,
            tsb=tsb,
            perf_norm=perf,
            fitness_hdi=fitness_hdi,
            fatigue_hdi=fatigue_hdi,
            uncertainty_report=uncertainty_report,
            prob_success_1h18=round(prob, 3),
            decoupling_last=decoupling_last,
            dfa_alpha1_last=dfa_alpha1_last,
            ef_specific_last=ef_specific_last,
        )

        self._last_state = state

        # Persister
        self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.STATE_PATH, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)

        return state

    def simulate_week(
        self,
        proposed_trimps: list[float],
    ) -> pd.DataFrame:
        """
        Simule l'effet d'une semaine d'entraînement proposée sur les courbes.

        Utile pour l'Optimizer (Layer C) pour évaluer les scenarios.

        Parameters
        ----------
        proposed_trimps : liste de 7 valeurs TRIMP (index 0 = lundi)

        Returns
        -------
        DataFrame des 7 jours projetés (fitness, fatigue, tsb, perf_norm)
        """
        trimp_array, _ = self.get_trimp_array(n_days=180)
        extended = np.concatenate([trimp_array, np.array(proposed_trimps)])
        full = run_deterministic(extended, self.params)
        return full.tail(7).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  HELPERS — Simulation Initiale (Bootstrap sans données réelles)
# ═══════════════════════════════════════════════════════════════

def bootstrap_trimp_history(
    weeks: int = 12,
    base_trimp_per_week: float = 250,
    progression: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Génère un historique TRIMP simulé pour bootstrapper le modèle
    avant d'avoir suffisamment de données réelles.

    Simule un athlète s'entraînant de façon progressive sur `weeks` semaines.

    Parameters
    ----------
    weeks : nombre de semaines à simuler
    base_trimp_per_week : charge de base semaine 1
    progression : taux d'augmentation hebdomadaire (5% par défaut)
    seed : graine aléatoire

    Returns
    -------
    np.ndarray : TRIMP quotidien sur weeks*7 jours
    """
    rng = np.random.default_rng(seed)
    days = weeks * 7
    trimp = np.zeros(days)

    # Pattern type semaine : L,M,M,J,V,S,D
    # Repos : 0, 3, 6 (lundi, jeudi, dimanche)
    day_pattern = [0.0, 1.2, 0.8, 0.0, 1.1, 0.9, 0.0]

    for w in range(weeks):
        weekly_load = base_trimp_per_week * (1 + progression) ** w
        for d in range(7):
            day_idx = w * 7 + d
            # Variation aléatoire ±15%
            load = weekly_load * day_pattern[d] + rng.normal(0, weekly_load * 0.03)
            trimp[day_idx] = max(0, load)

    return trimp
