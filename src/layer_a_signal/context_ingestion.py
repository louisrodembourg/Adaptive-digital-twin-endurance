"""
Layer A — Signal Lab : Ingestion du Contexte Quotidien
Capture et stockage des données subjectives et HRV matinales.

Le contexte quotidien alimente la Safety Valve (Layer C) pour adapter
le plan d'entraînement en temps réel.

Données collectées :
  - rmssd_ms       : HRV matinale (RMSSD en ms) — principal indicateur SNA
  - sleep_hours    : Durée de sommeil
  - sleep_quality  : Qualité de 1 (très mauvaise) à 5 (excellente)
  - rpe_yesterday  : RPE de la veille (Borg 1-10)
  - stress_level   : Stress perçu de 1 (minimal) à 5 (élevé)
  - notes          : Texte libre (LLM context parsing en Layer C)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


from ..config import load_params as _shared_load_params


def _load_params(config_path: str | Path | None = None) -> dict:
    """Parametres bruts — delegue au chargeur unique et mis en cache."""
    return _shared_load_params(config_path)


# ═══════════════════════════════════════════════════════════════
#  1. DATACLASS — Contexte Journalier
# ═══════════════════════════════════════════════════════════════

@dataclass
class DailyContext:
    """
    Représente le contexte physiologique et subjectif d'une journée.

    Attributes
    ----------
    date : date du jour
    rmssd_ms : RMSSD matinal en ms (HRV). None si non mesuré.
    sleep_hours : heures de sommeil (0-24)
    sleep_quality : qualité du sommeil de 1 (très mauvaise) à 5 (excellente)
    rpe_yesterday : RPE de la séance de la veille (Borg 1-10). None si repos.
    stress_level : stress perçu de 1 (minimal) à 5 (élevé)
    notes : texte libre pour LLM context parsing (ex: "jambes lourdes, stressé")
    """
    date: date
    rmssd_ms: Optional[float] = None
    sleep_hours: float = 7.0
    sleep_quality: int = 3       # 1-5
    rpe_yesterday: Optional[float] = None  # 1-10 Borg
    stress_level: int = 2        # 1-5
    notes: str = ""

    def __post_init__(self):
        # Validation des bornes
        self.sleep_quality = int(np.clip(self.sleep_quality, 1, 5))
        self.stress_level = int(np.clip(self.stress_level, 1, 5))
        self.sleep_hours = float(np.clip(self.sleep_hours, 0, 24))
        if self.rpe_yesterday is not None:
            self.rpe_yesterday = float(np.clip(self.rpe_yesterday, 1, 10))
        if self.rmssd_ms is not None:
            self.rmssd_ms = float(max(0, self.rmssd_ms))

    @property
    def fatigue_score(self) -> float:
        """
        Score de fatigue composite [0-1] combinant HRV, sommeil, stress, RPE.

        Plus le score est élevé, plus l'athlète est fatigué.
        Utilisé par la Safety Valve pour décider des ajustements.
        """
        components = []

        # HRV : rMSSD bas = fatigue du SNA
        if self.rmssd_ms is not None:
            # Normalisation : 20ms = très fatigué (1.0), 80ms = frais (0.0)
            hrv_score = 1.0 - np.clip((self.rmssd_ms - 20) / 60, 0, 1)
            components.append(hrv_score * 2.0)  # poids double

        # Sommeil
        sleep_score = 1.0 - np.clip((self.sleep_hours - 5) / 4, 0, 1)
        qual_score = (5 - self.sleep_quality) / 4
        components.append((sleep_score + qual_score) / 2)

        # Stress
        components.append((self.stress_level - 1) / 4)

        # RPE de la veille
        if self.rpe_yesterday is not None:
            components.append(np.clip((self.rpe_yesterday - 1) / 9, 0, 1))

        return float(np.mean(components)) if components else 0.5

    @property
    def is_high_fatigue(self) -> bool:
        """True si le score de fatigue dépasse 0.65 → alerte Safety Valve."""
        return self.fatigue_score >= 0.65

    @property
    def is_hrv_low(self, threshold_ms: float = 50.0) -> bool:
        """True si rMSSD < seuil (défaut 50ms) → bloquer séance haute intensité."""
        if self.rmssd_ms is None:
            return False
        return self.rmssd_ms < threshold_ms

    def to_dict(self) -> dict:
        d = asdict(self)
        d['date'] = str(self.date)
        d['fatigue_score'] = round(self.fatigue_score, 3)
        d['is_high_fatigue'] = self.is_high_fatigue
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'DailyContext':
        d = d.copy()
        if isinstance(d.get('date'), str):
            d['date'] = date.fromisoformat(d['date'])
        # Supprimer les clés calculées
        d.pop('fatigue_score', None)
        d.pop('is_high_fatigue', None)
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> 'DailyContext':
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ═══════════════════════════════════════════════════════════════
#  2. STORAGE — Persistance en Parquet
# ═══════════════════════════════════════════════════════════════

class DailyContextStore:
    """
    Gère le stockage et la récupération des contextes quotidiens.

    Les données sont stockées dans data/processed/daily_context.parquet
    (une ligne par jour, mise à jour incrémentale).
    """

    DEFAULT_PATH = Path(__file__).parents[2] / 'data' / 'processed' / 'daily_context.parquet'

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else self.DEFAULT_PATH

    def _load_existing(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_parquet(self.path)
        # DataFrame vide avec le bon schema
        return pd.DataFrame(columns=[
            'date', 'rmssd_ms', 'sleep_hours', 'sleep_quality',
            'rpe_yesterday', 'stress_level', 'notes', 'fatigue_score', 'is_high_fatigue'
        ])

    def save(self, ctx: DailyContext) -> None:
        """Sauvegarde ou met à jour le contexte du jour donné."""
        df = self._load_existing()
        row = pd.DataFrame([ctx.to_dict()])
        row['date'] = pd.to_datetime(row['date'])

        # Supprimer l'entrée existante pour cette date si elle existe
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] != row['date'].iloc[0]]

        df = pd.concat([df, row], ignore_index=True).sort_values('date')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.path, index=False)

    def load_last(self, n_days: int = 7) -> list[DailyContext]:
        """Retourne les n derniers contextes triés par date décroissante."""
        df = self._load_existing()
        if df.empty:
            return []
        df = df.sort_values('date', ascending=False).head(n_days)
        contexts = []
        for _, row in df.iterrows():
            try:
                contexts.append(DailyContext.from_dict(row.to_dict()))
            except Exception:
                continue
        return contexts

    def load_date(self, target_date: date) -> Optional[DailyContext]:
        """Retourne le contexte d'un jour précis ou None."""
        df = self._load_existing()
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        match = df[df['date'].dt.date == target_date]
        if match.empty:
            return None
        return DailyContext.from_dict(match.iloc[0].to_dict())

    def load_all(self) -> pd.DataFrame:
        """Retourne le DataFrame complet pour analyse."""
        return self._load_existing()


# ═══════════════════════════════════════════════════════════════
#  3. HELPERS — Saisie Rapide
# ═══════════════════════════════════════════════════════════════

def create_today_context(
    rmssd_ms: float | None = None,
    sleep_hours: float = 7.0,
    sleep_quality: int = 3,
    rpe_yesterday: float | None = None,
    stress_level: int = 2,
    notes: str = "",
    save: bool = True,
) -> DailyContext:
    """
    Crée et sauvegarde le contexte du jour courant en une seule ligne.

    Usage rapide (ex: depuis un script matin ou le dashboard Streamlit) :
        ctx = create_today_context(rmssd_ms=48, sleep_hours=7.5, sleep_quality=4)
        print(ctx.fatigue_score)  # → 0.31

    Parameters
    ----------
    rmssd_ms : HRV matinale en ms (None si non mesurée)
    sleep_hours : heures dormies
    sleep_quality : 1-5
    rpe_yesterday : RPE Borg 1-10 de la séance de la veille
    stress_level : 1-5
    notes : texte libre
    save : si True, sauvegarde dans daily_context.parquet

    Returns
    -------
    DailyContext prêt à être consommé par la Safety Valve
    """
    ctx = DailyContext(
        date=date.today(),
        rmssd_ms=rmssd_ms,
        sleep_hours=sleep_hours,
        sleep_quality=sleep_quality,
        rpe_yesterday=rpe_yesterday,
        stress_level=stress_level,
        notes=notes,
    )
    if save:
        store = DailyContextStore()
        store.save(ctx)
    return ctx
