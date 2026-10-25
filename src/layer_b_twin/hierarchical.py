"""
Modele hierarchique : quantifier l'heterogeneite au lieu de la constater.

Ce que la cohorte a laisse ouvert
---------------------------------
Rejouer le protocole athlete par athlete a montre un effet de charge chez
quatre sujets sur treize et rien chez les neuf autres, avec des estimations
ponctuelles centrees sur zero chez ces derniers — donc une heterogeneite
reelle, et non un defaut de mesure. Mais treize ajustements separes suivis
d'un test du signe ne disent pas DE COMBIEN les athletes diffèrent : ils
comptent des verdicts, ils n'estiment pas une dispersion.

Un modele hierarchique repond a cette question-la. Les gains deviennent
`k_a = mu + sigma * z_a` : `mu` est l'effet de population, `sigma` la
dispersion entre athletes — le nombre qui manquait — et chaque `k_a` emprunte
de la force aux autres, ce qui resserre les athletes mal mesures sans effacer
ceux qui s'ecartent vraiment.

Pourquoi les constantes de temps y gagnent
------------------------------------------
tau1 et tau2 ne sont pas identifiees sur un athlete : le posterieur du depot
ne les resserre que de 41 % par rapport au prior. Elles sont pourtant
partagees ici, donc contraintes par treize historiques au lieu d'un. C'est la
seule facon honnete de les estimer avec ces donnees, et c'est l'autre raison
d'ecrire ce modele.

Trois choix a declarer
----------------------
**Cibles reduites par athlete.** Les enveloppes vont de 170 a 260 W selon le
sujet. Mettre en commun un gain n'a de sens que si la grandeur expliquee a la
meme echelle partout, donc chaque enveloppe est centree-reduite sur son
propre historique. Les gains s'expriment alors en ecarts-types d'enveloppe,
comparables d'un athlete a l'autre.

**Un point tous les quatorze jours.** L'enveloppe est un maximum glissant sur
42 jours : deux jours consecutifs sont quasi identiques, et le noyau de
convolution coute `n_observations x n_jours` par athlete. Echantillonner tous
les quatorze jours ramene le cout a l'echelle du modele mono-athlete sans
perdre de structure — c'est la meme grandeur, moins souvent lue.

**Effet net echantillonne, pas chaque gain.** Comme dans le modele
mono-athlete : k1 et k2 ressortent correles a 0,98, seul leur effet net est
contraint, et les echantillonner separement donne une geometrie en crete qui
avait coute vingt-deux heures. On garde la parametrisation qui l'a ramenee a
vingt-six minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..layer_a_signal.envelope import EnvelopeSpec, rolling_envelope
from .bayesian_model import HAS_PYMC, load_kernel

if HAS_PYMC:
    import pymc as pm
    import pytensor

    # PyTensor fusionne les expressions par athlete en une seule expression C,
    # et clang plafonne l'imbrication de parentheses a 256. Au-dela d'une
    # quarantaine d'athletes la compilation echoue sur
    # « bracket nesting level exceeded maximum of 256 », ce qui n'a rien a voir
    # avec le modele et tout avec le compilateur. Le drapeau se pose ici parce
    # que c'est le module qui cree le graphe : l'utilisateur n'a pas a
    # connaitre cette limite, et un echec de compilation apres une heure
    # d'attente est le pire moment pour l'apprendre.
    _FLAG = "-fbracket-depth=4096"
    if _FLAG not in pytensor.config.gcc__cxxflags:
        pytensor.config.gcc__cxxflags = (
            f"{pytensor.config.gcc__cxxflags} {_FLAG}".strip()
        )

#: Un point d'enveloppe tous les quatorze jours. Voir la docstring du module :
#: l'enveloppe est un maximum glissant sur 42 jours, donc tres redondante au
#: pas journalier, et le noyau coute n_observations x n_jours.
DEFAULT_STRIDE_DAYS = 14

#: Observations exigees d'un athlete pour qu'il entre dans le modele. En
#: dessous, sa ligne n'apprend rien et ajoute trois parametres.
MIN_POINTS = 12


@dataclass(frozen=True)
class AthleteBlock:
    """Un athlete, reduit a ce dont la vraisemblance a besoin."""

    uuid: str
    trimp: np.ndarray            # charge quotidienne, calendrier sans trou
    obs_days: np.ndarray         # indices des jours lus, dans ce calendrier
    y: np.ndarray                # enveloppe centree-reduite a ces jours
    trend: np.ndarray = field(default_factory=lambda: np.array([]))
    fitness_stats: tuple[float, float] = (0.0, 1.0)
    fatigue_stats: tuple[float, float] = (0.0, 1.0)

    @property
    def n_days(self) -> int:
        return len(self.trimp)

    @property
    def n_points(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class CohortData:
    """La cohorte prete a etre ajustee."""

    blocks: tuple[AthleteBlock, ...] = field(default_factory=tuple)
    stride_days: int = DEFAULT_STRIDE_DAYS

    @property
    def uuids(self) -> tuple[str, ...]:
        return tuple(b.uuid for b in self.blocks)

    @property
    def n_athletes(self) -> int:
        return len(self.blocks)

    @property
    def n_points(self) -> int:
        return sum(b.n_points for b in self.blocks)

    def target_smoothness(self) -> tuple[float, float]:
        """
        Regularite de la cible : autocorrelation de rang 1 et pas median.

        La fenetre du maximum glissant ne fait pas que lisser : elle fixe
        combien de la variance de la cible est explicable par quoi que ce
        soit de regulier. Comparer deux ajustements a fenetres differentes
        sans cette mesure, c'est attribuer a la physiologie un deplacement
        des parametres que la cible explique seule. La cible etant
        centree-reduite par athlete, le pas s'exprime en ecarts-types et les
        athletes sont comparables.

        Rendue en MEDIANE sur les athletes, jamais en moyenne : les
        historiques sont tres inegaux.
        """
        autocorr, step = [], []
        for b in self.blocks:
            y = np.asarray(b.y, dtype=float)
            if len(y) < 3:
                continue
            step.append(float(np.median(np.abs(np.diff(y)))))
            a, c = y[:-1], y[1:]
            if np.std(a) > 0 and np.std(c) > 0:
                autocorr.append(float(np.corrcoef(a, c)[0, 1]))
        if not step:
            return (float("nan"), float("nan"))
        return (float(np.median(autocorr)) if autocorr else float("nan"),
                float(np.median(step)))

    def describe(self) -> str:
        auto, step = self.target_smoothness()
        lines = [f"{self.n_athletes} athlete(s), {self.n_points} point(s) "
                 f"d'enveloppe, un tous les {self.stride_days} jours",
                 f"Cible : autocorrelation de rang 1 mediane {auto:+.3f}, "
                 f"pas median {step:.3f} sigma", ""]
        for b in self.blocks:
            lines.append(f"  {b.uuid[:8]}  {b.n_days:5d} jours  {b.n_points:4d} points")
        return "\n".join(lines)


def prepare_athlete(
    uuid: str,
    daily_load: pd.DataFrame,
    observations: pd.DataFrame,
    load_column: str = "trimp_bike",
    value_column: str = "pi",
    window_days: int = 42,
    min_periods: int = 3,
    stride_days: int = DEFAULT_STRIDE_DAYS,
    tau1: float = 42.0,
    tau2: float = 12.0,
) -> AthleteBlock | None:
    """
    Reduit un athlete a son bloc, ou rend None s'il n'apprend rien.

    Les statistiques de standardisation des courbes sont calculees ici, aux
    constantes de temps par defaut, et figees : c'est ce que fait le modele
    mono-athlete, et cela garde l'echelle des priors interpretable meme quand
    tau est echantillonne.
    """
    if daily_load.empty or load_column not in daily_load.columns:
        return None
    if observations.empty or value_column not in observations.columns:
        return None

    daily = daily_load.sort_values("date").reset_index(drop=True)
    calendar = pd.DatetimeIndex(pd.to_datetime(daily["date"]))
    trimp = pd.to_numeric(daily[load_column], errors="coerce").fillna(0.0).to_numpy()

    source = (observations[["date", value_column]].dropna()
              .rename(columns={value_column: "pi"}))
    spec = EnvelopeSpec("pi", window_days, min_periods=min_periods)
    envelope = rolling_envelope(source, spec, calendar="daily")
    if envelope.empty:
        return None

    # Un point tous les `stride_days`, voir la docstring du module.
    envelope = envelope.iloc[::stride_days]
    dates = pd.DatetimeIndex(pd.to_datetime(envelope["date"]))
    positions = calendar.get_indexer(dates)
    keep = positions >= 0                     # une mesure hors calendrier de charge
    # `spec.label` plutot que le nom en dur : l'etiquette suit `window_days`.
    positions = positions[keep]
    values = envelope.iloc[keep][spec.label].to_numpy()
    if len(values) < MIN_POINTS:
        return None

    # Cible reduite : mettre en commun un gain exige une grandeur de meme
    # echelle chez tous. Les gains s'expriment alors en ecarts-types.
    std = float(np.std(values)) or 1.0
    y = (values - float(np.mean(values))) / std

    from .load_model import exponential_load
    fitness = exponential_load(trimp, tau1)[positions]
    fatigue = exponential_load(trimp, tau2)[positions]
    # Tendance temporelle reduite, pour mettre la charge en concurrence avec
    # le simple passage du temps. Sans ce terme, une constante de fitness qui
    # fuit vers des centaines de jours transforme la charge cumulee en
    # tendance, et le gain mesure l'une en croyant mesurer l'autre.
    days_elapsed = positions.astype(float)
    trend = (days_elapsed - days_elapsed.mean()) / (days_elapsed.std() or 1.0)

    return AthleteBlock(
        uuid=uuid,
        trimp=trimp,
        obs_days=positions.astype(int),
        y=y,
        trend=trend,
        fitness_stats=(float(fitness.mean()), float(fitness.std()) or 1.0),
        fatigue_stats=(float(fatigue.mean()), float(fatigue.std()) or 1.0),
    )


def build_hierarchical_model(data: CohortData, tau2_prior_days: float = 12.0):
    """
    Un modele, un couple de constantes de temps, un gain par athlete.

    Parametrisation non centree pour les gains — `k_a = mu + sigma * z_a` avec
    `z_a` standard. La forme centree place un entonnoir entre `sigma` et les
    `k_a` que NUTS traverse mal : c'est le defaut classique des modeles
    hierarchiques, et il se corrige par la parametrisation, pas par plus
    d'iterations.
    """
    if not HAS_PYMC:
        raise ImportError("PyMC absent. `pip install -e \".[dev]\"`.")
    if data.n_athletes == 0:
        raise ValueError("Aucun athlete a ajuster.")

    coords = {"athlete": list(data.uuids)}
    with pm.Model(coords=coords) as model:
        # Constantes de temps PARTAGEES : c'est le partage qui les rend
        # identifiables, la ou un athlete seul ne resserre son prior que de
        # 41 %.
        tau2 = pm.TruncatedNormal("tau2", mu=tau2_prior_days, sigma=5.0,
                                  lower=1.0, upper=60.0)
        delta = pm.HalfNormal("delta", sigma=20.0)
        tau1 = pm.Deterministic("tau1", tau2 + delta)

        # Effet net au niveau population, et sa dispersion entre athletes.
        # `sigma_k_net` est le nombre que la cohorte ne pouvait pas estimer.
        mu_k_net = pm.Normal("mu_k_net", mu=0.0, sigma=0.5)
        sigma_k_net = pm.HalfNormal("sigma_k_net", sigma=0.5)
        z_net = pm.Normal("z_net", mu=0.0, sigma=1.0, dims="athlete")
        k_net = pm.Deterministic("k_net", mu_k_net + sigma_k_net * z_net,
                                 dims="athlete")

        mu_k_fat = pm.Normal("mu_k_fat", mu=0.0, sigma=0.5)
        sigma_k_fat = pm.HalfNormal("sigma_k_fat", sigma=0.5)
        z_fat = pm.Normal("z_fat", mu=0.0, sigma=1.0, dims="athlete")
        k_fat = pm.Deterministic("k2", mu_k_fat + sigma_k_fat * z_fat,
                                 dims="athlete")
        k_fit = pm.Deterministic("k1", k_net + k_fat, dims="athlete")

        # Les cibles etant reduites, l'intercept est proche de zero et la
        # dispersion residuelle proche de un : les priors le disent.
        p0 = pm.Normal("p0", mu=0.0, sigma=0.5, dims="athlete")
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0, dims="athlete")

        # Tendance concurrente. Elle n'est pas la pour ameliorer l'ajustement
        # mais pour le disputer : si le gain de charge s'effondre quand elle
        # entre, c'est le passage du temps qu'on mesurait.
        mu_trend = pm.Normal("mu_trend", mu=0.0, sigma=0.5)
        sigma_trend = pm.HalfNormal("sigma_trend", sigma=0.5)
        z_trend = pm.Normal("z_trend", mu=0.0, sigma=1.0, dims="athlete")
        k_trend = pm.Deterministic("k_trend", mu_trend + sigma_trend * z_trend,
                                   dims="athlete")

        for i, block in enumerate(data.blocks):
            fitness = pm.math.dot(load_kernel(block.n_days, block.obs_days, tau1),
                                  block.trimp)
            fatigue = pm.math.dot(load_kernel(block.n_days, block.obs_days, tau2),
                                  block.trimp)
            f = (fitness - block.fitness_stats[0]) / block.fitness_stats[1]
            g = (fatigue - block.fatigue_stats[0]) / block.fatigue_stats[1]
            mu = p0[i] + k_fit[i] * f - k_fat[i] * g + k_trend[i] * block.trend
            pm.Normal(f"perf_{i}", mu=mu, sigma=sigma_obs[i], observed=block.y)

    model.cohort_data = data
    return model


def summarise_pooling(trace, data: CohortData) -> pd.DataFrame:
    """
    Les quantites qui repondent a la question posee.

    `sigma_k_net` d'abord : c'est la dispersion de l'effet net entre athletes,
    donc la mesure de l'heterogeneite que la cohorte ne pouvait que constater.
    Un intervalle qui exclut zero dit que les athletes diffèrent reellement ;
    un intervalle qui le contient dit que treize sujets ne suffisent pas a le
    montrer, ce qui est un resultat different de « ils sont identiques ».
    """
    import arviz as az

    names = ["tau1", "tau2", "mu_k_net", "sigma_k_net", "mu_k_fat", "sigma_k_fat",
             "mu_trend", "sigma_trend"]
    summary = az.summary(trace, var_names=names, hdi_prob=0.94)
    summary.insert(0, "parametre", summary.index)
    return summary.reset_index(drop=True)


def per_athlete_effects(trace, data: CohortData) -> pd.DataFrame:
    """Effet net par athlete, apres emprunt de force aux autres."""
    import arviz as az

    posterior = trace.posterior["k_net"]

    # La trace et la cohorte peuvent avoir divergé : le cache grandit des
    # qu'on telecharge des archives, et rejouer l'analyse sur une trace plus
    # ancienne donnait un KeyError d'xarray illisible. Le dire, et nommer le
    # coupable.
    sampled = set(map(str, posterior.coords["athlete"].values))
    unknown = [u for u in data.uuids if u not in sampled]
    if unknown:
        raise ValueError(
            f"{len(unknown)} athlete(s) de la cohorte absent(s) de la trace "
            f"(dont {unknown[0][:8]}). La cohorte a change depuis "
            "l'echantillonnage — relancer `cli pool`."
        )

    rows = []
    for uuid in data.uuids:
        draws = posterior.sel(athlete=uuid).values.ravel()
        low, high = az.hdi(draws, hdi_prob=0.94)
        rows.append({
            "athlete": uuid[:8],
            "k_net_moyen": float(draws.mean()),
            "hdi_bas": float(low),
            "hdi_haut": float(high),
            "exclut_zero": bool(low > 0 or high < 0),
        })
    return pd.DataFrame(rows).sort_values("k_net_moyen", ascending=False)
