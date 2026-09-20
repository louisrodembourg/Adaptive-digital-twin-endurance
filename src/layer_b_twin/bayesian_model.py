"""
Layer B — Digital Twin : Modèle d'État Bayésien (Banister étendu)

Implémente le modèle Fitness-Fatigue (Banister/Busso) dans un cadre
probabiliste PyMC pour quantifier l'incertitude sur l'état de l'athlète.

Architecture :
  - Fitness(t)     = Σᵢ TRIMP(i) × exp(-(t-i)/τ₁)
  - Fatigue(t)     = Σᵢ TRIMP(i) × exp(-(t-i)/τ₂)
  - Performance(t) = p₀ + k₁×Fitness(t) - k₂×Fatigue(t)
  - TSB(t)         = Fitness(t) - Fatigue(t)

Priors basés sur la littérature (Busso 2003, Clarke 2013) et calibrés
sur le record actuel de l'athlète (1h22 semi-marathon).

Outputs :
  - Courbes Fitness/Fatigue/TSB avec intervalles de confiance 95% (HDI)
  - ProbSuccess1h18 : probabilité de réussir l'objectif
  - Détection de sur-entraînement (TSB < -30) et sous-entraînement (TSB > 25)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# PyMC est importé conditionnellement (peut ne pas être installé)
try:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn(
        "PyMC non installé. Le modèle bayésien fonctionnera en mode déterministe.\n"
        "Installer avec : pip install pymc pytensor arviz",
        ImportWarning, stacklevel=2
    )


from ..config import load_params as _shared_load_params
from .load_model import compute_load_curves, exponential_load


def _load_params(config_path: str | Path | None = None) -> dict:
    """Parametres bruts — delegue au chargeur unique et mis en cache."""
    return _shared_load_params(config_path)


# ═══════════════════════════════════════════════════════════════
#  1. CALCUL DÉTERMINISTE FITNESS / FATIGUE
# ═══════════════════════════════════════════════════════════════

def compute_fitness_fatigue(
    trimp_series: np.ndarray,
    tau1: float = 42.0,
    tau2: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les courbes Fitness et Fatigue par convolution exponentielle.

Moyennes mobiles exponentielles de la charge :

      y[t] = y[t-1] × exp(-1/τ) + (1 - exp(-1/τ)) × TRIMP[t]

    Le facteur (1 - exp(-1/τ)) manquait. Sans lui la sortie est une somme
    cumulee amortie, qui croit avec le volume au lieu de se stabiliser : elle
    atteignait 4871 chez l'athlete du depot, d'ou une performance a 941,57 la
    ou les consommateurs attendaient 1,0.

    Parameters
    ----------
    trimp_series : array de TRIMP quotidien (une valeur par jour, 0 si repos)
    tau1 : constante de temps fitness (jours, défaut 42j)
    tau2 : constante de temps fatigue (jours, défaut 12j)

    Returns
    -------
    (fitness, fatigue) : deux arrays de même longueur que trimp_series
    """
    return (
        exponential_load(np.asarray(trimp_series, dtype=float), tau1),
        exponential_load(np.asarray(trimp_series, dtype=float), tau2),
    )


def compute_performance(
    fitness: np.ndarray,
    fatigue: np.ndarray,
    p0: float = 0.0,
    k1: float = 1.0,
    k2: float = 2.0,
    nonlinear: bool = True,
) -> np.ndarray:
    """
    Performance estimee depuis les courbes de charge.

        perf(t) = p0 + k1 × fitness(t) − k2 × fatigue(t)

    Les entrees doivent etre standardisees et p0, k1, k2 estimes, jamais lus
    dans un fichier de configuration : c'est la vraisemblance qui fixe
    l'echelle, en confrontant le modele a des index de performance mesures.

    La branche non lineaire d'origine a ete retiree. Elle multipliait le
    resultat par une exponentielle des le franchissement de seuils de
    fraicheur codes en dur, ce qui injectait trois constantes arbitraires
    dans la vraisemblance et rendait le modele non identifiable.

    Parameters
    ----------
    fitness, fatigue : arrays de même longueur
    p0 : performance de base, a charge nulle
    k1 : gain de fitness
    k2 : gain de fatigue (typiquement superieur a k1)
    nonlinear : conserve pour compatibilite d'appel, sans effet

    Returns
    -------
    np.ndarray : performance dans l'unite des observations fournies au modele
    """
    return p0 + k1 * np.asarray(fitness, dtype=float) - k2 * np.asarray(fatigue, dtype=float)


# ═══════════════════════════════════════════════════════════════
#  2. MODÈLE BAYÉSIEN PyMC
# ═══════════════════════════════════════════════════════════════

def load_kernel(days: int, obs_days: np.ndarray, tau):
    """
    Noyau de convolution exponentielle, evalue aux seuls jours d'observation.

    Remplace le `scan` recursif. Comme tau est echantillonne, la dynamique se
    reformule en produit matriciel :

        charge(t_obs) = somme sur i <= t_obs de
                        (1 - e^{-1/tau}) * e^{-(t_obs - i)/tau} * trimp(i)

    Deux raisons, et la premiere compte davantage. D'abord la correction : le
    scan d'origine calculait `fit_prev * decay + trimp_t`, sans le facteur
    (1 - e^{-1/tau}), soit exactement le defaut corrige dans load_model.py.
    Le reparer techniquement aurait reintroduit ici le bug corrige ailleurs ;
    exprimer la meme convolution supprime la duplication. Ensuite la vitesse :
    la vraisemblance n'a besoin des courbes qu'aux jours d'observation, soit
    une matrice de quelques dizaines de lignes au lieu d'un parcours de
    plusieurs milliers de pas, differentiable et sans boucle.

    `tau` peut etre un scalaire NumPy ou une variable PyTensor ; les
    operations utilisees existent dans les deux mondes.
    """
    lag = obs_days[:, None] - np.arange(days)[None, :]
    mask = (lag >= 0).astype(float)          # aucune charge future ne compte
    lag = np.where(lag >= 0, lag, 0.0)
    if HAS_PYMC and hasattr(tau, "type"):
        return (1.0 - pt.exp(-1.0 / tau)) * pt.exp(-lag / tau) * mask
    return (1.0 - np.exp(-1.0 / tau)) * np.exp(-lag / tau) * mask


def build_banister_model(
    daily_load: pd.DataFrame,
    observations: pd.DataFrame,
    params: dict | None = None,
    target: str = "perf_index",
):
    """
    Modele Banister probabiliste.

    Ce chemin n'avait jamais ete execute : il appelait `pt.scan`, symbole qui
    n'existe pas — c'est `pytensor.scan`. Quatre autres defauts l'auraient
    fait echouer des le premier echantillonnage, tous corriges ici.

    Priors. tau1 et tau2 suivaient des lois normales non bornees, donc
    admettaient des constantes de temps negatives : e^{-1/tau} devient alors
    un filtre explosif. Ils sont tronques. Et rien n'imposait tau1 > tau2,
    alors que les deux sont interchangeables dans la vraisemblance : les
    chaines auraient permute les etiquettes et r_hat serait ressorti autour
    de 2. tau1 se construit donc comme tau2 augmente d'un ecart positif.

    Echelle. k1 suivait une demi-normale d'ecart-type 1, alors que la charge
    vaut une cinquantaine de TRIMP par jour et la performance environ 1,0 :
    les gains sont de l'ordre du centieme, et l'echantillonneur aurait passe
    tout son budget a rejoindre la bonne echelle. Les entrees sont
    standardisees et les priors dimensionnes en consequence.

    Signe de k2. Il n'est pas contraint positif. L'ajustement par moindres
    carres le donne negatif sur ces donnees, ce qui signifierait qu'une
    fatigue elevee ameliore la performance. Forcer le signe masquerait le
    resultat : les seances rapides tombent pendant les blocs de charge, donc
    « fatigue elevee » est surtout un indicateur de « phase de qualite ».
    C'est un probleme d'endogeneite du moment de mesure, pas de colinearite —
    le conditionnement de la matrice vaut 5, ce qui disculpe cette derniere.
    Laisser le posterieur couvrir zero est precisement l'interet de
    l'approche bayesienne ici.
    """
    if not HAS_PYMC:
        raise ImportError("PyMC requis : pip install 'digital-twin-endurance[dev]'")

    params = params or _load_params()
    twin = params.get("twin", {})

    prepared = _align_observations(daily_load, observations, target)
    trimp = prepared["trimp"]
    obs_days = prepared["obs_days"]
    y = prepared["y"]
    n_days = len(trimp)

    with pm.Model() as model:
        tau2 = pm.TruncatedNormal(
            "tau2", mu=twin.get("tau2_prior_days", 12.0), sigma=5.0, lower=1.0, upper=60.0
        )
        # tau1 = tau2 + ecart positif : impose l'ordre et supprime la
        # permutation d'etiquettes entre chaines.
        delta = pm.HalfNormal("delta", sigma=20.0)
        tau1 = pm.Deterministic("tau1", tau2 + delta)

        k_fit = pm.Normal("k1", mu=0.0, sigma=0.1)
        k_fat = pm.Normal("k2", mu=0.0, sigma=0.1)
        p0 = pm.Normal("p0", mu=float(np.mean(y)), sigma=0.1)
        sigma = pm.HalfNormal("sigma_obs", sigma=float(np.std(y)) or 0.05)

        fitness = pm.math.dot(load_kernel(n_days, obs_days, tau1), trimp)
        fatigue = pm.math.dot(load_kernel(n_days, obs_days, tau2), trimp)

        # Standardisation par les constantes de l'historique complet, fixees
        # hors du modele pour que l'echelle des priors reste interpretable.
        f = (fitness - prepared["fitness_mean"]) / prepared["fitness_std"]
        g = (fatigue - prepared["fatigue_mean"]) / prepared["fatigue_std"]

        mu = pm.Deterministic("mu", p0 + k_fit * f - k_fat * g)
        pm.Normal("perf", mu=mu, sigma=sigma, observed=y)

    model.digital_twin_context = prepared  # sert aux diagnostics posterieurs
    return model


def _align_observations(daily_load: pd.DataFrame, observations: pd.DataFrame,
                        target: str) -> dict:
    """Aligne les observations sur le calendrier de charge et fige les echelles."""
    daily = daily_load.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.sort_values("date").reset_index(drop=True)

    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    obs = obs.dropna(subset=[target]).sort_values("date")

    index = {d: i for i, d in enumerate(daily["date"])}
    obs = obs[obs["date"].isin(index)]
    if len(obs) < 5:
        raise NotFittedError(
            f"Seulement {len(obs)} observation(s) alignee(s) sur le calendrier "
            f"de charge : il en faut au moins cinq."
        )

    trimp = pd.to_numeric(daily["trimp"], errors="coerce").fillna(0.0).to_numpy()
    curves = compute_load_curves(daily).to_frame()
    return {
        "trimp": trimp,
        "dates": daily["date"],
        "obs_days": np.array([index[d] for d in obs["date"]]),
        "obs_dates": obs["date"].reset_index(drop=True),
        "y": obs[target].to_numpy(dtype=float),
        "fitness_mean": float(curves["fitness"].mean()),
        "fitness_std": float(curves["fitness"].std() or 1.0),
        "fatigue_mean": float(curves["fatigue"].mean()),
        "fatigue_std": float(curves["fatigue"].std() or 1.0),
    }


def sample_model(model, draws: int = 1000, tune: int = 1000, chains: int = 4,
                 target_accept: float = 0.95, random_seed: int = 42,
                 progressbar: bool = False, cores: int | None = None):
    """
    Echantillonne le posterieur et renvoie l'InferenceData.

    `cores` par defaut a 1 : sur un historique reel, quatre chaines
    compilant PyTensor en parallele font tomber les processus fils avec une
    EOFError. L'echantillonnage sequentiel est plus lent mais fiable, et
    quatre chaines restent necessaires pour que r_hat ait un sens.

    target_accept a 0,95 plutot que 0,9 : la geometrie du posterieur est
    etroite du fait de la correlation entre k1 et k2, et le reglage par
    defaut laisse passer des divergences.
    """
    if not HAS_PYMC:
        raise ImportError("PyMC requis pour echantillonner.")
    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
            random_seed=random_seed, progressbar=progressbar,
            cores=1 if cores is None else cores,
            return_inferencedata=True,
        )


@dataclass(frozen=True)
class IdentifiabilityReport:
    """
    Ce que le posterieur dit de lui-meme.

    Un posterieur plat est un resultat : il indique que les donnees ne
    contraignent pas le parametre, ce qu'un ajustement ponctuel aurait
    presente comme une valeur estimee.
    """

    max_r_hat: float
    min_ess: float
    n_divergences: int
    k2_covers_zero: bool
    k1_k2_correlation: float
    tau_posterior_shrinkage: float
    summary: pd.DataFrame

    @property
    def converged(self) -> bool:
        return self.max_r_hat < 1.01 and self.n_divergences == 0

    @property
    def verdict(self) -> str:
        if not self.converged:
            return "Echantillonnage non converge : diagnostics non interpretables."
        problems = []
        if self.k2_covers_zero:
            problems.append("l'intervalle de credibilite de k2 contient zero")
        if abs(self.k1_k2_correlation) > 0.7:
            problems.append(f"k1 et k2 correles a {self.k1_k2_correlation:+.2f}")
        if self.tau_posterior_shrinkage < 0.2:
            problems.append("le posterieur des constantes de temps ne bouge pas du prior")
        if not problems:
            return "Parametres identifies."
        return "Modele mal identifie : " + ", ".join(problems) + "."


def assess_identifiability(trace, prior_tau1_sigma: float = 20.0) -> IdentifiabilityReport:
    """
    Diagnostic d'identifiabilite du posterieur.

    C'est l'usage honnete de l'inference bayesienne sur ce projet : non pas
    produire une estimation, mais montrer ce que les donnees contraignent
    reellement. Replique la critique publiee dans Scientific Reports en 2025
    sur le mauvais conditionnement du modele fitness-fatigue.
    """
    if not HAS_PYMC:
        raise ImportError("ArviZ requis pour les diagnostics.")

    summary = az.summary(trace, var_names=["tau1", "tau2", "k1", "k2", "p0", "sigma_obs"])
    post = trace.posterior

    k1 = post["k1"].values.ravel()
    k2 = post["k2"].values.ravel()
    lo, hi = np.percentile(k2, [2.5, 97.5])

    tau1_post_sd = float(post["tau1"].values.std())
    shrinkage = 1.0 - tau1_post_sd / prior_tau1_sigma

    divergences = int(trace.sample_stats["diverging"].values.sum()) if         "diverging" in trace.sample_stats else 0

    return IdentifiabilityReport(
        max_r_hat=float(summary["r_hat"].max()),
        min_ess=float(summary["ess_bulk"].min()),
        n_divergences=divergences,
        k2_covers_zero=bool(lo <= 0 <= hi),
        k1_k2_correlation=float(np.corrcoef(k1, k2)[0, 1]),
        tau_posterior_shrinkage=float(shrinkage),
        summary=summary,
    )


def posterior_predictions(trace, model) -> pd.DataFrame:
    """Moyenne posterieure et intervalle a 95 % aux jours d'observation."""
    ctx = model.digital_twin_context
    mu = trace.posterior["mu"].values.reshape(-1, len(ctx["y"]))
    return pd.DataFrame({
        "date": ctx["obs_dates"],
        "observed": ctx["y"],
        "predicted": mu.mean(axis=0),
        "lower": np.percentile(mu, 2.5, axis=0),
        "upper": np.percentile(mu, 97.5, axis=0),
    })


@dataclass(frozen=True)
class PerformanceParams:
    """
    Parametres du modele de performance, issus d'un ajustement.

    Ils ne se lisent pas dans un fichier de configuration. Les anciens
    k1_prior et k2_prior de params.yaml etaient des priors, utilises comme
    s'ils etaient des valeurs ajustees : c'est ce qui produisait une
    performance a 941,57 au lieu de 1,0. Un prior dit ce qu'on croit avant de
    regarder les donnees, pas ce qu'elles montrent.
    """

    p0: float
    k1: float
    k2: float
    fitness_mean: float
    fitness_std: float
    fatigue_mean: float
    fatigue_std: float
    tau1: float
    tau2: float
    n_observations: int
    source: str = "least_squares"

    def standardize(self, fitness: np.ndarray, fatigue: np.ndarray):
        f = (np.asarray(fitness, float) - self.fitness_mean) / (self.fitness_std or 1.0)
        g = (np.asarray(fatigue, float) - self.fatigue_mean) / (self.fatigue_std or 1.0)
        return f, g

    def predict(self, fitness: np.ndarray, fatigue: np.ndarray) -> np.ndarray:
        f, g = self.standardize(fitness, fatigue)
        return compute_performance(f, g, p0=self.p0, k1=self.k1, k2=self.k2)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "p0", "k1", "k2", "fitness_mean", "fitness_std", "fatigue_mean",
            "fatigue_std", "tau1", "tau2", "n_observations", "source")}

    @classmethod
    def from_dict(cls, d: dict) -> PerformanceParams:
        return cls(**{k: d[k] for k in (
            "p0", "k1", "k2", "fitness_mean", "fitness_std", "fatigue_mean",
            "fatigue_std", "tau1", "tau2", "n_observations")},
            source=d.get("source", "least_squares"))


class NotFittedError(RuntimeError):
    """Le modele n'a pas encore ete ajuste sur des performances observees."""


def fit_least_squares(
    daily_load: pd.DataFrame,
    observations: pd.DataFrame,
    tau1: float = 42.0,
    tau2: float = 12.0,
) -> PerformanceParams:
    """
    Ajuste p0, k1 et k2 par moindres carres sur les index observes.

    Sert de point de reference rapide et de repli explicite : l'echelle vient
    des observations, jamais d'une constante de configuration.
    """
    curves = compute_load_curves(daily_load, tau1, tau2)
    frame = curves.to_frame()

    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    merged = obs.merge(frame, on="date", how="inner").dropna(
        subset=["perf_index", "fitness", "fatigue"])
    if len(merged) < 5:
        raise NotFittedError(
            f"Seulement {len(merged)} observation(s) alignee(s) sur les courbes "
            f"de charge : il en faut au moins cinq pour ajuster le modele."
        )

    f_mean, f_std = float(frame["fitness"].mean()), float(frame["fitness"].std() or 1.0)
    g_mean, g_std = float(frame["fatigue"].mean()), float(frame["fatigue"].std() or 1.0)
    f = (merged["fitness"].to_numpy() - f_mean) / f_std
    g = (merged["fatigue"].to_numpy() - g_mean) / g_std

    design = np.column_stack([np.ones(len(f)), f, -g])
    coeffs, *_ = np.linalg.lstsq(design, merged["perf_index"].to_numpy(), rcond=None)
    p0, k1, k2 = (float(c) for c in coeffs)

    return PerformanceParams(
        p0=p0, k1=k1, k2=k2,
        fitness_mean=f_mean, fitness_std=f_std,
        fatigue_mean=g_mean, fatigue_std=g_std,
        tau1=tau1, tau2=tau2, n_observations=len(merged),
    )


def run_deterministic(
    daily_load: pd.DataFrame,
    fitted: PerformanceParams,
) -> pd.DataFrame:
    """
    Courbes de charge et performance predite, sans quantification d'incertitude.

    Exige un ajustement prealable et refuse de tourner sans. La version
    precedente prenait les priors du YAML pour des valeurs ajustees et
    produisait des nombres sans unite ni sens ; un repli silencieux sur des
    constantes est pire qu'une erreur, parce qu'il remplit un tableau de bord.
    """
    if fitted is None:
        raise NotFittedError(
            "Aucun ajustement disponible. Lancer `python -m src.cli fit` pour "
            "estimer p0, k1 et k2 sur des performances observees."
        )

    curves = compute_load_curves(daily_load, fitted.tau1, fitted.tau2)
    out = curves.to_frame()
    out["perf_index"] = fitted.predict(curves.fitness, curves.fatigue)
    return out
