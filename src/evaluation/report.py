"""
Rapport d'evaluation : confronter chaque variante aux memes references.

Le tableau produit place cote a cote le score d'ajustement et le score hors
echantillon. C'est l'ecart entre les deux colonnes qui informe : un modele
dont le R2 d'ajustement est honorable et le R2 hors echantillon negatif n'a
rien appris, il a memorise.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from ..layer_b_twin.load_model import (
    LOAD_COLUMNS,
    compute_load_curves,
    compute_load_curves_by_sport,
)
from .baselines import default_baselines
from .foundation import (
    ForecastContext,
    Timing,
    build_forecast_context,
    build_foundation_variants,
)
from .metrics import paired_block_bootstrap
from .validation import (
    Predictor,
    ValidationResult,
    compare_results,
    horizon_splits,
)


def banister_predictor(daily_load: pd.DataFrame, tau1: float = 42.0,
                       tau2: float = 12.0, with_fatigue: bool = True,
                       load_columns: tuple[str, ...] | None = None,
                       with_trend: bool = False) -> Predictor:
    """
    Fabrique une variante Banister ajustable par moindres carres.

    Les courbes de charge se calculent une fois sur tout l'historique — ce
    n'est pas une fuite : elles ne dependent que de l'entrainement passe, pas
    des performances a predire. Seuls p0, k1 et k2 sont ajustes par pli.

    `load_columns` donne UN COUPLE (k1, k2) PAR DISCIPLINE au lieu d'un seul
    pour toute la charge. C'est l'experience qui compte ici : la cible est un
    rapport de vitesses en course a pied, et les deux tiers du TRIMP viennent
    du velo. Un coefficient unique doit valoir a la fois pour un kilojoule de
    velo et pour un kilometre de course — il moyenne un transfert partiel et
    un transfert direct, et sort zero. Les constantes de temps restent
    partagees : separer aussi les delais ajouterait des parametres sur 90
    observations sans repondre a la question posee.

    `with_trend` ajoute une tendance temporelle lineaire en covariable
    CONCURRENTE. Elle est necessaire pour instruire un confondant precis :
    l'athlete du depot passe du velo a la course en 2018, et toute la fenetre
    d'evaluation est une transition de sport. Une progression due a
    l'apprentissage moteur ressemble exactement a une progression due a la
    charge. Si la tendance absorbe les coefficients de charge, c'est
    l'apprentissage qu'on mesurait.
    """
    if load_columns:
        curves = compute_load_curves_by_sport(daily_load, tau1, tau2, tuple(load_columns))
        pairs = [(f"fitness_{c.removeprefix('trimp_')}",
                  f"fatigue_{c.removeprefix('trimp_')}") for c in load_columns]
    else:
        curves = compute_load_curves(daily_load, tau1, tau2).to_frame()
        pairs = [("fitness", "fatigue")]

    # Une famille absente chez cet athlete donne une colonne identiquement
    # nulle : elle n'apporte pas une colonne de regression, elle la casse
    # (matrice singuliere, et un coefficient sans signification a rapporter).
    pairs = [(f, g) for f, g in pairs if (curves[f].std() or 0.0) > 0]
    if not pairs:
        raise ValueError(
            "Aucune colonne de charge exploitable : toutes les disciplines "
            "demandees sont absentes ou de charge nulle."
        )
    stats = {c: (curves[c].mean(), curves[c].std() or 1.0)
             for pair in pairs for c in pair}
    idx = curves.set_index("date")
    origin = idx.index.min()

    def design(dates: pd.Series) -> np.ndarray:
        rows = idx.reindex(pd.to_datetime(pd.Series(dates)).dt.normalize())
        cols = [np.ones(len(rows))]
        for fit_col, fat_col in pairs:
            mean, std = stats[fit_col]
            cols.append(((rows[fit_col] - mean) / std).to_numpy())
            if with_fatigue:
                mean, std = stats[fat_col]
                cols.append(-((rows[fat_col] - mean) / std).to_numpy())
        if with_trend:
            days = (rows.index - origin).days.to_numpy(dtype=float)
            cols.append((days - days.mean()) / (days.std() or 1.0))
        return np.column_stack(cols)

    def predict(dates_tr, y_tr, dates_te):
        X, y = design(dates_tr), np.asarray(y_tr, dtype=float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < X.shape[1] + 2:
            return np.full(len(dates_te), np.nan)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        return design(dates_te) @ beta

    return predict


def _run_baseline(baseline, dates_tr, y_tr, dates_te) -> np.ndarray:
    return baseline.fit_predict(dates_tr, y_tr, dates_te)


def build_variants(daily_load: pd.DataFrame,
                   taus: list[tuple[float, float]] | None = None,
                   context: ForecastContext | None = None,
                   models: tuple[str, ...] = (),
                   device: str = "cpu",
                   timings: dict[str, Timing] | None = None) -> dict[str, Predictor]:
    """
    Assemble references et variantes de modele sous une interface commune.

    Les references viennent en premier et ne sont jamais retirees : c'est ce
    qui empeche de presenter comme un succes un modele qui perd contre
    « predire la moyenne ».

    `context` ouvre les variantes a modele de fondation. Sans lui le tableau
    est exactement celui d'avant, au chiffre pres : les modeles de fondation
    ont besoin d'une grille journaliere que l'enveloppe rapportee aux seules
    dates d'observation ne fournit pas.
    """
    def _as_predictor(baseline) -> Predictor:
        # functools.partial lie la reference explicitement ; une lambda
        # capturerait la variable de boucle, donc la derniere reference.
        return functools.partial(_run_baseline, baseline)

    variants: dict[str, Predictor] = {
        b.name: _as_predictor(b) for b in default_baselines()
    }
    has_sports = all(c in daily_load.columns for c in LOAD_COLUMNS)
    for tau1, tau2 in taus or [(42.0, 12.0)]:
        suffix = f"t{tau1:.0f}/{tau2:.0f}"
        variants[f"banister {suffix}"] = banister_predictor(daily_load, tau1, tau2)
        variants[f"fitness seul t{tau1:.0f}"] = banister_predictor(
            daily_load, tau1, tau2, with_fatigue=False)
        if not has_sports:
            continue
        # Les variantes qui instruisent le resultat nul : un coefficient par
        # discipline, chaque charge specifique seule, et la charge mise en
        # concurrence avec une simple tendance temporelle.
        #
        # Une discipline que cet athlete ne pratique pas donne une colonne
        # nulle, et `banister_predictor` refuse de l'ajuster. C'est le cas
        # courant des le moment ou l'on sort de l'athlete de reference : un
        # cycliste pur n'a aucune charge de course. La variante est alors
        # omise plutot que de faire echouer tout le tableau.
        specific = {
            f"banister par sport {suffix}": LOAD_COLUMNS,
            f"banister course seule {suffix}": ("trimp_run",),
            # La charge velo contre une cible velo : le seul appariement ou la
            # discipline qui porte la charge est celle dont on mesure la
            # performance.
            f"banister velo seul {suffix}": ("trimp_bike",),
        }
        for name, columns in specific.items():
            try:
                variants[name] = banister_predictor(
                    daily_load, tau1, tau2, load_columns=columns)
            except ValueError:
                continue
        variants[f"banister + tendance {suffix}"] = banister_predictor(
            daily_load, tau1, tau2, with_trend=True)
    if context is not None and models:
        variants.update(build_foundation_variants(context, models, device, timings))
    return variants


def evaluate_results(observations: pd.DataFrame, daily_load: pd.DataFrame,
             target: str = "perf_index", n_splits: int = 5,
             taus: list[tuple[float, float]] | None = None,
             context: ForecastContext | None = None,
             models: tuple[str, ...] = (),
             splits: list | None = None,
             device: str = "cpu",
             timings: dict[str, Timing] | None = None) -> pd.DataFrame:
    """
    Passe toutes les variantes sous le meme protocole.

    `splits` impose un decoupage construit a l'avance — `horizon_splits` pour
    le protocole a horizon calendaire. Il est passe tel quel a toutes les
    variantes : c'est la seule facon de garder les lignes comparables.
    """
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    obs = obs.sort_values("date").dropna(subset=[target]).reset_index(drop=True)
    variants = build_variants(daily_load, taus, context, models, device, timings)
    return compare_results(variants, obs["date"], obs[target].to_numpy(),
                           n_splits=n_splits, splits=splits)


def results_table(results: dict[str, ValidationResult]) -> pd.DataFrame:
    """Range les resultats en tableau, classe sur le score hors echantillon."""
    rows = [r.to_dict() for r in results.values()]
    return pd.DataFrame(rows).sort_values("r2_out", ascending=False).reset_index(drop=True)


def evaluate(observations: pd.DataFrame, daily_load: pd.DataFrame,
             target: str = "perf_index", n_splits: int = 5,
             taus: list[tuple[float, float]] | None = None,
             context: ForecastContext | None = None,
             models: tuple[str, ...] = (),
             splits: list | None = None,
             device: str = "cpu",
             timings: dict[str, Timing] | None = None) -> pd.DataFrame:
    """Le tableau seul, pour les appelants qui n'ont pas besoin des plis."""
    return results_table(evaluate_results(
        observations, daily_load, target, n_splits, taus, context, models,
        splits, device, timings))


def paired_table(results: dict[str, ValidationResult],
                 pairs: list[tuple[str, str]], seed: int = 0) -> pd.DataFrame:
    """
    Compare des variantes deux a deux sur les memes points.

    Soustraire deux R2 agreges ne dit pas si l'ecart depasse le bruit. Ici
    chaque paire passe par un bootstrap apparie par blocs, et l'intervalle
    rendu est ce qui autorise — ou interdit — d'affirmer un gain.
    """
    rows = []
    for a, b in pairs:
        ra, rb = results.get(a), results.get(b)
        if ra is None or rb is None or ra.y_true.size == 0 or rb.y_true.size == 0:
            continue
        if ra.y_true.shape != rb.y_true.shape:
            raise ValueError(
                f"{a} et {b} n'ont pas ete evalues sur les memes points "
                "— la comparaison appariee n'a pas de sens."
            )
        stats = paired_block_bootstrap(ra.y_true, ra.y_pred, rb.y_pred, ra.blocks, seed=seed)
        rows.append({"variante": a, "reference": b, "delta_r2": ra.r2_out_of_sample
                     - rb.r2_out_of_sample, **stats})
    return pd.DataFrame(rows)


def format_pairs(table: pd.DataFrame) -> str:
    """Rend les comparaisons appariees lisibles en terminal."""
    if table.empty:
        return "Aucune comparaison appariee."
    # Le protocole par duree compare la MEME paire a plusieurs durees : sans
    # cette colonne les lignes sont indistinguables.
    by_duration = "duration_s" in table.columns
    head = f"{'duree':>7s} " if by_duration else ""
    lines = [
        head + f"{'variante':28s} {'contre':20s} {'d R2':>8s} {'gain':>8s} "
        f"{'IC 95 %':>20s} {'P(gain)':>8s}",
        "-" * (96 + len(head)),
    ]
    for _, r in table.iterrows():
        interval = f"[{r['ci_low']:+.4f} · {r['ci_high']:+.4f}]"
        prefix = f"{int(r['duration_s']):6d}s " if by_duration else ""
        lines.append(
            prefix + f"{r['variante']:28s} {r['reference']:20s} {r['delta_r2']:+8.4f} "
            f"{r['skill']:+8.4f} {interval:>20s} {r['p_a_better']:8.3f}"
        )
    return "\n".join(lines)


def format_report(table: pd.DataFrame) -> str:
    """Rend le tableau lisible en terminal, trie par score hors echantillon."""
    lines = [
        f"{'variante':30s} {'R2 ajust.':>10s} {'R2 h.-ech.':>11s} "
        f"{'ecart':>8s} {'RMSE':>8s} {'gain/moy.':>10s} {'plis':>5s}",
        "-" * 88,
    ]
    for _, r in table.iterrows():
        # Le nombre de plis est affiche parce qu'il n'est pas toujours le meme :
        # un modele dont la tete est entrainee sur un horizon court echoue sur
        # les plis les plus longs, et sa ligne porte alors sur moins de points
        # que les autres. Sans cette colonne la comparaison paraitrait a armes
        # egales.
        lines.append(
            f"{r['name']:30s} {r['r2_in']:+10.4f} {r['r2_out']:+11.4f} "
            f"{r['gap']:+8.4f} {r['rmse_out']:8.5f} {r['skill_vs_mean']:+10.4f} "
            f"{int(r['folds']):5d}"
        )
    winners = table[table["r2_out"] > 0]
    lines.append("")
    if winners.empty:
        lines.append("Aucune variante ne fait mieux que predire la moyenne.")
    else:
        lines.append(f"Meilleure variante hors echantillon : {winners.iloc[0]['name']}")
    return "\n".join(lines)

def load_ablation_pairs(tau1: float = 42.0, tau2: float = 12.0) -> list[tuple[str, str]]:
    """
    Les comparaisons qui instruisent la decomposition de la charge.

    Chacune repond a une question precise, et aucune ne se lit sans son
    intervalle : sur 85 points d'enveloppe autocorreles a 0,97, un ecart de
    R2 agrege de quelques centiemes est parfaitement compatible avec le
    hasard.

    - course seule CONTRE charge totale : la charge melangee nuisait-elle ?
      C'est le test direct du diagnostic (68,5 % du TRIMP est du velo, la
      cible ne mesure que la course).
    - course seule CONTRE moyenne : la charge specifique porte-t-elle quelque
      chose, ou vaut-il mieux predire la moyenne ?
    - course seule CONTRE persistance : la seule reference qui compte pour
      conclure.
    - par sport CONTRE charge totale : un couple de coefficients par
      discipline aide-t-il, ou n'ajoute-t-il que des parametres ?
    - plus tendance CONTRE charge totale : une tendance temporelle absorbe-t-
      elle la charge ? Si oui, c'est l'apprentissage moteur qu'on mesurait.
    """
    suffix = f"t{tau1:.0f}/{tau2:.0f}"
    run_only, total = f"banister course seule {suffix}", f"banister {suffix}"
    return [
        (run_only, total),
        (run_only, "moyenne"),
        (run_only, "persistance"),
        (f"banister par sport {suffix}", total),
        (f"banister + tendance {suffix}", total),
    ]

# ── protocole par duree d'effort ──────────────────────────────────────────

#: Durees retenues par `performance.DEFAULT_DURATIONS_S`, en secondes.
DURATIONS_S: tuple[int, ...] = (300, 600, 1200, 1800, 2700, 3600)

#: Variantes rapportees par duree. Volontairement court : la question posee
#: ici est « la dose-reponse depend-elle de la duree de l'effort ? », pas
#: « quelle variante gagne ». Chaque duree ajoutee multiplie les plis.
DURATION_VARIANTS: tuple[str, ...] = (
    "persistance", "moyenne", "banister course seule t42/12", "banister t42/12",
)


def evaluate_by_duration(
    observations: pd.DataFrame,
    daily_load: pd.DataFrame,
    durations: tuple[int, ...] = DURATIONS_S,
    taus: list[tuple[float, float]] | None = None,
    horizon_days: int = 28,
    min_train_days: int = 180,
    min_observations: int = 40,
    window_days: int = 42,
    min_periods: int = 3,
    variants: tuple[str, ...] = DURATION_VARIANTS,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evalue une enveloppe PAR DUREE d'effort au lieu d'un scalaire unique.

    `perf_index` prend un maximum sur les durees, puis l'enveloppe reprend un
    maximum sur 42 jours : un double cliquet, d'ou l'autocorrelation a 0,97.
    Un plafond a 5 minutes depend surtout de la puissance anaerobie, un
    plafond a 45 minutes de l'endurance aerobie ; les deux ne repondent ni au
    meme entrainement ni sur la meme echelle de temps. Les ecraser en un
    scalaire ne peut que diluer une dose-reponse qui existerait sur une seule
    des deux.

    Une duree comptant moins de `min_observations` mesures est ECARTEE et
    rapportee comme telle : sur ce jeu de donnees la duree 3600 s n'a que
    neuf observations, et un R2 sur nuit points ne veut rien dire.

    Renvoie (tableau long, comparaisons appariees contre la persistance).
    """
    taus = taus or [(42.0, 12.0)]
    rows, pairs = [], []
    for duration in durations:
        column = f"perf_index_{int(duration)}"
        if column not in observations.columns:
            continue
        source = (observations[["date", column]].dropna()
                  .rename(columns={column: "pi"}))
        if len(source) < min_observations:
            rows.append({"duration_s": int(duration), "n_obs": len(source),
                         "n_days": 0, "variante": "ecartee",
                         "r2_out": float("nan"), "folds": 0})
            continue

        context = build_forecast_context(source, daily_load, "pi",
                                         window_days=window_days,
                                         min_periods=min_periods)
        dense = (context.frame[[context.target_column]].dropna()
                 .rename(columns={context.target_column: "perf"})
                 .reset_index())
        splits = list(horizon_splits(dense["date"], horizon_days=horizon_days,
                                     min_train_days=min_train_days,
                                     step_days=horizon_days))
        if not splits:
            rows.append({"duration_s": int(duration), "n_obs": len(source),
                         "n_days": len(dense), "variante": "ecartee",
                         "r2_out": float("nan"), "folds": 0})
            continue

        results = evaluate_results(dense, daily_load, target="perf", taus=taus,
                                   splits=splits)
        for name in variants:
            result = results.get(name)
            if result is None:
                continue
            rows.append({"duration_s": int(duration), "n_obs": len(source),
                         "n_days": len(dense), "variante": name,
                         "r2_out": result.r2_out_of_sample, "folds": result.n_folds})

        specific = f"banister course seule t{taus[0][0]:.0f}/{taus[0][1]:.0f}"
        pair = paired_table(results, [(specific, "persistance")], seed=seed)
        if not pair.empty:
            pairs.append({"duration_s": int(duration), **pair.iloc[0].to_dict()})

    return pd.DataFrame(rows), pd.DataFrame(pairs)


def format_durations(table: pd.DataFrame) -> str:
    """Rend le tableau par duree lisible : une ligne par duree."""
    if table.empty:
        return "Aucune duree evaluable."
    kept = table[table["variante"] != "ecartee"]
    dropped = table[table["variante"] == "ecartee"]
    lines = []
    if not kept.empty:
        wide = kept.pivot_table(index=["duration_s", "n_obs", "n_days"],
                                columns="variante", values="r2_out").reset_index()
        names = [c for c in DURATION_VARIANTS if c in wide.columns]
        header = f"{'duree':>7s} {'obs':>5s} {'jours':>6s} " + " ".join(
            f"{n[:17]:>17s}" for n in names)
        lines += [header, "-" * len(header)]
        for _, r in wide.iterrows():
            lines.append(
                f"{int(r['duration_s']):6d}s {int(r['n_obs']):5d} {int(r['n_days']):6d} "
                + " ".join(f"{r[n]:+17.4f}" for n in names)
            )
    for _, r in dropped.iterrows():
        lines.append(f"{int(r['duration_s']):6d}s ecartee — {int(r['n_obs'])} observations, "
                     "trop peu pour un R2 interpretable")
    return "\n".join(lines)

#: Colonne de puissance critique retenue comme cible velo. Vingt minutes est
#: le compromis usuel : assez long pour etre limite par l'aerobie, assez court
#: pour etre atteint souvent (417 observations chez l'athlete de reference,
#: contre 273 a soixante minutes).
BIKE_TARGET_COLUMN = "cp20_w"

#: Un watt est un watt, sur route comme sur home-trainer : les deux
#: disciplines se regroupent pour une PUISSANCE, ce qui serait faux pour une
#: vitesse.
BIKE_SPORTS: tuple[str, ...] = ("bike", "virtualride")

#: Variante specifique a employer pour chaque canal de charge. Centralise ici
#: parce que trois modules en dependent — sensibilite, cohorte et
#: planificateur — et qu'un nom qui derive fait disparaitre silencieusement
#: une comparaison du rapport.
SPECIFIC_VARIANTS: dict[str, str] = {
    "trimp_run": "banister course seule",
    "trimp_bike": "banister velo seul",
}


def specific_variant(load_column: str, tau1: float = 42.0, tau2: float = 12.0) -> str:
    """Nom de la variante qui ne consomme que cette discipline."""
    try:
        base = SPECIFIC_VARIANTS[load_column]
    except KeyError:
        raise ValueError(
            f"Aucune variante specifique pour {load_column!r}. "
            f"Connues : {', '.join(sorted(SPECIFIC_VARIANTS))}."
        ) from None
    return f"{base} t{tau1:.0f}/{tau2:.0f}"
