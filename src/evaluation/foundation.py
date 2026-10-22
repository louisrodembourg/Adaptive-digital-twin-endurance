"""
Modeles de fondation de series temporelles comme references supplementaires.

Le depot a etabli que le modele de Banister ne bat pas la persistance sur ces
donnees, et que le posterieur place l'effet net de la charge a zero. Ce
resultat repose sur UN modele parametrique, ce qui laisse une objection
ouverte : peut-etre que la charge predit la performance, mais pas sous la
forme d'une convolution exponentielle a deux constantes de temps.

Ce module teste cette objection sans supposer de forme fonctionnelle. Un
modele de fondation qui accepte des covariables recoit l'enveloppe comme
cible et la charge comme covariable, puis on le compare au MEME modele prive
de la covariable. L'ecart entre les deux mesure l'information que la charge
porte sur la performance — c'est une ablation, pas un ajustement.

Un seul des trois modeles cables ici sait faire cela en zero-shot, et c'est
un resultat de la recherche plutot qu'un detail d'implementation :

- **Chronos-2** (120 M, Apache-2.0) accepte covariables passees et futures
  connues. C'est le seul qui tourne sans compte ni fine-tuning.
- **TabPFN-TS** (11 M) les accepte aussi, mais `tabpfn` 9.x exige un compte
  Prior Labs et une acceptation de licence interactive pour telecharger ses
  poids. Sans `TABPFN_TOKEN`, la variante ne produit aucun chiffre.
- **TTM** (855 k, Apache-2.0) annonce l'integration de variables exogenes,
  mais son dorsal est independant par canal en zero-shot : remplacer
  entierement les covariables laisse la prevision inchangee au bit pres. Son
  exogene demande un fine-tuning. Il n'est donc enregistre qu'en univarie.

Trois choix a ne pas laisser implicites.

**Les covariables incluent fitness et fatigue, pas seulement le TRIMP brut.**
Un modele pre-entraine sur des series generiques n'a aucune raison de
decouvrir un noyau exponentiel de 42 jours a partir de 500 points. Lui
fournir les regresseurs de Banister, calcules par la source de verite unique
du depot (`compute_load_curves`), empeche d'attribuer un resultat nul a une
mise en forme insuffisante. Le modele recoit exactement l'information de
Banister, libre de la combiner autrement.

**La charge future est fournie au modele.** Chronos-2 sait exploiter des
covariables futures connues, et nous connaissons la charge reellement
effectuee sur l'horizon. Ce n'est pas la question d'un planificateur — qui ne
connaitrait pas l'avenir — mais c'est exactement la bonne question ici : la
charge porte-t-elle de l'information sur la performance ? Lui refuser cette
information rendrait un resultat nul inexploitable.

**Le R2 d'ajustement n'existe pas pour ces modeles.** Ils n'ajustent rien :
les poids sont figes, l'inference est zero-shot. `cross_validate` demande
pourtant une prediction sur les dates d'entrainement pour calculer cette
colonne. Elle renvoie NaN ici, et c'est un resultat plutot qu'une lacune : un
ecart ajustement / hors echantillon vide signifie qu'il n'y a aucun
surapprentissage a mesurer, a l'exact inverse de Banister dont le meilleur
ajustement accompagnait le pire score hors echantillon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from ..layer_a_signal.envelope import EnvelopeSpec, rolling_envelope
from ..layer_b_twin.load_model import compute_load_curves
from .validation import Predictor

# Limites de Chronos-2 : 8192 pas de contexte, 1024 pas d'horizon. Nos series
# journalieres tiennent largement dessous, mais un horizon demande au-dela
# doit echouer franchement plutot que d'etre tronque en silence.
MAX_CONTEXT = 8192
MAX_HORIZON = 1024

COVARIATE_COLUMNS = ("trimp", "fitness", "fatigue")


@dataclass
class ForecastContext:
    """
    Cible et covariables sur une grille journaliere sans trou.

    L'enveloppe telle que le depot la rapportait n'existe qu'aux jours ou une
    mesure existe : 85 points espaces de 1 a 140 jours. Presque tous les
    modeles de fondation supposent un pas constant, et leur donner une serie
    irreguliere en pretendant qu'elle est reguliere deformerait les
    constantes de temps qu'ils en deduisent.

    La grille journaliere porte donc `NaN` la ou l'enveloppe n'est pas
    definie — pendant l'interruption de 135 jours de cet athlete, notamment.
    `NaN` veut dire « non defini », jamais « valeur reportee » : combler ces
    jours par la derniere valeur connue fabriquerait de la donnee et, pire,
    rendrait la cible artificiellement facile a prevoir par persistance.
    """

    frame: pd.DataFrame
    target_column: str
    covariate_columns: tuple[str, ...] = COVARIATE_COLUMNS

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame.index)

    def describe(self) -> dict:
        target = self.frame[self.target_column]
        return {
            "n_days": len(self.frame),
            "n_defined": int(target.notna().sum()),
            "start": str(self.dates.min().date()),
            "end": str(self.dates.max().date()),
            "covariates": list(self.covariate_columns),
        }


def build_forecast_context(
    observations: pd.DataFrame,
    daily_load: pd.DataFrame,
    column: str,
    window_days: int = 42,
    min_periods: int = 3,
    tau1: float = 42.0,
    tau2: float = 12.0,
) -> ForecastContext:
    """
    Assemble la grille journaliere depuis les observations brutes.

    Prend `perf_observations` (ou `sessions`), pas une enveloppe deja
    calculee : une enveloppe d'enveloppe n'a pas de sens, et la grille
    journaliere doit etre construite a partir des mesures.
    """
    spec = EnvelopeSpec(column=column, window_days=window_days, min_periods=min_periods)
    env = rolling_envelope(observations, spec, calendar="daily")
    if env.empty:
        raise ValueError(
            f"Enveloppe vide sur la colonne {column!r} : rien a prevoir. "
            "Verifier que les observations couvrent au moins "
            f"{min_periods} mesures dans une fenetre de {window_days} jours."
        )

    # Calendrier complet : les jours ou l'enveloppe n'est pas definie doivent
    # exister dans l'index, et y valoir NaN.
    grid = pd.date_range(env["date"].min(), env["date"].max(), freq="D")
    target = env.set_index("date")[spec.label].reindex(grid)

    curves = compute_load_curves(daily_load, tau1, tau2).to_frame().set_index("date")
    frame = pd.DataFrame({"target": target.to_numpy()}, index=grid)
    load = pd.to_numeric(
        pd.Series(daily_load["trimp"].to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(daily_load["date"]))),
        errors="coerce",
    )
    # Un jour sans seance est une charge nulle, pas une charge inconnue :
    # daily_load.parquet code deja les jours de repos par un zero explicite,
    # et les jours hors de son etendue sont anterieurs au suivi.
    frame["trimp"] = load.reindex(grid).fillna(0.0).to_numpy()
    for name in ("fitness", "fatigue"):
        frame[name] = curves[name].reindex(grid).ffill().fillna(0.0).to_numpy()

    frame.index.name = "date"
    return ForecastContext(frame=frame, target_column="target")


# -- adaptateurs de modeles ------------------------------------------------

@dataclass
class ModelSpec:
    """Ce qu'il faut savoir d'un modele pour le citer dans un rapport."""

    key: str
    repo_id: str
    parameters: str
    license: str
    covariates: bool


SPECS = {
    "chronos2": ModelSpec("chronos2", "amazon/chronos-2", "120M", "Apache-2.0", True),
    # Le code de TabPFN est sous Apache-2.0, mais `tabpfn` 9.x exige
    # l'acceptation interactive d'une licence et un compte Prior Labs pour
    # telecharger ses poids. Sans `TABPFN_TOKEN` la variante ne tourne pas, et
    # l'adaptateur attend ce jeton plutot que de se faire passer pour absent.
    "tabpfn-ts": ModelSpec("tabpfn-ts", "PriorLabs/tabpfn-time-series", "11M",
                           "Apache-2.0, poids sous compte", True),
    # Clef « ttm » et non « ttm-r3 » : le point de controle reellement charge
    # est R2, pour la raison expliquee dans _TtmAdapter. Nommer la variante R3
    # ferait passer un chiffre pour celui d'un autre modele. Et pas de
    # covariables en zero-shot, ce qui est mesure et non suppose.
    "ttm": ModelSpec("ttm", "ibm-granite/granite-timeseries-ttm-r2", "855k",
                        "Apache-2.0", False),
}


class _Adapter:
    """
    Socle commun aux modeles a interface « format long ».

    `chronos-forecasting` et `tabpfn-time-series` exposent le meme format
    long — `item_id`, `timestamp`, `target`, puis une colonne par covariable,
    et un `future_df` de meme forme sans la cible. Seuls l'appel et le nom des
    arguments changent, d'ou cette classe de base plutot que deux
    implementations paralleles.

    Le pipeline est charge une fois et mis en cache de classe : relire 120 M
    de parametres a chaque pli couterait bien plus que l'inference elle-meme
    (100 s de chargement contre 0,9 s de prevision, mesures sur ce M3 Pro).
    """

    key: str = ""
    _cache: ClassVar[dict[str, object]] = {}

    def __init__(self, quantile: float = 0.5, device: str = "cpu"):
        self.quantile = quantile
        self.device = device

    def _build(self):
        raise NotImplementedError

    def _pipeline(self):
        cache_key = f"{self.key}:{self.device}"
        if cache_key not in _Adapter._cache:
            _Adapter._cache[cache_key] = self._build()
        return _Adapter._cache[cache_key]

    def warm(self) -> None:
        """
        Force le chargement avant toute mesure de latence.

        Sans cela le premier appel chronometre inclut la lecture des poids, et
        la latence moyenne publiee serait celle d'un demarrage a froid repartie
        sur tous les plis.
        """
        self._pipeline()

    def _forecast(self, pipeline, context_df, future_df, horizon) -> pd.DataFrame:
        raise NotImplementedError

    def predict(self, context: pd.DataFrame, future: pd.DataFrame,
                horizon: int, covariates: tuple[str, ...]) -> np.ndarray:
        ctx = context.reset_index().rename(columns={"date": "timestamp"})
        ctx.insert(0, "item_id", "envelope")
        ctx = ctx[["item_id", "timestamp", "target", *covariates]]

        future_df = None
        if covariates:
            future_df = future.reset_index().rename(columns={"date": "timestamp"})
            future_df.insert(0, "item_id", "envelope")
            future_df = future_df[["item_id", "timestamp", *covariates]]

        out = self._forecast(self._pipeline(), ctx, future_df, horizon)
        return _read_quantile(out, self.quantile, horizon)


class _Chronos2Adapter(_Adapter):
    """Chronos-2, encodeur de 120 M de parametres, licence Apache-2.0."""

    key = "chronos2"

    def _build(self):
        from chronos import Chronos2Pipeline

        return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=self.device)

    def _forecast(self, pipeline, context_df, future_df, horizon):
        return pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=[self.quantile],
        )


class _TabPfnTsAdapter(_Adapter):
    """
    TabPFN-TS : un modele tabulaire detourne en prevision.

    Il reformule la serie en probleme de regression ligne par ligne, ce qui le
    rend naturellement a l'aise sur les series courtes — notre cas — et lui
    permet d'accepter des covariables sans architecture dediee.
    """

    key = "tabpfn-ts"

    def _build(self):
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        # Mode LOCAL explicite : le paquet sait aussi appeler l'API hebergee de
        # Prior Labs, ce qui enverrait les donnees de l'athlete sur un service
        # tiers. Ici tout tourne sur la machine.
        return TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)

    def _forecast(self, pipeline, context_df, future_df, horizon):
        if future_df is None:
            return pipeline.predict_df(
                context_df, prediction_length=horizon, quantiles=[self.quantile])
        return pipeline.predict_df(
            context_df, future_df=future_df, quantiles=[self.quantile])


class _TtmAdapter(_Adapter):
    """
    Tiny Time Mixers d'IBM : 855 000 parametres, licence Apache-2.0.

    Deux ecarts a la fiche du modele, tous deux mesures plutot que supposes.

    **C'est R2, pas R3.** `granite-tsfm` 0.3.10, derniere version publiee sur
    PyPI, ne sait pas charger les points de controle R3 : la moitie des poids
    — dont la tete multi-quantile, nouvelle en R3 — est reinitialisee au
    hasard et la sortie n'est pas finie. Charger R3 avec cette version
    produirait des chiffres sans aucun rapport avec le modele annonce.

    **Il ne voit pas les covariables.** La fiche R3 annonce l'« integration de
    variables exogenes », mais `enable_forecast_channel_mixing` vaut False
    dans le point de controle et le dorsal est independant par canal :
    remplacer entierement les covariables laisse la prevision du canal cible
    identique au bit pres. L'exogene de TTM demande un fine-tuning, que ce
    depot ne peut pas faire honnetement sur 545 points d'un seul athlete.
    `foundation_predictor` refuse donc `with_covariates=True` pour ce modele
    plutot que de publier deux lignes qu'on sait identiques.
    """

    key = "ttm"
    repo_id = "ibm-granite/granite-timeseries-ttm-r2"
    revision = "512-96-ft-r2.1"
    # Le point de controle impose son contexte et son horizon : ce ne sont pas
    # des reglages, ce sont les dimensions de la tete entrainee.
    context_length = 512
    prediction_length = 96
    # 8 = journalier, selon la table de frequences de `tsfm_public`. Ce modele
    # fait du prefix tuning sur la resolution : un jeton faux lui ferait lire
    # une serie horaire.
    freq_token = 8

    def _build(self):
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        model = TinyTimeMixerForPrediction.from_pretrained(
            self.repo_id, revision=self.revision)
        model.eval()
        return model

    def predict(self, context: pd.DataFrame, future: pd.DataFrame,
                horizon: int, covariates: tuple[str, ...]) -> np.ndarray:
        import torch

        if covariates:
            raise ValueError(
                "TTM ignore les covariables en zero-shot : son dorsal est "
                "independant par canal. Voir la docstring de _TtmAdapter."
            )
        if horizon > self.prediction_length:
            raise ValueError(
                f"Horizon de {horizon} au-dela de la tete entrainee "
                f"({self.prediction_length})."
            )

        values = context["target"].to_numpy(dtype=float)[-self.context_length:]
        observed = np.isfinite(values)
        if observed.sum() < 3:
            return np.full(horizon, np.nan)

        # Normalisation sur le contexte seul : prendre les statistiques de
        # toute la serie ferait entrer l'avenir dans la mise a l'echelle.
        mean = float(values[observed].mean())
        std = float(values[observed].std()) or 1.0
        scaled = np.where(observed, (values - mean) / std, 0.0)

        # Les jours non definis sont declares non observes, jamais combles :
        # TTM accepte un masque, ce qui evite de reporter la derniere valeur
        # connue et de biaiser le modele vers la persistance. Le rembourrage a
        # gauche obeit a la meme regle, le point de controle exigeant exactement
        # 512 pas.
        pad = self.context_length - scaled.size
        past = np.concatenate([np.zeros(pad), scaled])[None, :, None]
        mask = np.concatenate([np.zeros(pad, dtype=bool), observed])[None, :, None]

        with torch.no_grad():
            out = self._pipeline()(
                past_values=torch.tensor(past, dtype=torch.float32),
                past_observed_mask=torch.tensor(mask),
                freq_token=torch.tensor([self.freq_token], dtype=torch.long),
            ).prediction_outputs

        return out[0, :horizon, 0].numpy().astype(float) * std + mean


def _read_quantile(frame: pd.DataFrame, quantile: float, horizon: int) -> np.ndarray:
    """
    Extrait la colonne de quantile demandee, quel que soit son nom.

    Les deux paquets nomment leurs sorties differemment, et le nom change
    entre versions mineures. Chercher la colonne plutot que la supposer evite
    un echec silencieux : une colonne absente doit lever, pas faire renvoyer
    la premiere colonne numerique venue.
    """
    frame = frame.reset_index() if frame.index.nlevels > 1 else frame
    for name in (quantile, str(quantile), f"{quantile:.1f}"):
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            if values.size != horizon:
                raise ValueError(
                    f"Le modele a renvoye {values.size} points pour un horizon "
                    f"de {horizon}."
                )
            return values
    raise ValueError(
        f"Aucune colonne pour le quantile {quantile} dans {list(frame.columns)}. "
        "L'API du paquet a probablement change."
    )


_ADAPTERS = {
    "chronos2": _Chronos2Adapter,
    "tabpfn-ts": _TabPfnTsAdapter,
    "ttm": _TtmAdapter,
}


def is_available(model: str) -> bool:
    """Le paquet du modele est-il installe ? Sert a ne pas enregistrer une
    variante qu'on ne pourra pas executer."""
    import importlib.util

    modules = {"chronos2": "chronos", "tabpfn-ts": "tabpfn_time_series",
               "ttm": "tsfm_public"}
    name = modules.get(model)
    if name is None:
        raise ValueError(f"Modele inconnu : {model!r}")
    return importlib.util.find_spec(name) is not None


# -- prediction ------------------------------------------------------------

@dataclass
class Timing:
    """Latences mesurees, pour que le README cite des chiffres de cette machine."""

    calls: int = 0
    seconds: float = 0.0
    horizons: list[int] = field(default_factory=list)

    @property
    def mean_seconds(self) -> float:
        return self.seconds / self.calls if self.calls else float("nan")


def foundation_predictor(
    context: ForecastContext,
    model: str = "chronos2",
    with_covariates: bool = True,
    quantile: float = 0.5,
    device: str = "cpu",
    timing: Timing | None = None,
) -> Predictor:
    """
    Fabrique un prediteur conforme a l'interface de `src/evaluation`.

    `with_covariates=False` donne la variante d'ablation : meme modele, meme
    cible, meme decoupage, mais aveugle a la charge. La difference des deux
    scores est le seul chiffre qui reponde a la question du depot.
    """
    if model not in _ADAPTERS:
        raise ValueError(f"Modele inconnu : {model!r}")
    if with_covariates and not SPECS[model].covariates:
        raise ValueError(
            f"{model} n'exploite pas les covariables : enregistrer une variante "
            "« + charge » produirait une ligne identique a « sans charge »."
        )
    adapter = _ADAPTERS[model](quantile=quantile, device=device)
    covariates = context.covariate_columns if with_covariates else ()
    frame = context.frame

    def predict(dates_tr, y_tr, dates_te) -> np.ndarray:
        te = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates_te))).normalize()
        cutoff = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates_tr))).normalize().max()

        if te.empty:
            return np.array([], dtype=float)

        # Un modele zero-shot n'ajuste rien : il n'y a pas de R2 d'ajustement
        # a reporter. Le cas se produit quand `cross_validate` rappelle le
        # prediteur avec les dates d'entrainement comme dates de test.
        if te.min() <= cutoff:
            return np.full(len(te), np.nan)

        horizon = int((te.max() - cutoff).days)
        if horizon > MAX_HORIZON:
            raise ValueError(
                f"Horizon de {horizon} jours au-dela de la limite du modele "
                f"({MAX_HORIZON})."
            )

        ctx = frame.loc[frame.index <= cutoff].tail(MAX_CONTEXT)
        if ctx[context.target_column].notna().sum() < 3:
            return np.full(len(te), np.nan)

        future_index = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
        # `rename_axis` parce que `reindex` perd le nom de l'index, et que les
        # adaptateurs attendent une colonne `date` apres `reset_index`.
        future = frame.reindex(future_index).rename_axis("date")
        future["trimp"] = future["trimp"].fillna(0.0)
        future[["fitness", "fatigue"]] = future[["fitness", "fatigue"]].ffill()

        adapter.warm()
        started = time.perf_counter()
        values = adapter.predict(ctx, future, horizon, covariates)
        if timing is not None:
            timing.calls += 1
            timing.seconds += time.perf_counter() - started
            timing.horizons.append(horizon)

        series = pd.Series(values, index=future_index)
        return series.reindex(te).to_numpy(dtype=float)

    return predict


def variant_name(model: str, with_covariates: bool) -> str:
    """Nom affiche dans le tableau. Doit dire si la charge est vue ou non :
    c'est la seule difference entre les deux lignes."""
    return f"{model} {'+ charge' if with_covariates else 'sans charge'}"


def build_foundation_variants(
    context: ForecastContext,
    models: tuple[str, ...] = ("chronos2",),
    device: str = "cpu",
    timings: dict[str, Timing] | None = None,
) -> dict[str, Predictor]:
    """
    Les variantes a ajouter au tableau, par paires avec et sans covariables.

    Un modele dont le paquet manque n'est PAS enregistre. `cross_validate`
    avale les exceptions et rapporterait `n_folds=0`, ce qui ressemblerait a
    un resultat mesure alors que rien n'a tourne.
    """
    variants: dict[str, Predictor] = {}
    for model in models:
        if not is_available(model):
            continue
        # Un modele aveugle aux covariables n'a pas de paire : la seconde ligne
        # serait la premiere, et la presenter comme une ablation serait faux.
        options = (False, True) if SPECS[model].covariates else (False,)
        for with_cov in options:
            name = variant_name(model, with_cov)
            timing = Timing()
            if timings is not None:
                timings[name] = timing
            variants[name] = foundation_predictor(
                context, model=model, with_covariates=with_cov,
                device=device, timing=timing)
    return variants
