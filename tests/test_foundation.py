"""
Modeles de fondation comme references supplementaires.

Ces tests tournent hors ligne : le modele est remplace par un double dont la
sortie est connue. Ce qui est verifie ici n'est pas la qualite des previsions
— elle se mesure, elle ne s'affirme pas — mais le plomberie qui les rend
interpretables : alignement des dates, causalite de la coupure, absence de
R2 d'ajustement, et refus de laisser une panne ressembler a un resultat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import foundation
from src.evaluation.foundation import (
    SPECS,
    ForecastContext,
    Timing,
    build_forecast_context,
    build_foundation_variants,
    foundation_predictor,
    is_available,
    variant_name,
)
from src.evaluation.validation import cross_validate, horizon_splits


def _observations(n: int = 80, freq: str = "5D") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=n, freq=freq),
        "pi": rng.normal(1.0, 0.05, n),
    })


def _daily_load(start: str = "2018-06-01", days: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=days, freq="D"),
        "trimp": rng.gamma(2.0, 40.0, days) * (rng.random(days) > 0.4),
    })


@pytest.fixture
def context() -> ForecastContext:
    return build_forecast_context(_observations(), _daily_load(), "pi")


class _Echo:
    """
    Double de modele : renvoie la derniere valeur connue de la cible.

    Choisi pour etre exactement la persistance, donc verifiable a la main. Il
    enregistre aussi ce qu'il a recu, ce qui permet d'affirmer que le
    prediteur ne lui a pas passe d'information future.
    """

    def __init__(self, *_, **__):
        self.seen: list[dict] = []

    def warm(self) -> None:
        """Rien a charger. Presente parce que le prediteur prechauffe le
        modele avant de demarrer son chronometre."""

    def predict(self, context, future, horizon, covariates):
        self.seen.append({
            "last_context_date": context.index.max(),
            "first_future_date": future.index.min(),
            "covariates": tuple(covariates),
            "horizon": horizon,
        })
        last = context["target"].dropna().iloc[-1]
        return np.full(horizon, float(last))


@pytest.fixture
def echo(monkeypatch) -> _Echo:
    double = _Echo()
    monkeypatch.setitem(foundation._ADAPTERS, "chronos2", lambda **kwargs: double)
    return double


# -- la grille journaliere -------------------------------------------------

def test_context_is_a_gapless_daily_calendar(context):
    dates = context.dates
    assert (dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert list(context.covariate_columns) == ["trimp", "fitness", "fatigue"]


def test_context_marks_undefined_days_as_nan_rather_than_filling_them():
    """
    Une interruption de mesure doit laisser des NaN. Reporter la derniere
    valeur connue fabriquerait de la donnee et, pire, rendrait la cible
    artificiellement facile a prevoir par persistance.
    """
    obs = pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03",
                                "2019-10-01", "2019-10-02", "2019-10-03"]),
        "pi": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    })
    ctx = build_forecast_context(obs, _daily_load("2018-12-01", 400), "pi")
    hole = ctx.frame.loc["2019-04-01":"2019-08-01", "target"]
    assert hole.isna().all()
    assert ctx.frame["target"].notna().sum() < len(ctx.frame)


def test_context_covariates_are_never_nan(context):
    """
    Un jour sans seance est une charge nulle, pas une charge inconnue :
    daily_load code deja le repos par un zero explicite.
    """
    assert context.frame[list(context.covariate_columns)].notna().all().all()


def test_context_refuses_an_empty_envelope():
    with pytest.raises(ValueError, match="Enveloppe vide"):
        build_forecast_context(_observations(n=2), _daily_load(), "pi")


# -- causalite et alignement -----------------------------------------------

def test_prediction_is_read_at_the_requested_dates(context, echo):
    predict = foundation_predictor(context, "chronos2")
    dates = pd.Series(context.dates)
    train, test = dates[:300], dates[300:320]

    out = predict(train, context.frame["target"].to_numpy()[:300], test)
    assert out.shape == (len(test),)
    assert np.isfinite(out).all()


def test_the_model_never_sees_a_date_beyond_the_cutoff(context, echo):
    predict = foundation_predictor(context, "chronos2")
    dates = pd.Series(context.dates)
    train, test = dates[:300], dates[300:320]

    predict(train, context.frame["target"].to_numpy()[:300], test)
    call = echo.seen[-1]
    assert call["last_context_date"] == train.max()
    assert call["first_future_date"] == train.max() + pd.Timedelta(days=1)


def test_the_horizon_counts_days_not_observations(context, echo):
    predict = foundation_predictor(context, "chronos2")
    dates = pd.Series(context.dates)
    train, test = dates[:300], dates[[320, 340]]

    predict(train, context.frame["target"].to_numpy()[:300], test)
    assert echo.seen[-1]["horizon"] == (test.max() - train.max()).days


def test_ablation_hides_the_covariates_from_the_model(context, echo):
    dates = pd.Series(context.dates)
    y = context.frame["target"].to_numpy()
    for with_cov, expected in [(True, ("trimp", "fitness", "fatigue")), (False, ())]:
        foundation_predictor(context, "chronos2", with_covariates=with_cov)(
            dates[:300], y[:300], dates[300:320])
        assert echo.seen[-1]["covariates"] == expected


def test_a_horizon_beyond_the_model_limit_raises(context, echo):
    predict = foundation_predictor(context, "chronos2")
    dates = pd.Series(context.dates)
    far = pd.Series([dates.iloc[10] + pd.Timedelta(days=foundation.MAX_HORIZON + 1)])
    with pytest.raises(ValueError, match="au-dela de la limite"):
        predict(dates[:10], context.frame["target"].to_numpy()[:10], far)


# -- le R2 d'ajustement n'existe pas ---------------------------------------

def test_in_sample_prediction_is_nan_because_nothing_was_fitted(context, echo):
    """
    Un modele zero-shot n'ajuste rien : il n'y a pas de R2 d'ajustement a
    reporter. NaN est la reponse juste, pas une lacune.
    """
    predict = foundation_predictor(context, "chronos2")
    dates = pd.Series(context.dates)
    y = context.frame["target"].to_numpy()
    assert np.isnan(predict(dates, y, dates)).all()


def test_cross_validate_reports_no_overfitting_gap(context, echo):
    dates = pd.Series(context.dates)
    y = context.frame["target"].to_numpy()
    defined = pd.Series(y).notna().to_numpy()
    dates, y = dates[defined].reset_index(drop=True), y[defined]

    splits = list(horizon_splits(dates, horizon_days=28, min_train_days=120, step_days=28))
    result = cross_validate("chronos2", foundation_predictor(context, "chronos2"),
                            dates, y, splits=splits)
    assert result.n_folds == len(splits)
    assert np.isnan(result.r2_in_sample)
    assert np.isnan(result.overfitting_gap)
    assert np.isfinite(result.r2_out_of_sample)


# -- enregistrement des variantes ------------------------------------------

def test_a_missing_package_is_not_registered_as_a_variant(monkeypatch, context):
    """
    `cross_validate` avale les exceptions : une variante non executable
    ressortirait en `folds = 0` avec des NaN, ce qui ressemblerait a un
    resultat mesure. Mieux vaut ne pas l'enregistrer du tout.
    """
    monkeypatch.setattr(foundation, "is_available", lambda model: False)
    assert build_foundation_variants(context, ("chronos2",)) == {}


def test_variants_come_in_pairs_with_and_without_covariates(monkeypatch, context):
    monkeypatch.setattr(foundation, "is_available", lambda model: True)
    variants = build_foundation_variants(context, ("chronos2",))
    assert set(variants) == {variant_name("chronos2", True), variant_name("chronos2", False)}


def test_variant_names_say_whether_the_load_is_visible():
    assert "sans charge" in variant_name("chronos2", False)
    assert "+ charge" in variant_name("chronos2", True)


def test_timings_are_recorded_for_the_readme(context, echo):
    timing = Timing()
    predict = foundation_predictor(context, "chronos2", timing=timing)
    dates = pd.Series(context.dates)
    y = context.frame["target"].to_numpy()
    predict(dates[:300], y[:300], dates[300:320])
    assert timing.calls == 1
    assert timing.seconds >= 0.0
    assert np.isfinite(timing.mean_seconds)


def test_an_empty_test_block_returns_an_empty_forecast(context, echo):
    predict = foundation_predictor(context, "chronos2")
    dates = pd.Series(context.dates)
    out = predict(dates[:100], context.frame["target"].to_numpy()[:100], dates[100:100])
    assert out.shape == (0,)


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="Modele inconnu"):
        is_available("timesfm")


def test_every_declared_model_carries_its_licence():
    """
    Deux modeles mieux classes publiquement — Moirai-2 et TimesFM 3.0 — ont
    ete ecartes pour licence non commerciale, et les poids de TabPFN exigent
    un compte. Porter la licence dans le code evite qu'un successeur en
    ajoute un troisieme sans s'en apercevoir.
    """
    for spec in SPECS.values():
        assert spec.license.startswith("Apache-2.0")
        assert spec.repo_id and spec.parameters
    assert "compte" in SPECS["tabpfn-ts"].license, (
        "Les poids de TabPFN ne se telechargent pas sans compte Prior Labs : "
        "ecrire « Apache-2.0 » tout court laisserait croire le contraire."
    )


# -- le quantile ------------------------------------------------------------

def test_a_missing_quantile_column_raises_rather_than_guessing():
    frame = pd.DataFrame({"mean": [1.0, 2.0], "predictions": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Aucune colonne pour le quantile"):
        foundation._read_quantile(frame, 0.5, 2)


def test_a_truncated_forecast_raises():
    frame = pd.DataFrame({"0.5": [1.0, 2.0]})
    with pytest.raises(ValueError, match="pour un horizon"):
        foundation._read_quantile(frame, 0.5, 5)


# -- TTM : ce que la fiche annonce et ce que le modele fait -----------------

def test_ttm_refuses_covariates_rather_than_ignoring_them(context):
    """
    La fiche de TTM annonce l'integration de variables exogenes. En zero-shot
    son dorsal est independant par canal : la variante « + charge » serait
    identique a « sans charge », et l'enregistrer la ferait passer pour une
    ablation. Le refus est explicite pour cette raison.
    """
    with pytest.raises(ValueError, match="n'exploite pas les covariables"):
        foundation_predictor(context, "ttm", with_covariates=True)


def test_a_model_without_covariates_gets_no_pair(monkeypatch, context):
    monkeypatch.setattr(foundation, "is_available", lambda model: True)
    variants = build_foundation_variants(context, ("ttm",))
    assert set(variants) == {variant_name("ttm", False)}


def test_the_declared_covariate_support_matches_the_adapters():
    """
    `SPECS` pilote l'enregistrement des variantes : une declaration fausse
    publierait une ligne vide ou une ligne en double. Elle doit donc rester
    alignee sur ce que chaque adaptateur accepte reellement.
    """
    assert SPECS["chronos2"].covariates
    assert not SPECS["ttm"].covariates
    assert SPECS["ttm"].repo_id.endswith("ttm-r2"), (
        "La clef dit « ttm » parce que le point de controle charge est R2 : "
        "granite-tsfm 0.3.10 ne sait pas lire R3."
    )


@pytest.mark.network
@pytest.mark.slow
def test_ttm_ignores_its_covariates_in_zero_shot():
    """
    La mesure qui justifie de n'enregistrer TTM qu'en univarie. On remplace
    entierement les canaux de covariables et on verifie que la prevision du
    canal cible ne bouge pas d'un bit. Si ce test se met a echouer, c'est que
    le point de controle a gagne un melange de canaux — et la variante
    « + charge » redevient legitime.
    """
    import torch

    from src.evaluation.foundation import _TtmAdapter

    model = _TtmAdapter()._pipeline()
    gen = torch.Generator().manual_seed(0)
    past = torch.randn(1, _TtmAdapter.context_length, 4, generator=gen)
    freq = torch.tensor([_TtmAdapter.freq_token], dtype=torch.long)

    with torch.no_grad():
        before = model(past_values=past, freq_token=freq).prediction_outputs
    altered = past.clone()
    altered[:, :, 1:] = torch.randn(1, _TtmAdapter.context_length, 3, generator=gen) * 50 + 100
    with torch.no_grad():
        after = model(past_values=altered, freq_token=freq).prediction_outputs

    assert torch.equal(before[0, :, 0], after[0, :, 0])


@pytest.mark.network
@pytest.mark.slow
def test_chronos2_does_react_to_its_covariates(context):
    """
    Le pendant du test precedent : Chronos-2 doit, lui, changer d'avis quand
    la charge change. Sans cette verification, un resultat nul pourrait venir
    d'une covariable silencieusement ignoree plutot que d'une absence d'effet.
    """
    dates = pd.Series(context.dates)
    y = context.frame["target"].to_numpy()
    cut = len(dates) - 28
    train, test = dates[:cut], dates[cut:]

    with_load = foundation_predictor(context, "chronos2", with_covariates=True)
    baseline = with_load(train, y[:cut], test)

    shuffled = ForecastContext(frame=context.frame.copy(),
                               target_column=context.target_column)
    rng = np.random.default_rng(0)
    for column in shuffled.covariate_columns:
        shuffled.frame[column] = rng.permutation(shuffled.frame[column].to_numpy())
    altered = foundation_predictor(shuffled, "chronos2", with_covariates=True)(
        train, y[:cut], test)

    assert np.isfinite(baseline).all() and np.isfinite(altered).all()
    assert not np.allclose(baseline, altered)
