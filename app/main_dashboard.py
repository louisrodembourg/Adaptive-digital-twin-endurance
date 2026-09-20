"""
Tableau de bord.

Lit exclusivement les artefacts produits par `python -m src.cli ingest` et
n'invente aucune donnee. La version precedente appelait
`bootstrap_trimp_history()` au premier lancement et enregistrait
quatre-vingt-quatre seances generees sous l'etiquette `source="fit_file"` :
tous les graphiques affiches etaient synthetiques, sans que rien ne
l'indique. Un magasin vide affiche desormais la commande a lancer.

Quatre pages. La derniere, Diagnostics, est celle qui porte le resultat du
projet : elle publie les scores hors echantillon de chaque variante, y
compris quand le modele perd contre la persistance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config
from src.evaluation.report import evaluate
from src.layer_a_signal.envelope import (
    EnvelopeSpec,
    describe_predictability,
    envelope_series,
    rolling_envelope,
)
from src.layer_b_twin.load_model import acute_chronic_ratio, compute_load_curves
from src.pipeline.orchestrator import (
    load_calibration,
    load_daily_load,
    load_perf_observations,
    load_sessions,
    provenance_of,
)

st.set_page_config(page_title="Digital Twin — Endurance", page_icon="🏃", layout="wide")

ACCENT = "#4c8bf5"
MUTED = "#8a8f98"
WARN = "#e0a355"


# ── chargement ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_store() -> dict[str, pd.DataFrame | dict | None]:
    """Les trois artefacts et la calibration, tels quels."""
    return {
        "sessions": load_sessions(),
        "daily": load_daily_load(),
        "observations": load_perf_observations(),
        "calibration": load_calibration(),
    }


def empty_state() -> None:
    st.title("Digital Twin — Endurance")
    st.warning("Aucune donnee ingeree.")
    st.markdown(
        "Ce tableau de bord ne lit que des donnees reelles et n'en fabrique "
        "aucune. Pour le remplir :"
    )
    st.code(
        "python -m src.cli screen-athletes --n 200   # choisir un athlete\n"
        "python -m src.cli ingest                    # construire les artefacts\n"
        "python -m src.cli calibrate                 # reporter les estimations",
        language="bash",
    )
    st.stop()


store = load_store()
sessions: pd.DataFrame = store["sessions"]
if sessions.empty:
    empty_state()

daily: pd.DataFrame = store["daily"]
observations: pd.DataFrame = store["observations"]
calibration: dict | None = store["calibration"]

try:
    cfg = load_config()
except ConfigError:
    cfg = None


# ── barre laterale ────────────────────────────────────────────────────────

provenance = provenance_of(sessions)

with st.sidebar:
    st.title("Digital Twin")
    if provenance.is_trustworthy:
        st.success(provenance.banner)
    else:
        st.error(provenance.banner)
    st.caption(
        f"{pd.to_datetime(daily['date']).min():%b %Y} → "
        f"{pd.to_datetime(daily['date']).max():%b %Y}"
    )
    st.divider()
    page = st.radio(
        "Page",
        ["Charge", "Performance", "Seances", "Diagnostics"],
        label_visibility="collapsed",
    )
    st.divider()
    tau1 = st.slider("Constante de fitness (j)", 20, 90, 42, 2)
    tau2 = st.slider("Constante de fatigue (j)", 3, 25, 12, 1)
    if tau1 <= tau2:
        st.error("La constante de fitness doit depasser celle de fatigue.")
        st.stop()


curves = compute_load_curves(daily, float(tau1), float(tau2)).to_frame()


def _line(fig: go.Figure, x, y, name: str, color: str, width: float = 2.0,
          dash: str | None = None) -> None:
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines",
                             line={"color": color, "width": width, "dash": dash}))


def _layout(fig: go.Figure, ytitle: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis_title=ytitle, hovermode="x unified",
        legend={"orientation": "h", "y": 1.1, "x": 0},
    )
    return fig


# ── page Charge ───────────────────────────────────────────────────────────

if page == "Charge":
    st.header("Charge d'entrainement")
    st.caption(
        "Moyennes mobiles exponentielles du TRIMP quotidien. Toutes les "
        "disciplines comptent : la charge est systemique."
    )

    last = curves.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fitness", f"{last['fitness']:.0f}", help="TRIMP/jour, constante longue")
    c2.metric("Fatigue", f"{last['fatigue']:.0f}", help="TRIMP/jour, constante courte")
    c3.metric("Fraicheur", f"{last['form']:+.0f}", help="fitness moins fatigue, decale d'un jour")
    c4.metric("Seances", f"{len(sessions)}")

    fig = go.Figure()
    _line(fig, curves["date"], curves["fitness"], "Fitness", ACCENT)
    _line(fig, curves["date"], curves["fatigue"], "Fatigue", WARN)
    _line(fig, curves["date"], curves["form"], "Fraicheur", MUTED, 1.5, "dot")
    st.plotly_chart(_layout(fig, "TRIMP / jour"), width="stretch")

    st.subheader("Rapport charge aigue sur chronique")
    st.caption(
        "Sept jours contre vingt-huit. Contrairement au rMSSD et au DFA, "
        "cette regle ne demande que la charge — elle se declenche vraiment."
    )
    acwr = acute_chronic_ratio(daily).rename("acwr")
    a = pd.DataFrame({"date": daily["date"], "acwr": acwr}).dropna()
    fig = go.Figure()
    _line(fig, a["date"], a["acwr"], "ACWR", ACCENT)
    bound = cfg.planner.get("acwr_upper_bound", 1.5) if cfg else 1.5
    fig.add_hline(y=bound, line_dash="dash", line_color=WARN,
                  annotation_text=f"seuil {bound}")
    st.plotly_chart(_layout(fig, "ratio"), width="stretch")


# ── page Performance ──────────────────────────────────────────────────────

elif page == "Performance":
    st.header("Performance")
    st.info(
        "**Ce qui est mesure ici est une enveloppe**, pas l'effort d'une seance. "
        "Le meilleur effort d'une seance decrit ce qu'etait la seance — "
        "fractionne ou recuperation — et non la capacite de l'athlete : son "
        "autocorrelation vaut -0,12 en course et -0,00 au velo, soit du bruit "
        "autour d'un niveau. L'enveloppe glissante repond a « quel est le "
        "meilleur dont cet athlete ait ete capable recemment » et monte a 0,79 "
        "et 0,93 respectivement."
    )

    window = st.slider("Fenetre de l'enveloppe (jours)", 14, 120, 42, 7)

    tabs = st.tabs(["Course", "Velo"])

    with tabs[0]:
        if observations.empty:
            st.warning("Aucune observation de performance en course.")
        else:
            obs = observations[["date", "perf_index"]].rename(columns={"perf_index": "pi"})
            env = rolling_envelope(obs, EnvelopeSpec("pi", window, min_periods=3))
            col = f"pi_env{window}"
            stats = describe_predictability(env[col]) if not env.empty else {}
            c1, c2, c3 = st.columns(3)
            c1.metric("Observations", f"{len(observations)}")
            c2.metric("Autocorrelation", f"{stats.get('autocorr_lag1', float('nan')):+.2f}",
                      help="proche de zero = bruit, rien a modeliser")
            c3.metric("Enveloppe actuelle", f"{env[col].iloc[-1]:.3f}" if not env.empty else "—")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=obs["date"], y=obs["pi"], name="effort de seance",
                mode="markers", marker={"color": MUTED, "size": 5, "opacity": 0.6}))
            if not env.empty:
                _line(fig, env["date"], env[col], f"enveloppe {window} j", ACCENT, 2.5)
            st.plotly_chart(_layout(fig, "index de performance"), width="stretch")

    with tabs[1]:
        if "cp20_w" not in sessions.columns or sessions["cp20_w"].notna().sum() == 0:
            st.warning("Aucune puissance critique disponible.")
        else:
            env = envelope_series(sessions, "cp20_w", window, sport_filter="bike")
            col = f"cp20_w_env{window}"
            bike = sessions[sessions["sport"].str.lower().str.startswith("bike")]
            raw = bike[["date", "cp20_w"]].dropna()
            stats = describe_predictability(env[col]) if not env.empty else {}

            c1, c2, c3 = st.columns(3)
            c1.metric("Seances avec puissance", f"{len(raw)}")
            c2.metric("Autocorrelation", f"{stats.get('autocorr_lag1', float('nan')):+.2f}")
            c3.metric("Enveloppe actuelle", f"{env[col].iloc[-1]:.0f} W" if not env.empty else "—")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=raw["date"], y=raw["cp20_w"], name="puissance 20 min de seance",
                mode="markers", marker={"color": MUTED, "size": 5, "opacity": 0.6}))
            if not env.empty:
                _line(fig, env["date"], env[col], f"enveloppe {window} j", ACCENT, 2.5)
            st.plotly_chart(_layout(fig, "watts"), width="stretch")


# ── page Seances ──────────────────────────────────────────────────────────

elif page == "Seances":
    st.header("Seances")

    by_sport = sessions["sport"].value_counts()
    st.caption(
        "Toutes les disciplines alimentent la charge. "
        + ", ".join(f"{k} {v}" for k, v in by_sport.head(6).items())
    )
    if "Unknown" in by_sport.index:
        st.caption(
            f":grey[{by_sport['Unknown']} seances sans discipline renseignee comptent "
            "dans la charge ; VirtualRide est distingue de Bike.]"
        )

    trimp_ref = pd.to_numeric(sessions.get("trimp_reference"), errors="coerce")
    ours = pd.to_numeric(sessions.get("trimp"), errors="coerce")
    both = pd.DataFrame({"nous": ours, "gc": trimp_ref}).dropna()

    if len(both) > 20:
        st.subheader("Recoupement du TRIMP")
        slope, intercept = np.polyfit(both["gc"], both["nous"], 1)
        resid = both["nous"] - (slope * both["gc"] + intercept)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pente", f"{slope:.3f}", help="1.0 = meme echelle que GoldenCheetah")
        c2.metric("Ordonnee", f"{intercept:+.1f}")
        c3.metric("Ecart-type residuel", f"{resid.std():.1f}")
        st.caption(
            "Une regression, pas un coefficient de rang : Spearman est invariant "
            "par transformation monotone et ne prouverait que l'ordre. Or le TRIMP "
            "est une grandeur d'echelle qui alimente une convolution lineaire."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=both["gc"], y=both["nous"], mode="markers",
                                 marker={"color": ACCENT, "size": 5, "opacity": 0.5},
                                 name="seances"))
        lim = [0, float(both.max().max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="identite",
                                 line={"color": MUTED, "dash": "dash"}))
        fig.update_layout(xaxis_title="TRIMP GoldenCheetah", yaxis_title="TRIMP calcule")
        st.plotly_chart(_layout(fig), width="stretch")

    st.subheader("Detail")
    cols = [c for c in ("date", "sport", "duration_total_s", "trimp", "trimp_reference",
                        "hr_mean", "hr_max", "cp20_w", "ef_global", "cardiac_decoupling")
            if c in sessions.columns]
    st.dataframe(sessions[cols].sort_values("date", ascending=False),
                 width="stretch", height=420)


# ── page Diagnostics ──────────────────────────────────────────────────────

elif page == "Diagnostics":
    st.header("Diagnostics du modele")
    st.caption(
        "Chaque variante passe sous le meme protocole : validation croisee "
        "temporelle, references declarees d'avance, classement sur le score "
        "hors echantillon et jamais sur l'ajustement."
    )

    target = st.radio("Grandeur a expliquer", ["Enveloppe course", "Effort de seance"],
                      horizontal=True)

    if observations.empty:
        st.warning("Aucune observation disponible.")
        st.stop()

    obs = observations[["date", "perf_index"]].rename(columns={"perf_index": "pi"})
    if target == "Enveloppe course":
        frame = rolling_envelope(obs, EnvelopeSpec("pi", 42, min_periods=3))
        frame = frame.rename(columns={"pi_env42": "perf"})
    else:
        frame = obs.rename(columns={"pi": "perf"})

    if len(frame) < 30:
        st.warning(f"Seulement {len(frame)} points : trop peu pour valider.")
        st.stop()

    table = evaluate(frame, daily, target="perf",
                     taus=[(float(tau1), float(tau2)), (60.0, 15.0)])

    winners = table[table["r2_out"] > 0]
    if winners.empty:
        st.error(
            "Aucune variante ne fait mieux que predire la moyenne. "
            "Quand l'observable est l'effort de chaque seance, la serie est "
            "du bruit autour d'un niveau et il n'y a rien a apprendre."
        )
    else:
        best = winners.iloc[0]
        st.success(f"Meilleure variante hors echantillon : **{best['name']}** "
                   f"(R² = {best['r2_out']:+.3f})")

    display = table.rename(columns={
        "name": "variante", "r2_in": "R² ajustement", "r2_out": "R² hors echantillon",
        "gap": "ecart", "rmse_out": "RMSE", "skill_vs_mean": "gain / moyenne",
    })
    st.dataframe(
        display[["variante", "R² ajustement", "R² hors echantillon", "ecart",
                 "RMSE", "gain / moyenne"]].style.format({
            "R² ajustement": "{:+.4f}", "R² hors echantillon": "{:+.4f}",
            "ecart": "{:+.4f}", "RMSE": "{:.5f}", "gain / moyenne": "{:+.4f}",
        }),
        width="stretch", hide_index=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["name"], y=table["r2_in"], name="ajustement",
                         marker_color=MUTED))
    fig.add_trace(go.Bar(x=table["name"], y=table["r2_out"], name="hors echantillon",
                         marker_color=ACCENT))
    fig.add_hline(y=0, line_color="#555")
    fig.update_layout(barmode="group", xaxis_tickangle=-30)
    st.plotly_chart(_layout(fig, "R²", 420), width="stretch")

    st.markdown(
        "**Comment lire la colonne ecart.** Elle vaut ajustement moins hors "
        "echantillon. Un modele qui colle aux donnees vues et s'effondre sur "
        "les suivantes a memorise plutot qu'appris. Sur l'enveloppe de course, "
        "Banister obtient le meilleur ajustement de toutes les variantes et "
        "parmi les pires scores hors echantillon.\n\n"
        "**Pourquoi la persistance figure parmi les references.** Une enveloppe "
        "glissante est fortement autocorrelee par construction, donc facile a "
        "predire par simple inertie. Un modele de charge qui ne bat pas "
        "« repeter la derniere valeur » n'apporte rien, quel que soit son R²."
    )

    if calibration:
        with st.expander("Calibration de l'athlete"):
            curve = calibration.get("reference_curve", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("FC max estimee", f"{calibration.get('hr_max_estimated', float('nan')):.0f}")
            c2.metric("FC repos estimee", f"{calibration.get('hr_rest_estimated', float('nan')):.0f}")
            c3.metric("Exposant de Riegel", f"{curve.get('exponent', float('nan')):.4f}")
            st.caption(
                "Les predictions de chrono au-dela de quarante-cinq minutes sont "
                "des extrapolations de la courbe de reference, ajustee sur des "
                "efforts de cinq a quarante-cinq minutes. Aucun semi-marathon ne "
                "figure dans les donnees de cet athlete."
            )
