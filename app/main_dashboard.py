"""
Tableau de bord.

Lit exclusivement les artefacts produits par `python -m src.cli ingest` et
n'invente aucune donnee. La version precedente appelait
`bootstrap_trimp_history()` au premier lancement et enregistrait
quatre-vingt-quatre seances generees sous l'etiquette `source="fit_file"` :
tous les graphiques affiches etaient synthetiques, sans que rien ne
l'indique. Un magasin vide affiche desormais la commande a lancer.

Cinq pages. Diagnostics publie les scores hors echantillon de chaque variante,
y compris quand le modele perd contre la persistance. Mesures lit les
artefacts deja produits par `cli evaluate`, `cohort` et `pool` plutot que de
recalculer : ces protocoles coutent des minutes a des dizaines de minutes, et
un tableau de bord qui les relance a chaque clic finit par ne plus etre
ouvert. Une section dont l'artefact manque affiche la commande qui le produit,
et non un graphique vide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import (
    ABLATION_FILE,
    COHORT_FILE,
    DURATION_PAIRS_FILE,
    DURATIONS_FILE,
    FLOORS_FILE,
    PLAN_RULES_FILE,
    POOLING_EFFECTS_FILE,
    POOLING_FILE,
    RESPONDERS_CORR_FILE,
    RESULTS_FILE,
)
from src.config import ConfigError, load_config
from src.evaluation.cohort import detection_contrast
from src.evaluation.readiness import assess_readiness
from src.evaluation.report import BIKE_SPORTS, BIKE_TARGET_COLUMN, evaluate
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


@st.cache_data(ttl=300)
def load_measurements() -> dict[str, pd.DataFrame | None]:
    """
    Les artefacts de mesure, ou None quand ils n'ont pas ete produits.

    On ne recalcule rien ici : `cli evaluate` prend des minutes, `cli pool`
    des dizaines. Un artefact absent est une information — la commande n'a
    pas tourne — et doit s'afficher comme telle.
    """
    try:
        processed = load_config().paths.processed
    except ConfigError:
        return {}
    names = {
        "resultats": RESULTS_FILE, "ablation": ABLATION_FILE,
        "durees": DURATIONS_FILE, "planchers": FLOORS_FILE,
        "paires_durees": DURATION_PAIRS_FILE,
        "cohorte": COHORT_FILE, "pooling": POOLING_FILE,
        "regles": PLAN_RULES_FILE, "repondeurs": RESPONDERS_CORR_FILE,
        "effets": POOLING_EFFECTS_FILE,
        # Les deux experiences qui CONTESTENT l'ajustement publie. Elles
        # ecrivent a part pour ne pas l'ecraser, donc il faut les relire a
        # part : sans elles, l'onglet presenterait tau1 = 70,3 comme une
        # mesure physiologique, ce que la mesure a precisement refute.
        "pooling_f21": POOLING_FILE.replace(".parquet", "_f21.parquet"),
        "pooling_n60": POOLING_FILE.replace(".parquet", "_n60.parquet"),
    }
    out: dict[str, pd.DataFrame | None] = {}
    for key, filename in names.items():
        path = processed / filename
        try:
            out[key] = pd.read_parquet(path) if path.exists() else None
        except Exception:
            out[key] = None
    return out


def _fenetre_comparee(publie: pd.DataFrame,
                      controle: pd.DataFrame | None,
                      reduit: pd.DataFrame | None) -> None:
    """
    Ce que la fenetre de lissage explique des constantes de temps.

    Sans ce bloc, l'onglet afficherait tau1 = 70,3 jours comme une mesure
    physiologique. L'elasticite mesuree vaut 0,22 : environ un quart de ces
    constantes vient du protocole, et la cible elle-meme change entre les
    deux ajustements — l'ecart ne se lit pas sans le controle de composition.
    """
    if reduit is None or controle is None:
        st.info(
            "Les deux ajustements qui contestent ces constantes n'ont pas "
            "tourne. `python -m src.cli pool --fenetre 21` les deplace-t-il ? "
            "`--exclure` dit si c'est la fenetre ou la cohorte."
        )
        return

    colonnes = {"publie (f42, 62 ath.)": publie,
                "controle (f42, 60 ath.)": controle,
                "fenetre reduite (f21, 60 ath.)": reduit}
    lignes = []
    for nom in ("tau1", "tau2", "mu_k_net", "sigma_k_net"):
        ligne = {"parametre": nom}
        for etiquette, table in colonnes.items():
            sel = table.loc[table["parametre"] == nom]
            ligne[etiquette] = (
                f"{sel['mean'].iloc[0]:+.3f} "
                f"[{sel['hdi_3%'].iloc[0]:+.3f} · {sel['hdi_97%'].iloc[0]:+.3f}]"
                if not sel.empty else "—"
            )
        lignes.append(ligne)
    st.dataframe(pd.DataFrame(lignes), width="stretch", hide_index=True)
    st.warning(
        "Fenetre d'enveloppe divisee par deux : tau1 perd 14,4 % et tau2 "
        "18,9 %, soit une elasticite de 0,22 et 0,30 — zero pour de la "
        "physiologie pure, un pour un pur apprentissage de la fenetre. Les "
        "DEUX hypotheses sont fausses, et environ un quart de ces constantes "
        "est du protocole. La colonne de controle est ce qui le prouve : a "
        "fenetre constante, retirer les deux athletes que la fenetre reduite "
        "perd ne deplace rien."
    )


def _cibles(table: pd.DataFrame) -> list[str]:
    """
    Les cibles presentes dans un artefact, ou une seule anonyme.

    Les artefacts produits avant que `cli evaluate` marque ses cibles n'ont
    pas de colonne `cible`. Rendre `[""]` dans ce cas laisse le tableau de
    bord les afficher sans titre plutot que de planter sur un ancien fichier.
    """
    if "cible" not in table.columns:
        return [""]
    return list(dict.fromkeys(table["cible"].astype(str)))


def _missing(label: str, command: str) -> None:
    """Dit quelle commande produit l'artefact manquant."""
    st.info(f"{label} — artefact absent.")
    st.code(command, language="bash")


def empty_state() -> None:
    st.title("Digital Twin — Endurance")
    st.warning("Aucune donnee ingeree.")
    st.markdown(
        "Ce tableau de bord ne lit que des donnees reelles et n'en fabrique "
        "aucune. Pour le remplir :"
    )
    st.code(
        "python -m src.cli fetch-athletes --critere course --keep 3"
        "   # choisir un athlete\n"
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
        ["Charge", "Performance", "Seances", "Diagnostics", "Mesures"],
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

# ── page Mesures ──────────────────────────────────────────────────────────

elif page == "Mesures":
    st.header("Ce que la mesure a etabli")
    st.caption(
        "Lit les artefacts deja produits, sans rien recalculer. Un intervalle "
        "qui contient zero se lit comme une absence de preuve, pas comme une "
        "preuve d'absence — et les deux sont distingues ici."
    )
    m = load_measurements()
    (t_mat, t_disc, t_duree, t_plancher, t_cohorte, t_pool,
     t_regles, t_qui) = st.tabs(
        ["Maturite", "Appariement", "Par duree", "Plancher", "Cohorte",
         "Mise en commun", "Regles du plan", "Qui repond"])

    # -- maturite ---------------------------------------------------------
    with t_mat:
        st.subheader("Ce que le systeme peut affirmer aujourd'hui")
        lowered = sessions["sport"].astype(str).str.lower()
        bike = sessions[lowered.str.startswith(BIKE_SPORTS)]
        power = (bike[["date", BIKE_TARGET_COLUMN]].dropna()
                 .rename(columns={BIKE_TARGET_COLUMN: "pi"})
                 if BIKE_TARGET_COLUMN in sessions.columns else pd.DataFrame())
        readiness = assess_readiness(daily, power, cohort=m.get("cohorte"))

        done = len(readiness.available)
        total = len(readiness.capabilities)
        st.metric("Capacites disponibles", f"{done} / {total}")
        st.progress(done / total if total else 0.0)
        for capability in readiness.capabilities:
            if capability.available:
                st.markdown(f"**oui** — {capability.name} · {capability.detail}")
            else:
                st.markdown(
                    f":grey[**non** — {capability.name} · {capability.detail}]"
                    + (f"  \n:orange[manque : {capability.missing}]"
                       if capability.missing else "")
                )
        if readiness.prior and not readiness.capability(
                "verdict de dose-reponse").available:
            st.info(readiness.prior)
        st.caption(
            "Le verdict de dose-reponse demande 180 jours de contexte puis huit "
            "plis de 28 jours. Avant cela le systeme ne dit rien de cet athlete, "
            "et le dire vaut mieux que se taire."
        )

    # -- appariement par discipline ---------------------------------------
    with t_disc:
        st.subheader("Apparier la charge a la discipline de la cible")
        resultats, ablation = m.get("resultats"), m.get("ablation")
        if resultats is None:
            _missing("Resultats d'evaluation", "python -m src.cli evaluate")
        else:
            protocoles = list(dict.fromkeys(resultats["protocol"]))
            choisi = st.selectbox("Protocole", protocoles,
                                  index=len(protocoles) - 1)
            vue = resultats[resultats["protocol"] == choisi]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=vue["name"], y=vue["r2_in"],
                                 name="ajustement", marker_color=MUTED))
            fig.add_trace(go.Bar(x=vue["name"], y=vue["r2_out"],
                                 name="hors echantillon", marker_color=ACCENT))
            fig.add_hline(y=0, line_color="#555")
            fig.update_layout(barmode="group", xaxis_tickangle=-30)
            st.plotly_chart(_layout(fig, "R²", 420), width="stretch")
            st.dataframe(
                vue[["name", "r2_in", "r2_out", "gap", "folds"]].rename(columns={
                    "name": "variante", "r2_in": "R² ajustement",
                    "r2_out": "R² hors echantillon", "gap": "ecart",
                    "folds": "plis"}).style.format({
                        "R² ajustement": "{:+.4f}",
                        "R² hors echantillon": "{:+.4f}", "ecart": "{:+.4f}"}),
                width="stretch", hide_index=True)
            st.caption(
                "La colonne plis compte les decoupages effectivement couverts : "
                "un modele dont la tete est entrainee sur un horizon court "
                "echoue sur les plis longs, et sa ligne porte alors sur moins "
                "de points que les autres."
            )
        if ablation is not None and not ablation.empty:
            st.markdown("**Comparaisons appariees** — c'est l'intervalle qui decide.")
            # Les deux cibles sont affichees cote a cote, et non derriere un
            # selecteur : la cible velo porte les seuls gains du depot dont
            # l'intervalle exclut zero, et la cible course le resultat nul le
            # plus cite. Cacher l'une des deux fausse la lecture de l'autre.
            for cible in _cibles(ablation):
                vue_c = (ablation[ablation["cible"] == cible]
                         if "cible" in ablation.columns else ablation)
                if cible:
                    st.markdown(f"Cible **{cible}**")
                st.dataframe(
                    vue_c[["variante", "reference", "skill", "ci_low", "ci_high",
                           "p_a_better"]].rename(columns={
                        "skill": "gain", "ci_low": "IC bas", "ci_high": "IC haut",
                        "p_a_better": "P(gain)"}).style.format({
                            "gain": "{:+.4f}", "IC bas": "{:+.4f}",
                            "IC haut": "{:+.4f}", "P(gain)": "{:.3f}"}),
                    width="stretch", hide_index=True)

    # -- par duree d'effort -----------------------------------------------
    with t_duree:
        st.subheader("Une enveloppe par duree d'effort")
        durees = m.get("durees")
        if durees is None:
            _missing("Protocole par duree", "python -m src.cli evaluate")
        else:
            gardees = durees[durees["variante"] != "ecartee"]
            if gardees.empty:
                st.warning("Aucune duree retenue.")
            else:
                fig = go.Figure()
                for variante in dict.fromkeys(gardees["variante"]):
                    serie = gardees[gardees["variante"] == variante]
                    fig.add_trace(go.Scatter(
                        x=serie["duration_s"], y=serie["r2_out"], name=variante,
                        mode="lines+markers"))
                fig.add_hline(y=0, line_color="#555")
                fig.update_xaxes(title="duree de l'effort (s)")
                st.plotly_chart(_layout(fig, "R² hors echantillon", 400),
                                width="stretch")
                st.caption(
                    "La persistance s'effondre quand l'effort s'allonge, et "
                    "c'est le seul endroit ou l'ecart avec le modele de charge "
                    "cesse d'exclure zero — non parce que la charge progresse, "
                    "mais parce que la reference faiblit."
                )
            ecartees = durees[durees["variante"] == "ecartee"]
            for _, ligne in ecartees.iterrows():
                st.caption(
                    f":grey[{int(ligne['duration_s'])} s ecartee — "
                    f"{int(ligne['n_obs'])} observations, trop peu pour un R².]")
            paires = m.get("paires_durees")
            if paires is not None and not paires.empty:
                st.markdown(
                    "**Charge specifique contre persistance, duree par duree** "
                    "— un ecart de R² lu sur la courbe ne dit pas s'il depasse "
                    "le bruit ; l'intervalle, lui, le dit.")
                st.dataframe(
                    paires[["duration_s", "skill", "ci_low", "ci_high",
                            "p_a_better"]].rename(columns={
                        "duration_s": "duree (s)", "skill": "gain",
                        "ci_low": "IC bas", "ci_high": "IC haut",
                        "p_a_better": "P(gain)"}).style.format({
                            "gain": "{:+.4f}", "IC bas": "{:+.4f}",
                            "IC haut": "{:+.4f}", "P(gain)": "{:.3f}"}),
                    width="stretch", hide_index=True)

    # -- plancher de detectabilite ----------------------------------------
    with t_plancher:
        st.subheader("Quel effet ce protocole saurait voir")
        planchers = m.get("planchers")
        if planchers is None:
            _missing("Plancher de detectabilite", "python -m src.cli evaluate")
        else:
            # Un plancher qualifie un PROTOCOLE, pas un athlete : celui de la
            # cible course vaut 2 sigma contre la persistance, celui de la
            # cible velo 1 sigma. Afficher les deux est la seule facon de voir
            # que la comparaison la plus citee du depot est aussi la moins
            # puissante.
            for cible in _cibles(planchers):
                vue_p = (planchers[planchers["cible"] == cible]
                         if "cible" in planchers.columns else planchers)
                if cible:
                    st.markdown(f"#### Cible {cible}")
                fig = go.Figure()
                for reference in dict.fromkeys(vue_p["reference"]):
                    serie = vue_p[vue_p["reference"] == reference]
                    fig.add_trace(go.Scatter(x=serie["delta"], y=serie["skill"],
                                             name=f"contre {reference}",
                                             mode="lines+markers"))
                fig.add_hline(y=0, line_color="#555")
                fig.update_xaxes(title="effet injecte (ecarts-types d'enveloppe)")
                st.plotly_chart(_layout(fig, "gain apparie", 380),
                                width="stretch", key=f"plancher-{cible}")
                vus = vue_p[vue_p["detecte"]]
                for reference in dict.fromkeys(vue_p["reference"]):
                    atteint = vus[vus["reference"] == reference]["delta"]
                    if atteint.empty:
                        st.markdown(f"Contre **{reference}** : aucun plancher "
                                    "atteint.")
                    elif float(atteint.min()) == 0.0:
                        st.markdown(f"Contre **{reference}** : effet deja "
                                    "detecte **sans injection** — la charge "
                                    "reelle suffit.")
                    else:
                        st.markdown(f"Contre **{reference}** : plancher a "
                                    f"**{float(atteint.min()):.2f} ecart-type**.")
            st.caption(
                "Un effet SYNTHETIQUE d'amplitude connue est ajoute a "
                "l'enveloppe reelle, puis le protocole est repasse inchange. "
                "Le plancher est optimiste : le signal injecte a exactement la "
                "forme que le modele ajuste."
            )

    # -- qui repond -------------------------------------------------------
    with t_qui:
        st.subheader("Qu'est-ce qui distingue un repondeur ?")
        qui = m.get("repondeurs")
        if qui is None:
            _missing("Predicteurs de la reponse", "python -m src.cli responders")
        else:
            testes = qui[qui["ecarte"].isna()] if "ecarte" in qui else qui
            fig = go.Figure()
            fig.add_trace(go.Bar(y=testes["predicteur"], x=testes["rho"],
                                 orientation="h", marker_color=ACCENT,
                                 name="correlation de rang"))
            # Le rho que l'echantillon saurait distinguer du hasard, de part et
            # d'autre : sans ces deux bornes, un rho de 0,24 se lit comme un
            # resultat alors qu'il est sous le seuil.
            for signe in (1, -1):
                fig.add_trace(go.Scatter(
                    y=testes["predicteur"], x=signe * testes["rho_detectable"],
                    mode="markers", marker={"symbol": "line-ns-open",
                                            "size": 14, "color": MUTED},
                    name="seuil detectable" if signe == 1 else None,
                    showlegend=signe == 1))
            fig.add_vline(x=0, line_color="#555")
            fig.update_xaxes(title="correlation de rang avec le gain de charge")
            st.plotly_chart(_layout(fig, "", 40 * len(testes) + 120),
                            width="stretch")
            st.dataframe(
                testes[["predicteur", "n", "rho", "p", "rho_detectable",
                        "significatif"]].rename(columns={
                    "rho_detectable": "rho detectable"}).style.format({
                        "rho": "{:+.3f}", "p": "{:.4f}",
                        "rho detectable": "{:.3f}"}),
                width="stretch", hide_index=True)
            if "ecarte" in qui:
                for _, ligne in qui[qui["ecarte"].notna()].iterrows():
                    st.caption(f":grey[{ligne['predicteur']} ecarte — "
                               f"{ligne['ecarte']}.]")
            st.caption(
                "Aucun predicteur ne franchit son seuil, et aucune famille ne "
                "separe les repondeurs hors echantillon. La correlation la "
                "plus forte reste la longueur d'historique — qui n'est pas une "
                "variable connue a l'inscription, donc ne resout pas le "
                "demarrage a froid."
            )

    # -- regles du planificateur ------------------------------------------
    with t_regles:
        st.subheader("Ce que les bornes du planificateur valent face aux donnees")
        regles = m.get("regles")
        if regles is None:
            _missing("Validation des bornes du plan",
                     "python -m src.cli validate-plan")
        else:
            st.caption(
                "Chaque borne est jugee sur DEUX statistiques et DEUX regimes. "
                "Le meilleur effort croit avec le nombre de sorties, donc une "
                "borne qui ne tient que sur lui tient sur un artefact ; "
                "l'effort median y est insensible. Et une semaine a ACWR eleve "
                "est souvent une reprise apres coupure, dont la fenetre de "
                "reference est deprimee — d'ou le regime restreint aux semaines "
                "precedees d'un entrainement regulier."
            )
            couleur = {"soutenue": "#2e7d32", "contredite": "#c62828",
                       "indeterminable": MUTED}
            fig = go.Figure()
            for verdict in dict.fromkeys(regles["verdict"]):
                vue_r = regles[regles["verdict"] == verdict]
                fig.add_trace(go.Bar(
                    y=vue_r["regle"] + " · " + vue_r["regime"]
                    + " · " + vue_r["statistique"],
                    x=vue_r["ecart_moyen"], name=verdict, orientation="h",
                    marker_color=couleur.get(verdict, MUTED)))
            fig.add_vline(x=0, line_color="#555")
            fig.update_xaxes(title="ecart d'issue respect - violation "
                                   "(ecarts-types de puissance)")
            st.plotly_chart(_layout(fig, "", 40 * len(regles) + 120),
                            width="stretch")
            st.dataframe(
                regles[["regle", "regime", "statistique", "athletes",
                        "ecart_moyen", "en_faveur", "p", "plancher",
                        "verdict"]].rename(columns={
                    "ecart_moyen": "ecart", "en_faveur": "en faveur"}).style.format({
                        "ecart": "{:+.3f}", "p": "{:.4f}",
                        "plancher": "{:.2f}"}),
                width="stretch", hide_index=True)
            contredites = regles[regles["verdict"] == "contredite"]
            if not contredites.empty:
                st.warning(
                    "Borne(s) contredite(s) : "
                    + ", ".join(dict.fromkeys(contredites["regle"]))
                    + ". Le planificateur les applique encore — lire la colonne "
                      "regime avant de conclure."
                )

    # -- cohorte ----------------------------------------------------------
    with t_cohorte:
        st.subheader("Un resultat sur un athlete est une anecdote")
        cohorte = m.get("cohorte")
        if cohorte is None:
            _missing("Replication sur la cohorte", "python -m src.cli cohort")
        else:
            contre_moyenne = cohorte[cohorte["reference"] == "moyenne"].copy()
            contre_moyenne = contre_moyenne.sort_values("skill")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=contre_moyenne["skill"],
                y=contre_moyenne["uuid"].str.slice(0, 8),
                error_x={"type": "data", "symmetric": False,
                         "array": contre_moyenne["ci_high"] - contre_moyenne["skill"],
                         "arrayminus": contre_moyenne["skill"] - contre_moyenne["ci_low"]},
                mode="markers", marker={"color": ACCENT, "size": 9},
                name="gain contre la moyenne"))
            fig.add_vline(x=0, line_color="#555")
            fig.update_xaxes(title="gain apparie, IC 95 %")
            st.plotly_chart(_layout(fig, "", 420), width="stretch")

            detectes = int((contre_moyenne["ci_low"] > 0).sum())
            contre = int((contre_moyenne["ci_high"] < 0).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Athletes", len(contre_moyenne))
            c2.metric("IC excluant zero, en faveur", detectes)
            c3.metric("IC excluant zero, contre", contre)

            contraste = detection_contrast(cohorte)
            if not contraste.empty:
                st.markdown("**Heterogeneite reelle, ou defaut de mesure ?**")
                st.dataframe(contraste.style.format({
                    "gain_median": "{:+.4f}", "largeur_ic_median": "{:.3f}",
                    "puissances_median": "{:.0f}", "plis_median": "{:.0f}"}),
                    width="stretch", hide_index=True)
                st.caption(
                    "Si l'effet etait universel mais noye, les estimations "
                    "ponctuelles des non detectes resteraient positives. "
                    "Centrees sur zero, elles disent que l'effet n'existe que "
                    "chez certains."
                )

    # -- mise en commun ---------------------------------------------------
    with t_pool:
        st.subheader("Mise en commun hierarchique des athletes")
        pooling, effets = m.get("pooling"), m.get("effets")
        if pooling is None:
            _missing("Ajustement hierarchique",
                     "python -m src.cli pool      # des dizaines de minutes")
        else:
            st.dataframe(pooling, width="stretch", hide_index=True)
            st.caption(
                "Les constantes de temps sont PARTAGEES : c'est le partage qui "
                "les rend identifiables, la ou un athlete seul ne resserre son "
                "prior que de 41 %."
            )
            _fenetre_comparee(pooling, m.get("pooling_n60"), m.get("pooling_f21"))
        if effets is not None and not effets.empty:
            ordonne = effets.sort_values("k_net_moyen")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ordonne["k_net_moyen"], y=ordonne["athlete"],
                error_x={"type": "data", "symmetric": False,
                         "array": ordonne["hdi_haut"] - ordonne["k_net_moyen"],
                         "arrayminus": ordonne["k_net_moyen"] - ordonne["hdi_bas"]},
                mode="markers", marker={"color": ACCENT, "size": 9},
                name="effet net"))
            fig.add_vline(x=0, line_color="#555")
            fig.update_xaxes(title="effet net, HDI 94 %")
            st.plotly_chart(_layout(fig, "", 420), width="stretch")
            st.warning(
                "Cet ajustement est DANS l'echantillon. Le meme modele, sur les "
                "memes athletes, explique le passe chez presque tous et n'en "
                "prevoit la suite que chez un tiers : c'est la signature du "
                "surapprentissage, et c'est pourquoi l'onglet Cohorte compte "
                "davantage que celui-ci."
            )
