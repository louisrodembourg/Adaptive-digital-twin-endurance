"""
UI Layer — Dashboard Streamlit : Digital Twin Semi-Marathon

Interface de monitoring et de validation humaine du système.

Pages :
  1. Overview     : Jauge P(1h18), courbes Fitness/Fatigue/TSB, compte à rebours
  2. Plan Semaine : Calendrier + formulaire contexte quotidien + XAI
  3. Analyse      : Graphes multi-signaux d'une séance (HR, vitesse, puissance)
  4. Tendances    : EF_spec, TRIMP cumulatif, historique rMSSD

Usage :
    streamlit run app/main_dashboard.py
"""

import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ajouter le répertoire racine au path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.layer_a_signal.context_ingestion import (
    DailyContext, DailyContextStore, create_today_context
)
from src.layer_b_twin.state_estimator import StateEstimator, bootstrap_trimp_history
from src.layer_c_planner.safety_valve import check_safety_constraints, format_constraints_summary
from src.layer_c_planner.llm_context_parser import parse_context_to_constraints, SoftConstraints
from src.layer_c_planner.optimizer import optimize_week, generate_taper_plan

# ─────────────────────────────────────────────────────────────
#  CONFIG PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Twin — Semi-Marathon",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2130; border-radius: 10px; padding: 16px;
        text-align: center; border: 1px solid #2d3250;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #7c83fd; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .alert-box {
        background: #2d1e1e; border-left: 4px solid #e05c5c;
        padding: 10px 14px; border-radius: 4px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CACHE — Estimateur et courbes
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_estimator() -> StateEstimator:
    est = StateEstimator()
    if est._trimp_df.empty:
        trimp = bootstrap_trimp_history(weeks=12, base_trimp_per_week=220)
        for i, t in enumerate(trimp):
            if t > 0:
                session_date = date.today() - timedelta(days=len(trimp) - i)
                est.add_session(float(t), session_date=session_date)
    return est


@st.cache_data(ttl=300)
def get_curves(n_days: int = 120) -> pd.DataFrame:
    return get_estimator().update_model(n_days=n_days)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏃 Digital Twin")
    st.caption("Semi-Marathon 1h22 → 1h18")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Plan Semaine", "Analyse Séance", "Tendances"],
    )
    st.divider()

    race_date = st.date_input(
        "📅 Date de la course",
        value=date.today() + timedelta(days=84),
        min_value=date.today(),
    )
    days_to_race = (race_date - date.today()).days
    st.metric("Jours jusqu'à la course", days_to_race)


# ─────────────────────────────────────────────────────────────
#  PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Overview — État de l'athlète")

    curves = get_curves()
    state = get_estimator().get_current_state()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        prob = (state.prob_success_1h18 or 0) * 100
        st.metric("P(Réussir 1h18)", f"{prob:.0f}%")
    with col2:
        st.metric("Temps estimé", state.estimated_time_str)
    with col3:
        st.metric(
            "TSB (Forme)", f"{state.tsb:.1f}",
            delta="Frais" if state.tsb > 0 else "Fatigué",
            delta_color="normal" if state.tsb > 0 else "inverse",
        )
    with col4:
        st.metric("Fitness", f"{state.fitness:.1f}")

    if state.is_overreaching:
        st.markdown(
            '<div class="alert-box">⚠️ <b>ALERTE SUR-ENTRAÎNEMENT</b> — '
            f'TSB={state.tsb:.1f} < -30. Repos forcé.</div>',
            unsafe_allow_html=True
        )
    if state.needs_calibration:
        st.markdown(
            '<div class="alert-box">🔬 <b>CALIBRATION REQUISE</b> — '
            'Séance test 5km à planifier.</div>',
            unsafe_allow_html=True
        )

    st.divider()

    col_gauge, col_curves = st.columns([1, 2])

    with col_gauge:
        st.subheader("Probabilité 1h18")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': '%', 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#7c83fd"},
                'steps': [
                    {'range': [0, 30], 'color': "#3d1e1e"},
                    {'range': [30, 60], 'color': "#3d3d1e"},
                    {'range': [60, 100], 'color': "#1e3d1e"},
                ],
                'threshold': {
                    'line': {'color': "#5ce05c", 'width': 3},
                    'thickness': 0.75, 'value': 80,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)', font={'color': '#ffffff'},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{days_to_race}j</div>
            <div class="metric-label">avant la course<br>{race_date.strftime('%d %b %Y')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_curves:
        st.subheader("Fitness · Fatigue · TSB")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curves['day'], y=curves['fitness'],
            name='Fitness', line=dict(color='#5ce05c', width=2)))
        fig.add_trace(go.Scatter(x=curves['day'], y=curves['fatigue'],
            name='Fatigue', line=dict(color='#e05c5c', width=2)))
        fig.add_trace(go.Scatter(x=curves['day'], y=curves['tsb'],
            name='TSB', line=dict(color='#7c83fd', width=2),
            fill='tozeroy', fillcolor='rgba(124,131,253,0.1)'))
        fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.3)')
        fig.add_hrect(y0=-100, y1=-30, fillcolor='rgba(224,92,92,0.1)',
                      line_width=0, annotation_text="Sur-entraîn.")
        fig.update_layout(
            height=300, margin=dict(t=10, b=30, l=40, r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,1)',
            font={'color': '#ffffff'}, legend=dict(orientation='h', y=1.05),
            xaxis=dict(title='Jours', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Courbe performance
    st.subheader("Performance Normalisée")
    target_perf = 4920 / 4680
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=curves['day'], y=curves['perf_norm'],
        name='Perf estimée', line=dict(color='#ffd700', width=2)))
    fig_perf.add_hline(y=target_perf, line_dash='dot', line_color='#5ce05c',
                       annotation_text="Objectif 1h18")
    fig_perf.add_hline(y=1.0, line_dash='dash', line_color='rgba(255,255,255,0.3)',
                       annotation_text="Record 1h22")
    fig_perf.update_layout(
        height=250, margin=dict(t=10, b=30, l=40, r=80),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,1)',
        font={'color': '#ffffff'},
        xaxis=dict(title='Jours', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Perf norm.', gridcolor='rgba(255,255,255,0.05)'),
    )
    st.plotly_chart(fig_perf, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  PAGE 2 — PLAN SEMAINE
# ─────────────────────────────────────────────────────────────

elif page == "Plan Semaine":
    st.title("Plan de la Semaine")

    col_form, col_plan = st.columns([1, 2])

    with col_form:
        st.subheader("Contexte du jour")
        with st.form("daily_context_form"):
            rmssd = st.number_input("rMSSD (ms)", min_value=10.0, max_value=150.0,
                                    value=55.0, step=1.0)
            sleep_h = st.slider("Sommeil (heures)", 4.0, 10.0, 7.5, 0.5)
            sleep_q = st.select_slider(
                "Qualité sommeil", options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: ["Très mauvaise","Mauvaise","Moyenne","Bonne","Excellente"][x-1],
            )
            rpe_prev = st.slider("RPE hier (Borg 1-10)", 0.0, 10.0, 0.0, 0.5)
            stress = st.select_slider(
                "Stress", options=[1, 2, 3, 4, 5], value=2,
                format_func=lambda x: ["Minimal","Faible","Moyen","Élevé","Très élevé"][x-1],
            )
            notes = st.text_area("Notes libres (IA)", height=80,
                                 placeholder="Ex: jambes lourdes, week-end chargé...")
            submitted = st.form_submit_button("Enregistrer & Générer le plan", type="primary")

        if submitted:
            ctx = create_today_context(
                rmssd_ms=rmssd if rmssd > 0 else None,
                sleep_hours=sleep_h, sleep_quality=sleep_q,
                rpe_yesterday=rpe_prev if rpe_prev > 0 else None,
                stress_level=stress, notes=notes, save=True,
            )
            st.session_state['daily_context'] = ctx
            st.success(f"Enregistré — Score fatigue : {ctx.fatigue_score:.0%}")

    with col_plan:
        st.subheader("Plan Optimal")
        ctx = st.session_state.get('daily_context')
        state = get_estimator().get_current_state()

        hard_constraints = check_safety_constraints(
            rmssd_ms=ctx.rmssd_ms if ctx else None,
            tsb=state.tsb,
            needs_calibration=state.needs_calibration,
            race_date=race_date,
        )

        soft_constraints = SoftConstraints()
        if ctx and ctx.notes:
            with st.spinner("Analyse IA..."):
                soft_constraints = parse_context_to_constraints(
                    ctx.notes, use_heuristic_fallback=True)
            if soft_constraints.is_restrictive():
                st.info(f"🤖 {soft_constraints.summary()}")

        if hard_constraints:
            st.warning(format_constraints_summary(hard_constraints))

        is_taper = days_to_race <= 14
        if is_taper:
            week_plan = generate_taper_plan(race_date=race_date,
                                            current_fitness=state.fitness)
        else:
            trimp_arr, _ = get_estimator().get_trimp_array(n_days=14)
            prev_trimp = float(trimp_arr[:7].sum()) or 250.0
            week_plan = optimize_week(
                hard_constraints=hard_constraints,
                soft_constraints=soft_constraints,
                previous_week_trimp=prev_trimp,
                race_date=race_date,
            )

        type_colors = {
            "REST": "#555", "RECOVERY": "#2196f3", "ENDURANCE": "#4caf50",
            "THRESHOLD": "#ff9800", "VMA": "#f44336",
            "SPECIFIC": "#9c27b0", "STRENGTH": "#607d8b", "CALIBRATION": "#00bcd4",
        }

        cols = st.columns(7)
        for day_plan, col in zip(week_plan.days, cols):
            color = type_colors.get(day_plan.session_type, "#555")
            with col:
                st.markdown(f"""
                <div style="background:{color}22;border:1px solid {color};
                    border-radius:8px;padding:8px;text-align:center;min-height:110px;">
                    <b style="color:{color}">{day_plan.day_name[:3]}</b><br>
                    <small>{day_plan.date.strftime('%d/%m')}</small><br>
                    <b>{day_plan.session_type}</b><br>
                    <small>TRIMP≈{day_plan.trimp_estimated:.0f}</small>
                </div>""", unsafe_allow_html=True)

        st.markdown(
            f"**TRIMP total :** {week_plan.total_trimp:.0f} | "
            f"**Séances :** {week_plan.total_sessions} | "
            f"**Spécificité :** {week_plan.specificity_score:.2f}"
        )
        with st.expander("Explication des choix (XAI)"):
            st.text(week_plan.xai_report())


# ─────────────────────────────────────────────────────────────
#  PAGE 3 — ANALYSE SÉANCE
# ─────────────────────────────────────────────────────────────

elif page == "Analyse Séance":
    st.title("Analyse de Séance")

    processed_dir = ROOT / 'data' / 'processed'
    parquet_files = sorted(processed_dir.glob("records_*.parquet"), reverse=True)

    if not parquet_files:
        st.warning("Aucun fichier traité. Lancer d'abord le pipeline d'ingestion.")
        st.stop()

    selected_file = st.selectbox("Séance", parquet_files,
                                  format_func=lambda p: p.stem)
    df = pd.read_parquet(selected_file)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if 'hr' in df.columns: st.metric("FC moy.", f"{df['hr'].mean():.0f} bpm")
    with c2:
        if 'speed_kmh' in df.columns: st.metric("Vitesse", f"{df['speed_kmh'].mean():.1f} km/h")
    with c3:
        if 'power' in df.columns: st.metric("Puissance", f"{df['power'].mean():.0f} W")
    with c4:
        if 'distance_m' in df.columns: st.metric("Distance", f"{df['distance_m'].max()/1000:.1f} km")

    time_axis = np.arange(len(df)) / 60
    fig = go.Figure()
    if 'hr' in df.columns:
        fig.add_trace(go.Scatter(x=time_axis, y=df['hr'], name='FC (bpm)',
            yaxis='y1', line=dict(color='#e05c5c', width=1.5)))
    if 'speed_kmh' in df.columns:
        fig.add_trace(go.Scatter(x=time_axis, y=df['speed_kmh'], name='Vitesse (km/h)',
            yaxis='y2', line=dict(color='#5ce05c', width=1.5)))
    if 'power' in df.columns:
        fig.add_trace(go.Scatter(x=time_axis, y=df['power'], name='Puissance (W)',
            yaxis='y3', opacity=0.6, line=dict(color='#ffd700', width=1)))

    fig.update_layout(
        height=400, xaxis=dict(title='Temps (min)', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='FC', titlefont=dict(color='#e05c5c'),
                   gridcolor='rgba(255,255,255,0.05)'),
        yaxis2=dict(title='Vitesse', titlefont=dict(color='#5ce05c'),
                    overlaying='y', side='right'),
        yaxis3=dict(title='Puissance', titlefont=dict(color='#ffd700'),
                    overlaying='y', side='right', anchor='free', position=1.05),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,1)',
        font={'color': '#ffffff'}, margin=dict(t=30, b=30, l=60, r=80),
        legend=dict(orientation='h', y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    if '_flag_cleaned' in df.columns:
        n = df['_flag_cleaned'].sum()
        st.caption(f"Nettoyage v2 : {n} points corrigés ({100*n/len(df):.1f}%)")

    if 'hr' in df.columns:
        st.subheader("Distribution Zones FC")
        hr_max = 205
        bounds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        names = ["Z1 Récup", "Z2 End.", "Z3 Tempo", "Z4 Seuil", "Z5 VMA"]
        counts = [((df['hr'] >= hr_max*bounds[i]) & (df['hr'] < hr_max*bounds[i+1])).sum()
                  / len(df) * 100 for i in range(5)]
        fig_z = px.bar(x=names, y=counts,
                       color=names,
                       color_discrete_sequence=['#4caf50','#2196f3','#ff9800','#ff5722','#f44336'],
                       labels={'x': 'Zone', 'y': '% du temps'})
        fig_z.update_layout(height=250, showlegend=False,
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,1)',
                             font={'color': '#ffffff'}, margin=dict(t=10, b=30))
        st.plotly_chart(fig_z, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  PAGE 4 — TENDANCES
# ─────────────────────────────────────────────────────────────

elif page == "Tendances":
    st.title("Tendances & Progression")

    est = get_estimator()
    trimp_arr, date_range = est.get_trimp_array(n_days=84)

    st.subheader("Charge d'entraînement (TRIMP)")
    trimp_df = pd.DataFrame({
        'date': date_range,
        'trimp': trimp_arr,
        'trimp_7d': pd.Series(trimp_arr).rolling(7, min_periods=1).sum().values,
    })
    fig_trimp = go.Figure()
    fig_trimp.add_trace(go.Bar(x=trimp_df['date'], y=trimp_df['trimp'],
        name='Quotidien', marker_color='#7c83fd', opacity=0.6))
    fig_trimp.add_trace(go.Scatter(x=trimp_df['date'], y=trimp_df['trimp_7d'],
        name='7j glissant', line=dict(color='#ffd700', width=2)))
    fig_trimp.update_layout(
        height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,1)',
        font={'color': '#ffffff'}, margin=dict(t=10, b=30),
        yaxis=dict(title='TRIMP', gridcolor='rgba(255,255,255,0.05)'),
    )
    st.plotly_chart(fig_trimp, use_container_width=True)

    st.subheader("EF Spécifique (indicateur clé 1h18)")
    sessions_file = ROOT / 'data' / 'processed' / 'session_summary.parquet'
    if sessions_file.exists():
        sess = pd.read_parquet(sessions_file)
        if 'ef_specific' in sess.columns and sess['ef_specific'].notna().any():
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=sess.index, y=sess['ef_specific'],
                mode='lines+markers', line=dict(color='#5ce05c', width=2)))
            fig_ef.add_hline(y=0.027, line_dash='dot', line_color='#ffd700',
                             annotation_text="Cible 1h18")
            fig_ef.update_layout(
                height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,1)',
                font={'color': '#ffffff'}, margin=dict(t=10, b=30),
                yaxis=dict(title='EF_spec', gridcolor='rgba(255,255,255,0.05)'),
            )
            st.plotly_chart(fig_ef, use_container_width=True)
        else:
            st.info("EF_spec disponible après plusieurs séances à l'allure cible (3'40\"/km).")
    else:
        st.info("Lancer le pipeline pour générer les résumés de séances.")

    st.subheader("HRV matinal (rMSSD)")
    store = DailyContextStore()
    ctx_df = store.load_all()
    if not ctx_df.empty and 'rmssd_ms' in ctx_df.columns:
        ctx_df = ctx_df.dropna(subset=['rmssd_ms'])
        if not ctx_df.empty:
            fig_hrv = go.Figure()
            fig_hrv.add_trace(go.Scatter(
                x=pd.to_datetime(ctx_df['date']), y=ctx_df['rmssd_ms'],
                mode='lines+markers', line=dict(color='#2196f3', width=2),
                fill='tozeroy', fillcolor='rgba(33,150,243,0.1)'))
            fig_hrv.add_hline(y=50, line_dash='dash', line_color='rgba(255,165,0,0.7)',
                              annotation_text="Seuil haute intensité (50ms)")
            fig_hrv.update_layout(
                height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,1)',
                font={'color': '#ffffff'}, margin=dict(t=10, b=30),
                yaxis=dict(title='rMSSD (ms)', gridcolor='rgba(255,255,255,0.05)'),
            )
            st.plotly_chart(fig_hrv, use_container_width=True)
    else:
        st.info("Renseigner le rMSSD chaque matin dans 'Plan Semaine'.")


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("Digital Twin Semi-Marathon · PyMC · OR-Tools · NeuroKit2 · Streamlit · 1h22 → 1h18")
