import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-top: 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_predictions():
    conn = sqlite3.connect('data/engine_logbook.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df

@st.cache_data
def load_clean_data():
    return pd.read_csv('data/processed/clean_data.csv')

predictions_df = load_predictions()
clean_df = load_clean_data()

COST_SAVINGS_PER_CATCH = 50000

# ==================== HEADER ====================
st.markdown('<p class="main-header">🔧 Predictive Maintenance Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Driven Jet Engine Health Monitoring System — NASA C-MAPSS Dataset</p>', unsafe_allow_html=True)
st.divider()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📊 About This System")
    st.markdown("""
    This dashboard visualizes predictions from a dual-model
    predictive maintenance pipeline:

    - **XGBoost** — failure classification
    - **PyTorch** — remaining useful life (RUL) regression
    - **RAG (FAISS + Sentence Transformers)** — automated
      repair recommendations retrieved from technical manuals

    Predictions are logged to a SQL database and visualized
    here in real time.
    """)
    st.divider()
    st.caption("Final Year Project — Predictive Maintenance AI")
    st.caption(f"Total logged predictions: {len(predictions_df)}")

# ==================== TOP KPI ROW ====================
col1, col2, col3, col4 = st.columns(4)

total_engines = predictions_df['machine_id'].nunique()
failing_count = (predictions_df['status'] == 'Failing').sum()
healthy_count = (predictions_df['status'] == 'Healthy').sum()
total_savings = failing_count * COST_SAVINGS_PER_CATCH

with col1:
    st.metric("Total Engines Monitored", total_engines)
with col2:
    st.metric("Critical Alerts", failing_count, delta=f"{failing_count} flagged", delta_color="inverse")
with col3:
    st.metric("Healthy Readings", healthy_count)
with col4:
    st.metric("💰 Estimated Cost Savings", f"₹{total_savings:,}")

st.divider()

# ==================== MAIN LAYOUT: TABS ====================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Fleet Overview", "🔍 Engine Drill-Down", "🚨 Active Alerts", "🧠 Model Insights"])

# ---------- TAB 1: FLEET OVERVIEW ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fleet Status Distribution")
        status_counts = predictions_df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig = px.pie(
            status_counts, values='Count', names='Status',
            color='Status',
            color_discrete_map={'Failing': '#EF4444', 'Healthy': '#22C55E'},
            hole=0.5
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confidence Distribution")
        fig2 = px.histogram(
            predictions_df, x='confidence', color='status',
            color_discrete_map={'Failing': '#EF4444', 'Healthy': '#22C55E'},
            nbins=20, barmode='overlay', opacity=0.75
        )
        fig2.update_layout(xaxis_title="Model Confidence", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Predicted Remaining Useful Life (RUL) Across Fleet")
    fig3 = px.scatter(
        predictions_df, x='machine_id', y='predicted_rul', color='status',
        color_discrete_map={'Failing': '#EF4444', 'Healthy': '#22C55E'},
        size='confidence', hover_data=['cycle', 'timestamp']
    )
    fig3.update_layout(xaxis_title="Machine ID", yaxis_title="Predicted RUL (cycles)")
    fig3.add_hline(y=30, line_dash="dash", line_color="orange",
                    annotation_text="Failure Threshold (RUL=30)")
    st.plotly_chart(fig3, use_container_width=True)

# ---------- TAB 2: ENGINE DRILL-DOWN ----------
with tab2:
    st.subheader("Individual Engine Inspection")
    selected_engine = st.selectbox(
        "Select Machine ID", sorted(predictions_df['machine_id'].unique())
    )

    engine_preds = predictions_df[predictions_df['machine_id'] == selected_engine].sort_values('cycle')
    engine_sensors = clean_df[clean_df['unit_number'] == selected_engine].sort_values('time_in_cycles')

    st.markdown("**Select which recorded checkpoint to inspect:**")
    cycle_options = engine_preds['cycle'].tolist()
    selected_cycle = st.select_slider("Cycle", options=cycle_options, value=cycle_options[0])

    latest = engine_preds[engine_preds['cycle'] == selected_cycle].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Status at This Cycle", latest['status'])
    c2.metric("Confidence", f"{latest['confidence']:.1%}")
    c3.metric("Predicted RUL", f"{latest['predicted_rul']:.1f} cycles")

    st.markdown("**Sensor Trend Over Operational Cycles**")
    sensor_choice = st.selectbox(
        "Select sensor to visualize",
        [c for c in clean_df.columns if c.startswith('sensor_')]
    )
    fig4 = px.line(
        engine_sensors, x='time_in_cycles', y=sensor_choice,
        title=f"{sensor_choice} — Machine {selected_engine}"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Logged Prediction History**")
    st.dataframe(engine_preds[['cycle', 'status', 'confidence', 'predicted_rul', 'timestamp']],
                 use_container_width=True, hide_index=True)
# ---------- TAB 3: ACTIVE ALERTS ----------
with tab3:
    st.subheader("🚨 Engines Requiring Immediate Attention")
    alerts = predictions_df[predictions_df['status'] == 'Failing'].sort_values('predicted_rul')

    if len(alerts) == 0:
        st.success("No critical alerts at this time.")
    else:
        st.warning(f"{len(alerts)} prediction(s) flagged as Failing across the fleet.")
        display_alerts = alerts[['machine_id', 'cycle', 'confidence', 'predicted_rul', 'timestamp']].copy()
        display_alerts['confidence'] = display_alerts['confidence'].apply(lambda x: f"{x:.1%}")
        display_alerts['predicted_rul'] = display_alerts['predicted_rul'].apply(lambda x: f"{x:.1f} cycles")
        display_alerts.columns = ['Machine ID', 'Cycle', 'Confidence', 'Predicted RUL', 'Logged At']
        st.dataframe(display_alerts, use_container_width=True, hide_index=True)

# ---------- TAB 4: MODEL INSIGHTS ----------
with tab4:
    st.subheader("Model Explainability (SHAP)")
    shap_path = 'reports/figures/shap_summary.png'
    if os.path.exists(shap_path):
        image = Image.open(shap_path)
        st.image(image, caption="XGBoost Feature Importance (SHAP values)", use_container_width=True)
        st.caption(
            "The engineered feature `sensor_2_roll_mean_5` (a 5-cycle rolling average) "
            "ranks as the single most important predictor, validating the feature "
            "engineering performed during data preprocessing."
        )
    else:
        st.info("SHAP summary image not found. Run the SHAP analysis in 02_model_training.ipynb first.")

    st.divider()
    st.subheader("Model Performance Summary")
    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.markdown("**XGBoost Classifier (Test Set)**")
        st.markdown("""
        | Metric | Score |
        |---|---|
        | Precision | 0.81 |
        | Recall | 0.90 |
        | F1-Score | 0.85 |
        | ROC-AUC | 0.989 |
        """)

    with perf_col2:
        st.markdown("**PyTorch RUL Regressor (Test Set)**")
        st.markdown("""
        | Metric | Score |
        |---|---|
        | RMSE | 22.51 cycles |
        | MAE | 18.19 cycles |
        """)

st.divider()
st.caption("Built with Streamlit • XGBoost • PyTorch • FAISS • SHAP — Predictive Maintenance AI Final Year Project")