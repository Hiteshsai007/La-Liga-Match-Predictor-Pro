import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from backend import (
    load_data,
    build_features_and_model,
    compute_recent_stats,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Page config & styling
# -----------------------------
st.set_page_config(
    page_title="⚽ La Liga Match Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    --card-bg: rgba(255,255,255,0.05);
    --card-glow: rgba(0,255,127,0.3);
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --accent-green: #00ff7f;
    --accent-gold: #ffd700;
    --accent-red: #ff4b4b;
    --gradient-bar: linear-gradient(90deg, #00ff7f, #ffd700, #ff4b4b);
}

.main {background: var(--bg-primary);}
.stApp {background: var(--bg-primary);}
h1, h2, h3 {color: var(--text-primary); font-weight: 800; text-shadow: 0 2px 10px rgba(0,0,0,0.5);}
h1 {font-size: clamp(2rem, 5vw, 3.5rem); background: var(--gradient-bar); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.metric-label {color: var(--text-secondary) !important;}
.metric-value {color: var(--accent-green) !important; font-weight: 800; font-size: clamp(1.2rem, 3vw, 2rem);}
.stButton > button {background: linear-gradient(45deg, #00ff7f, #00cc66); border-radius: 25px; border: none; color: white; font-weight: 600; transition: all 0.3s ease; padding: 0.75rem 2rem;}
.stButton > button:hover {transform: translateY(-3px) scale(1.02); box-shadow: 0 15px 30px rgba(0,255,127,0.4);}
.stSelectbox > div > div > div, .stSlider > div > div {background-color: var(--card-bg); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);}
[data-testid="stMetricLabel"] {font-size: clamp(0.8rem, 2vw, 1rem);}
[data-testid="stMetricValue"] {font-size: clamp(1.5rem, 4vw, 2.5rem);}
</style>
""", unsafe_allow_html=True)

plt.style.use("default")
sns.set_palette("husl")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.03); 
           border-radius: 20px; border: 1px solid rgba(0,255,127,0.2); 
           box-shadow: 0 20px 40px rgba(0,0,0,0.3); margin-bottom: 2rem;'>
    <h1 style='margin: 0;'>⚽ La Liga Match Predictor Pro</h1>
    <p style='color: var(--text-secondary); font-size: clamp(1.1rem, 2.5vw, 1.4rem); margin: 0.5rem 0;'>
        AI-Powered Predictions • Historical Analytics • Live Stats ⚡
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 🎯 **Dashboard Controls**")
    uploaded_file = st.file_uploader("📁 Upload LaLiga CSV", type=["csv"])

    if uploaded_file is not None:
        df_raw = load_data(uploaded_file)
    else:
        st.info("🧪 Using sample La Liga data (upload real CSV anytime)")
        np.random.seed(42)
        data = {
            "Season": ["2023-24"]*50 + ["2024-25"]*50,
            "Date": pd.date_range("2023-08-01", periods=100, freq="3D").strftime("%d/%m/%Y"),
            "HomeTeam": np.random.choice(
                ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia"], 100
            ),
            "AwayTeam": np.random.choice(
                ["Villarreal", "Betis", "Athletic Club", "Getafe", "Osasuna"], 100
            ),
            "FTHG": np.random.choice([0,1,2,3,4],100, p=[0.2,0.3,0.3,0.15,0.05]),
            "FTAG": np.random.choice([0,1,2,3,4],100, p=[0.25,0.35,0.25,0.1,0.05]),
            "FTR": np.random.choice(["H","D","A"],100,p=[0.45,0.25,0.3]),
        }
        df_raw = pd.DataFrame(data)

    st.success("✅ Dataset ready!")

    # -----------------------------
    # Build model
    # -----------------------------
    @st.cache_data
    def _cached_build(df_raw_):
        # Feature engineering
        df_ = df_raw_.copy()
        df_["HomeForm"] = df_.groupby("HomeTeam")["FTR"].transform(lambda x: (x=="H").rolling(5,1).mean())
        df_["AwayForm"] = df_.groupby("AwayTeam")["FTR"].transform(lambda x: (x=="A").rolling(5,1).mean())
        df_["HomeGoalsAvg"] = df_.groupby("HomeTeam")["FTHG"].transform(lambda x: x.rolling(5,1).mean())
        df_["AwayGoalsAvg"] = df_.groupby("AwayTeam")["FTAG"].transform(lambda x: x.rolling(5,1).mean())
        
        # Encode target
        target = "FTR"
        X = df_[["HomeForm","AwayForm","HomeGoalsAvg","AwayGoalsAvg"]].fillna(0)
        y = df_[target]

        # Train RandomForest
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42))
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        acc = pipeline.score(X_test, y_test)

        # Classification report and confusion matrix
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred = pipeline.predict(X_test)
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return df_, X, y, pipeline, acc, clf_report, cm

    with st.spinner("🔄 Training AI Model..."):
        df, X, y, model, acc, clf_report, cm = _cached_build(df_raw)

# -----------------------------
# Tabs
# -----------------------------
tab_pred, tab_perf, tab_history, tab_team_stats = st.tabs(
    ["🔮 Match Predictor", "📊 Model Analytics", "📜 Match History", "🏆 Team Stats"]
)

# -----------------------------
# Functions
# -----------------------------
def create_prob_chart(prob_dict, title):
    fig = go.Figure(go.Bar(
        x=list(prob_dict.keys()),
        y=list(prob_dict.values()),
        marker_color=["#00ff7f","#ffd700","#ff4b4b"]
    ))
    fig.update_layout(title=title,
                      yaxis_tickformat=".0%",
                      template="plotly_dark",
                      yaxis=dict(range=[0,1]))
    return fig

# -----------------------------
# 1️⃣ Prediction tab
# -----------------------------
with tab_pred:
    st.markdown("---")
    col1, col2 = st.columns(2)
    teams_all = sorted(set(df["HomeTeam"].unique()).union(df["AwayTeam"].unique()))
    with col1:
        st.markdown("### 🏠 Home Team")
        seasons = sorted(df["Season"].unique())
        season_input = st.selectbox("📅 Season", seasons, index=len(seasons)-1)
        home_team = st.selectbox("⚽ Home Team", teams_all, key="home_team")
    with col2:
        st.markdown("### ✈️ Away Team")
        away_team = st.selectbox("⚽ Away Team", [t for t in teams_all if t != home_team], key="away_team")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 📈 Team Form")
        home_form = st.slider("🏠 Home Form", 0.0, 1.0, 0.7, 0.01)
        home_goals_avg = st.number_input("⚽ Home Goals/Game", 0.0, 5.0, 1.8, 0.1)
    with col4:
        st.markdown("### 📉 Opponent Form")
        away_form = st.slider("✈️ Away Form", 0.0, 1.0, 0.7, 0.01)
        away_goals_avg = st.number_input("⚽ Away Goals/Game", 0.0, 5.0, 1.2, 0.1)

    if st.button("🧠 Auto-fill from History"):
        try:
            hf, af, hga, aga = compute_recent_stats(df, season_input, home_team, away_team)
            st.success(f"✅ Loaded stats\n🏠 {home_team}: Form={hf:.2f}, Goals={hga:.2f}\n✈️ {away_team}: Form={af:.2f}, Goals={aga:.2f}")
            home_form, away_form, home_goals_avg, away_goals_avg = hf, af, hga, aga
        except Exception:
            st.warning("⚠️ No recent history available")

    if st.button("🚀 Predict Match Outcome"):
        upcoming_df = pd.DataFrame([{
            "HomeForm": home_form,
            "AwayForm": away_form,
            "HomeGoalsAvg": home_goals_avg,
            "AwayGoalsAvg": away_goals_avg
        }])
        prediction = model.predict(upcoming_df)[0]
        proba = model.predict_proba(upcoming_df)[0]
        result_map = {"H": "🏠 Home Win", "D": "🤝 Draw", "A": "✈️ Away Win"}
        label_order = model.named_steps["classifier"].classes_

        st.markdown("---")
        st.subheader("📊 AI Prediction")
        st.markdown(f"### {home_team} vs {away_team} → **{result_map[prediction]}**")

        prob_dict = {result_map[lbl]: p for lbl, p in zip(label_order, proba)}
        col_hw, col_dr, col_aw = st.columns(3)
        col_hw.metric("🏠 Home Win", f"{prob_dict.get('🏠 Home Win', 0):.1%}")
        col_dr.metric("🤝 Draw", f"{prob_dict.get('🤝 Draw', 0):.1%}")
        col_aw.metric("✈️ Away Win", f"{prob_dict.get('✈️ Away Win', 0):.1%}")

        fig_pred = create_prob_chart(prob_dict, f"{home_team} vs {away_team} Probabilities")
        st.plotly_chart(fig_pred, use_container_width=True)

# -----------------------------
# 2️⃣ Performance tab
# -----------------------------
with tab_perf:
    st.markdown("### 🎯 Model Performance Overview")
    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("✅ Validation Accuracy", f"{acc:.1%}")
    with col2:
        report_df = pd.DataFrame(clf_report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("### 🔍 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["H","D","A"], yticklabels=["H","D","A"], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# -----------------------------
# 3️⃣ History tab
# -----------------------------
with tab_history:
    st.markdown("### 📜 Filter & Explore Match History")
    c1,c2,c3 = st.columns(3)
    with c1:
        season_filter = st.multiselect("📅 Seasons", sorted(df["Season"].unique()))
    with c2:
        home_filter = st.multiselect("🏠 Home Teams", sorted(df["HomeTeam"].unique()))
    with c3:
        away_filter = st.multiselect("✈️ Away Teams", sorted(df["AwayTeam"].unique()))

    df_view = df.copy()
    if season_filter: df_view = df_view[df_view["Season"].isin(season_filter)]
    if home_filter: df_view = df_view[df_view["HomeTeam"].isin(home_filter)]
    if away_filter: df_view = df_view[df_view["AwayTeam"].isin(away_filter)]

    display_cols = ["Season","Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HomeForm","AwayForm","HomeGoalsAvg","AwayGoalsAvg"]
    st.dataframe(df_view[display_cols].sort_values("Date",ascending=False), use_container_width=True, hide_index=True)

# -----------------------------
# 4️⃣ Team stats tab
# -----------------------------
with tab_team_stats:
    st.markdown("### 🏆 Team Performance Analytics")
    teams_all = sorted(set(df["HomeTeam"].unique()).union(df["AwayTeam"].unique()))
    team_selected = st.selectbox("⚽ Select Team", teams_all)

    df_team_home = df[df["HomeTeam"]==team_selected]
    df_team_away = df[df["AwayTeam"]==team_selected]
    total_matches = len(df_team_home) + len(df_team_away)

    wins = (df_team_home["FTR"]=="H").sum() + (df_team_away["FTR"]=="A").sum()
    draws = (df_team_home["FTR"]=="D").sum() + (df_team_away["FTR"]=="D").sum()
    losses = total_matches - wins - draws

    goals_for = df_team_home["FTHG"].sum() + df_team_away["FTAG"].sum()
    goals_against = df_team_home["FTAG"].sum() + df_team_away["FTHG"].sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("⚽ Matches", total_matches)
    c2.metric("🥇 Win Rate", f"{(wins/total_matches)*100:.1f}%" if total_matches else "N/A")
    c3.metric("⚽ Goals For", f"{goals_for/total_matches:.2f}" if total_matches else "N/A")
    c4.metric("🛡️ Goals Against", f"{goals_against/total_matches:.2f}" if total_matches else "N/A")

    results_data = {"🥇 Wins": wins, "🤝 Draws": draws, "😞 Losses": losses}
    fig_res = create_prob_chart(results_data, f"{team_selected} - Results")
    st.plotly_chart(fig_res, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center; color:#cbd5e1; padding:1rem;'>⚽ Built for La Liga fans • Deployed on Streamlit Cloud</div>", unsafe_allow_html=True)
