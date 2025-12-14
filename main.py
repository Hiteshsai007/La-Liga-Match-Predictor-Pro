import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from backend import load_data, compute_recent_stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="⚽ La Liga Match Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# SAFE CSS (NO SCROLL BREAK)
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

.main, .stApp {
    background: linear-gradient(135deg, #0f0f23, #1a1a2e, #16213e);
}

h1, h2, h3 {
    color: #f8fafc;
    font-weight: 800;
}

.stButton > button {
    background: linear-gradient(45deg, #00ff7f, #00cc66);
    border-radius: 25px;
    border: none;
    font-weight: 600;
}

[data-testid="stMultiSelect"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:2rem;">
<h1>⚽ La Liga Match Predictor Pro</h1>
<p style="color:#cbd5e1;">AI Predictions • Analytics • Team Stats</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar: Load data
# --------------------------------------------------
with st.sidebar:
    uploaded_file = st.file_uploader("Upload La Liga CSV", type=["csv"])

    if uploaded_file:
        df_raw = load_data(uploaded_file)
    else:
        np.random.seed(42)
        df_raw = pd.DataFrame({
            "Season": ["2024-25"] * 120,
            "Date": pd.date_range("2024-08-01", periods=120, freq="3D"),
            "HomeTeam": np.random.choice(
                ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla",
                 "Valencia", "Betis", "Villarreal", "Athletic Club"], 120
            ),
            "AwayTeam": np.random.choice(
                ["Getafe", "Osasuna", "Mallorca", "Sociedad",
                 "Alaves", "Celta", "Granada", "Las Palmas"], 120
            ),
            "FTHG": np.random.randint(0, 5, 120),
            "FTAG": np.random.randint(0, 5, 120),
            "FTR": np.random.choice(["H", "D", "A"], 120, p=[0.45, 0.25, 0.30])
        })

    st.success("Dataset ready")

# --------------------------------------------------
# Feature engineering + model
# --------------------------------------------------
@st.cache_data
def build_model(df):
    df = df.copy()

    df["HomeForm"] = df.groupby("HomeTeam")["FTR"].transform(lambda x: (x == "H").rolling(5, 1).mean())
    df["AwayForm"] = df.groupby("AwayTeam")["FTR"].transform(lambda x: (x == "A").rolling(5, 1).mean())
    df["HomeGoalsAvg"] = df.groupby("HomeTeam")["FTHG"].transform(lambda x: x.rolling(5, 1).mean())
    df["AwayGoalsAvg"] = df.groupby("AwayTeam")["FTAG"].transform(lambda x: x.rolling(5, 1).mean())

    X = df[["HomeForm", "AwayForm", "HomeGoalsAvg", "AwayGoalsAvg"]].fillna(0)
    y = df["FTR"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    return df, model, acc, classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred)

df, model, acc, clf_report, cm = build_model(df_raw)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_pred, tab_perf, tab_hist, tab_team = st.tabs(
    ["🔮 Predictor", "📊 Model", "📜 History", "🏆 Team Stats"]
)

# --------------------------------------------------
# Predictor tab (SCROLL FIXED)
# --------------------------------------------------
with tab_pred:
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))

    home_team = st.multiselect("🏠 Home Team", teams, max_selections=1, default=[teams[0]])[0]
    away_team = st.multiselect(
        "✈️ Away Team",
        [t for t in teams if t != home_team],
        max_selections=1,
        default=[[t for t in teams if t != home_team][0]]
    )[0]

    home_form = st.slider("Home Form", 0.0, 1.0, 0.6)
    away_form = st.slider("Away Form", 0.0, 1.0, 0.6)
    home_goals = st.number_input("Home Goals Avg", 0.0, 5.0, 1.5)
    away_goals = st.number_input("Away Goals Avg", 0.0, 5.0, 1.2)

    if st.button("Predict"):
        X_new = pd.DataFrame([{
            "HomeForm": home_form,
            "AwayForm": away_form,
            "HomeGoalsAvg": home_goals,
            "AwayGoalsAvg": away_goals
        }])

        pred = model.predict(X_new)[0]
        probs = model.predict_proba(X_new)[0]
        labels = model.named_steps["rf"].classes_

        st.subheader(f"{home_team} vs {away_team}")
        for l, p in zip(labels, probs):
            st.metric(l, f"{p:.1%}")

# --------------------------------------------------
# Performance tab
# --------------------------------------------------
with tab_perf:
    st.metric("Accuracy", f"{acc:.1%}")
    st.dataframe(pd.DataFrame(clf_report).transpose())

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# --------------------------------------------------
# History tab
# --------------------------------------------------
with tab_hist:
    st.dataframe(df.sort_values("Date", ascending=False), use_container_width=True)

# --------------------------------------------------
# Team stats tab
# --------------------------------------------------
with tab_team:
    team = st.multiselect("Select Team", teams, max_selections=1, default=[teams[0]])[0]

    home = df[df["HomeTeam"] == team]
    away = df[df["AwayTeam"] == team]

    wins = (home["FTR"] == "H").sum() + (away["FTR"] == "A").sum()
    matches = len(home) + len(away)

    st.metric("Matches", matches)
    st.metric("Wins", wins)
    st.metric("Win Rate", f"{wins / matches * 100:.1f}%" if matches else "N/A")

st.markdown("<hr><center>⚽ Streamlit • La Liga Analytics</center>", unsafe_allow_html=True)
