import pandas as pd
import numpy as np
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# Data + model utilities
# =========================
@st.cache_data
def load_data(path_or_file):
    return pd.read_csv(path_or_file)


def get_all_teams(df_raw: pd.DataFrame):
    home_teams = df_raw["HomeTeam"].dropna().unique()
    away_teams = df_raw["AwayTeam"].dropna().unique()
    return sorted(set(home_teams).union(set(away_teams)))


def get_all_seasons(df_raw: pd.DataFrame):
    return sorted(df_raw["Season"].dropna().unique())


def build_features_and_model(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # Sort by date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # Form calculation
    def calculate_team_points(row, team_name):
        if row["HomeTeam"] == team_name:
            if row["FTR"] == "H":
                return 3
            elif row["FTR"] == "D":
                return 1
            return 0
        elif row["AwayTeam"] == team_name:
            if row["FTR"] == "A":
                return 3
            elif row["FTR"] == "D":
                return 1
            return 0
        return 0

    df["HomeTeamPoints"] = df.apply(
        lambda r: calculate_team_points(r, r["HomeTeam"]), axis=1
    )
    df["AwayTeamPoints"] = df.apply(
        lambda r: calculate_team_points(r, r["AwayTeam"]), axis=1
    )

    # Rolling form (0–1)
    df["HomeForm"] = (
        df.groupby("HomeTeam")["HomeTeamPoints"]
        .rolling(5, min_periods=1)
        .mean()
        .div(3)
        .reset_index(level=0, drop=True)
    )
    df["AwayForm"] = (
        df.groupby("AwayTeam")["AwayTeamPoints"]
        .rolling(5, min_periods=1)
        .mean()
        .div(3)
        .reset_index(level=0, drop=True)
    )

    # Expanding goals average
    df["HomeGoalsAvg"] = (
        df.groupby("HomeTeam")["FTHG"]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["AwayGoalsAvg"] = (
        df.groupby("AwayTeam")["FTAG"]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df = df.dropna(
        subset=["HomeForm", "AwayForm", "HomeGoalsAvg", "AwayGoalsAvg"], how="all"
    )

    feature_cols = [
        "Season",
        "HomeTeam",
        "AwayTeam",
        "HomeForm",
        "AwayForm",
        "HomeGoalsAvg",
        "AwayGoalsAvg",
    ]
    X = df[feature_cols]
    y = df["FTR"]

    if len(X) == 0:
        raise ValueError("No valid data after feature engineering")

    categorical_cols = ["Season", "HomeTeam", "AwayTeam"]
    numeric_cols = ["HomeForm", "AwayForm", "HomeGoalsAvg", "AwayGoalsAvg"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("num", SimpleImputer(strategy="mean"), numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    if len(X) < 4:
        X_train, X_test = X, X[:1] if len(X) > 0 else X
        y_train, y_test = y, y[:1] if len(y) > 0 else y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["H", "D", "A"])

    return df, X, y, model, acc, report, cm


def compute_recent_stats(df, season, home_team, away_team, window=5):
    sub = df[df["Season"] == season].copy()

    home_matches = sub[
        (sub["HomeTeam"] == home_team) | (sub["AwayTeam"] == home_team)
    ]
    away_matches = sub[
        (sub["HomeTeam"] == away_team) | (sub["AwayTeam"] == away_team)
    ]

    home_last = home_matches.tail(window)
    away_last = away_matches.tail(window)

    def team_stats(matches, team_name):
        points = 0
        goals = 0
        n_matches = len(matches)

        if n_matches == 0:
            return 0.5, 1.0

        for _, row in matches.iterrows():
            if row["HomeTeam"] == team_name:
                if row["FTR"] == "H":
                    points += 3
                elif row["FTR"] == "D":
                    points += 1
                goals += row["FTHG"]
            else:
                if row["FTR"] == "A":
                    points += 3
                elif row["FTR"] == "D":
                    points += 1
                goals += row["FTAG"]

        form = points / (n_matches * 3)
        goals_avg = goals / n_matches
        return form, goals_avg

    home_form, home_goals_avg = team_stats(home_last, home_team)
    away_form, away_goals_avg = team_stats(away_last, away_team)

    return home_form, away_form, home_goals_avg, away_goals_avg


# =========================
# Streamlit app
# =========================
st.set_page_config(
    page_title="Football Match Predictor",
    layout="wide",  # helps reduce vertical scrolling
)

st.title("Football Match Predictor")

# Use sidebar for controls to reduce vertical scroll in main area
with st.sidebar:
    st.header("Upload & Settings")

    uploaded_file = st.file_uploader("Upload match CSV", type=["csv"])

    window = st.number_input(
        "Recent form window (matches)",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

# Load raw data
df_raw = load_data(uploaded_file)

# Prepare dropdown options from RAW data (fixes missing teams)
all_teams = get_all_teams(df_raw)
all_seasons = get_all_seasons(df_raw)

# Train model and engineered df
with st.spinner("Training model..."):
    df_feat, X, y, model, acc, report, cm = build_features_and_model(df_raw)

# Compact input layout using columns
st.subheader("Prediction Inputs")
col1, col2, col3 = st.columns(3)

with col1:
    season = st.selectbox("Season", all_seasons)

with col2:
    home_team = st.selectbox("Home Team", all_teams)

with col3:
    away_team = st.selectbox(
        "Away Team",
        [t for t in all_teams if t != home_team] if len(all_teams) > 1 else all_teams,
    )

# Compute recent stats using engineered df
home_form, away_form, home_goals_avg, away_goals_avg = compute_recent_stats(
    df_feat, season, home_team, away_team, window=window
)

# Build single-row feature input for prediction
input_df = pd.DataFrame(
    {
        "Season": [season],
        "HomeTeam": [home_team],
        "AwayTeam": [away_team],
        "HomeForm": [home_form],
        "AwayForm": [away_form],
        "HomeGoalsAvg": [home_goals_avg],
        "AwayGoalsAvg": [away_goals_avg],
    }
)

if st.button("Predict Result", type="primary"):
    proba = model.predict_proba(input_df)[0]
    classes = model.named_steps["classifier"].classes_

    # Put probabilities in a small table
    prob_df = pd.DataFrame({"Result": classes, "Probability": proba}).sort_values(
        "Probability", ascending=False
    )

    st.markdown("### Predicted Result")
    st.metric(
        label="Most likely outcome",
        value=prob_df.iloc[0]["Result"],
        delta=f"{prob_df.iloc[0]['Probability']*100:.1f}%",
    )

    st.markdown("### Outcome probabilities")
    st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)

# Put detailed metrics in an expander to avoid scrolling
with st.expander("Model performance details"):
    st.write(f"Accuracy: **{acc:.3f}**")
    st.write("Confusion matrix (H, D, A):")
    st.write(cm)
    st.write("Classification report:")
    st.json(report)
