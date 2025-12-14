# backend.py
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(path_or_file):
    """Load CSV from path or uploaded file object."""
    return pd.read_csv(path_or_file)


def build_features_and_model(df_raw):
    """Feature engineering + model training. Returns df, X, y, model, metrics."""
    df = df_raw.copy()

    # Ensure Date is sorted chronologically
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date")

    # Binary outcomes for rolling form
    df["HomeWin"] = (df["FTR"] == "H").astype(int)
    df["AwayWin"] = (df["FTR"] == "A").astype(int)

    # Rolling form (last 5 matches) per team
    df["HomeForm"] = (
        df.groupby("HomeTeam")["HomeWin"]
        .rolling(5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["AwayForm"] = (
        df.groupby("AwayTeam")["AwayWin"]
        .rolling(5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Expanding goals average per team
    df["HomeGoalsAvg"] = (
        df.groupby("HomeTeam")["FTHG"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["AwayGoalsAvg"] = (
        df.groupby("AwayTeam")["FTAG"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Drop rows with NaN created by rolling/expanding
    df = df.dropna(subset=["HomeForm", "AwayForm", "HomeGoalsAvg", "AwayGoalsAvg"])

    # Features and target
    X = df[
        [
            "Season",
            "HomeTeam",
            "AwayTeam",
            "HomeForm",
            "AwayForm",
            "HomeGoalsAvg",
            "AwayGoalsAvg",
        ]
    ]
    y = df["FTR"]

    categorical_cols = ["Season", "HomeTeam", "AwayTeam"]
    numeric_cols = ["HomeForm", "AwayForm", "HomeGoalsAvg", "AwayGoalsAvg"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", SimpleImputer(strategy="mean"), numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=["H", "D", "A"])

    return df, X, y, model, acc, report, cm


def compute_recent_stats(df, season, home_team, away_team, window=5):
    """Compute recent form and goal averages for two teams in a season."""
    sub = df[df["Season"] == season].copy()
    home_sub = sub[(sub["HomeTeam"] == home_team) | (sub["AwayTeam"] == home_team)]
    away_sub = sub[(sub["HomeTeam"] == away_team) | (sub["AwayTeam"] == away_team)]

    # Home team recent form and goals
    home_last = home_sub.tail(window)
    home_points = 0
    home_goals = 0
    if not home_last.empty:
        for _, r in home_last.iterrows():
            if r["HomeTeam"] == home_team:
                if r["FTR"] == "H":
                    home_points += 3
                elif r["FTR"] == "D":
                    home_points += 1
                home_goals += r["FTHG"]
            else:
                if r["FTR"] == "A":
                    home_points += 3
                elif r["FTR"] == "D":
                    home_points += 1
                home_goals += r["FTAG"]
        home_form = home_points / (window * 3)
        home_goals_avg = home_goals / window
    else:
        home_form = 0.5
        home_goals_avg = 1.0

    # Away team recent form and goals
    away_last = away_sub.tail(window)
    away_points = 0
    away_goals = 0
    if not away_last.empty:
        for _, r in away_last.iterrows():
            if r["HomeTeam"] == away_team:
                if r["FTR"] == "H":
                    away_points += 3
                elif r["FTR"] == "D":
                    away_points += 1
                away_goals += r["FTHG"]
            else:
                if r["FTR"] == "A":
                    away_points += 3
                elif r["FTR"] == "D":
                    away_points += 1
                away_goals += r["FTAG"]
        away_form = away_points / (window * 3)
        away_goals_avg = away_goals / window
    else:
        away_form = 0.5
        away_goals_avg = 1.0

    return home_form, away_form, home_goals_avg, away_goals_avg
