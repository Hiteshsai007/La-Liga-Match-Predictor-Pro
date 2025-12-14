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
    df = df.sort_values("Date").reset_index(drop=True)

    # Corrected form calculation: points from team's perspective (win=3, draw=1, loss=0)
    def calculate_team_points(row, team_name):
        """Calculate points for a team regardless of home/away role."""
        if row["HomeTeam"] == team_name:
            if row["FTR"] == "H": return 3
            elif row["FTR"] == "D": return 1
            return 0
        elif row["AwayTeam"] == team_name:
            if row["FTR"] == "A": return 3
            elif row["FTR"] == "D": return 1
            return 0
        return 0

    # Calculate points for each team in their matches
    df["HomeTeamPoints"] = df.apply(lambda row: calculate_team_points(row, row["HomeTeam"]), axis=1)
    df["AwayTeamPoints"] = df.apply(lambda row: calculate_team_points(row, row["AwayTeam"]), axis=1)

    # Rolling form (last 5 matches) - normalized to [0,1]
    df["HomeForm"] = (
        df.groupby("HomeTeam")["HomeTeamPoints"]
        .rolling(5, min_periods=1)  # Allow fewer matches
        .mean()
        .div(3)  # Normalize: max points per match = 3
        .reset_index(level=0, drop=True)
    )

    df["AwayForm"] = (
        df.groupby("AwayTeam")["AwayTeamPoints"]
        .rolling(5, min_periods=1)
        .mean()
        .div(3)
        .reset_index(level=0, drop=True)
    )

    # Expanding goals average per team
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

    # Only drop rows where ALL features are NaN (should be none now)
    df = df.dropna(subset=["HomeForm", "AwayForm", "HomeGoalsAvg", "AwayGoalsAvg"], how='all')

    # Features and target
    feature_cols = [
        "Season", "HomeTeam", "AwayTeam", 
        "HomeForm", "AwayForm", "HomeGoalsAvg", "AwayGoalsAvg"
    ]
    X = df[feature_cols]
    y = df["FTR"]

    if len(X) == 0:
        raise ValueError("No valid data after feature engineering")

    categorical_cols = ["Season", "HomeTeam", "AwayTeam"]
    numeric_cols = ["HomeForm", "AwayForm", "HomeGoalsAvg", "AwayGoalsAvg"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", SimpleImputer(strategy="mean"), numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    # Ensure enough samples for train/test split
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
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=["H", "D", "A"])

    return df, X, y, model, acc, report, cm

def compute_recent_stats(df, season, home_team, away_team, window=5):
    """Compute recent form and goal averages for two teams in a season."""
    sub = df[df["Season"] == season].copy()
    
    # Get recent matches for each team (where they played home OR away)
    home_matches = sub[(sub["HomeTeam"] == home_team) | (sub["AwayTeam"] == home_team)]
    away_matches = sub[(sub["HomeTeam"] == away_team) | (sub["AwayTeam"] == away_team)]
    
    home_last = home_matches.tail(window)
    away_last = away_matches.tail(window)

    def team_stats(matches, team_name):
        """Calculate form and goals for a team from their recent matches."""
        points = 0
        goals = 0
        n_matches = len(matches)
        
        if n_matches == 0:
            return 0.5, 1.0  # Default average values
        
        for _, row in matches.iterrows():
            if row["HomeTeam"] == team_name:
                # Home match
                if row["FTR"] == "H": points += 3
                elif row["FTR"] == "D": points += 1
                goals += row["FTHG"]
            else:
                # Away match
                if row["FTR"] == "A": points += 3
                elif row["FTR"] == "D": points += 1
                goals += row["FTAG"]
        
        form = points / (n_matches * 3)  # Normalize to [0,1]
        goals_avg = goals / n_matches
        return form, goals_avg

    home_form, home_goals_avg = team_stats(home_last, home_team)
    away_form, away_goals_avg = team_stats(away_last, away_team)

    return home_form, away_form, home_goals_avg, away_goals_avg
