import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from backend import (
    load_data,
    build_features_and_model,
    compute_recent_stats,
)

# -----------------------------
# Enhanced Page Config & Styling
# -----------------------------
st.set_page_config(
    page_title="⚽ La Liga Match Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Responsive CSS
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
.stButton > button:hover {transform: translateY(-2px); box-shadow: 0 10px 25px var(--card-glow);}
.stSelectbox > div > div > div, .stSlider > div > div {background-color: var(--card-bg); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);}
[data-testid="stMetricLabel"] {font-size: clamp(0.8rem, 2vw, 1rem);}
[data-testid="stMetricValue"] {font-size: clamp(1.5rem, 4vw, 2.5rem);}

.stPlotlyChart canvas {max-height: 600px !important;}
</style>
""", unsafe_allow_html=True)

# Global matplotlib style for professional charts
plt.style.use('default')
sns.set_palette("husl")

# -----------------------------
# Gradient Header with Animation
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
# Enhanced Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 🎯 **Dashboard Controls**")
    
    uploaded_file = st.file_uploader("📁 Upload LaLiga CSV", type=["csv"])
    if uploaded_file is not None:
        df_raw = load_data(uploaded_file)
    else:
        # ✅ STREAMLIT CLOUD COMPATIBLE - Sample data
        st.info("🧪 Using sample La Liga data (upload real CSV anytime)")
        np.random.seed(42)
        data = {
            'Season': ['2023-24']*50 + ['2024-25']*50,
            'Date': pd.date_range('2023-08-01', periods=100, freq='3D').strftime('%d/%m/%Y'),
            'HomeTeam': np.random.choice(['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia'], 100),
            'AwayTeam': np.random.choice(['Villarreal', 'Betis', 'Athletic Club', 'Getafe', 'Osasuna'], 100),
            'FTHG': np.random.choice([0,1,2,3,4], 100, p=[0.2,0.3,0.3,0.15,0.05]),
            'FTAG': np.random.choice([0,1,2,3,4], 100, p=[0.25,0.35,0.25,0.1,0.05]),
            'FTR': np.random.choice(['H', 'D', 'A'], 100, p=[0.45, 0.25, 0.30])
        }
        df_raw = pd.DataFrame(data)
    
    st.success("✅ Dataset ready!")
    
    # Model loading...
    @st.cache_data
    def _cached_build(df_raw):
        return build_features_and_model(df_raw)
    
    with st.spinner("🔄 Training AI Model..."):
        df, X, y, model, acc, clf_report, cm = _cached_build(df_raw)

# -----------------------------
# Main Tabs with Icons
# -----------------------------
tab_pred, tab_perf, tab_history, tab_team_stats = st.tabs([
    "🔮 **Match Predictor**", 
    "📊 **Model Analytics**", 
    "📜 **Match History**", 
    "🏆 **Team Stats**"
])

def create_responsive_bar_chart(data_dict, title, subtitle="", colors=['#00ff7f', '#ffd700', '#ff4b4b']):
    """Create professional responsive bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white', dpi=100)
    bars = ax.bar(data_dict.keys(), data_dict.values(), 
                  color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    
    # Responsive styling
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#1e293b')
    ax.set_ylabel("Probability / Count", fontsize=14, fontweight='bold', color='#1e293b')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(data_dict.values())*0.01,
                f'{height:.1%}' if height <= 1 else f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Professional grid and styling
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#e2e8f0')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig

def create_line_chart_responsive(data_df, title, team_name):
    """Create professional responsive line chart"""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white', dpi=100)
    
    # Plot with fill
    ax.plot(data_df["Season"], data_df["WinRate"], marker="o", linewidth=4, 
            markersize=12, color="#00ff7f", markerfacecolor="#00ff7f", markeredgecolor="white", markeredgewidth=2)
    ax.fill_between(data_df["Season"], data_df["WinRate"], alpha=0.25, color="#00ff7f")
    
    ax.set_title(f"{team_name} - Win Rate Evolution", fontsize=20, fontweight='bold', pad=25, color='#1e293b')
    ax.set_ylabel("Win Rate", fontsize=16, fontweight='bold', color='#1e293b')
    ax.set_ylim(0, 1)
    
    # Professional styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#e2e8f0')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig

# 1. Enhanced Prediction Tab
with tab_pred:
    st.markdown("---")
    
    # Matchup selector with cards
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("### 🏠 **Home Team**")
        seasons = sorted(df["Season"].unique())
        season_input = st.selectbox("📅 Season", seasons, index=len(seasons)-1)
        home_team = st.selectbox("⚽ Home Team", sorted(df["HomeTeam"].unique()))
    
    with col2:
        st.markdown("### ✈️ **Away Team**")
        away_team = st.selectbox("⚽ Away Team", sorted(df["AwayTeam"].unique()))
        if away_team == home_team:
            st.error("⚠️ Home and Away teams must be different!")
    
    # Form inputs with gradient cards
    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.markdown("### 📈 **Team Form**")
        home_form = st.slider("🏠 Home Form", 0.0, 1.0, 0.7, 0.01, 
                            help="Recent performance (0=poor, 1=excellent)")
        home_goals_avg = st.number_input("⚽ Home Goals/Game", 0.0, 5.0, 1.8, 0.1)
    
    with col4:
        st.markdown("### 📉 **Opponent Form**")
        away_form = st.slider("✈️ Away Form", 0.0, 1.0, 0.7, 0.01)
        away_goals_avg = st.number_input("⚽ Away Goals/Game", 0.0, 5.0, 1.2, 0.1)
    
    # Auto-fill with history
    if st.button("🧠 Auto-fill from History", use_container_width=True):
        try:
            hf, af, hga, aga = compute_recent_stats(df, season_input, home_team, away_team)
            st.success(f"✅ Loaded recent stats!\n🏠 {home_team}: Form={hf:.2f}, Goals={hga:.2f}\n✈️ {away_team}: Form={af:.2f}, Goals={aga:.2f}")
            st.rerun()
        except:
            st.warning("⚠️ No recent history available")
    
    # Prediction button
    if st.button("🚀 **Predict Match Outcome**", use_container_width=True):
        upcoming_df = pd.DataFrame([{
            "Season": season_input, "HomeTeam": home_team, "AwayTeam": away_team,
            "HomeForm": home_form, "AwayForm": away_form,
            "HomeGoalsAvg": home_goals_avg, "AwayGoalsAvg": away_goals_avg
        }])
        
        prediction = model.predict(upcoming_df)[0]
        proba = model.predict_proba(upcoming_df)[0]
        
        result_map = {"H": "🏠 Home Win", "D": "🤝 Draw", "A": "✈️ Away Win"}
        label_order = model.named_steps["classifier"].classes_
        
        # Prediction card
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #00ff7f22, #ffd70022); 
                    border-radius: 25px; border: 3px solid var(--accent-green); box-shadow: 0 15px 35px rgba(0,255,127,0.2);'>
            <h2 style='color: var(--text-primary); margin: 0;'>📊 AI PREDICTION</h2>
            <h1 style='color: var(--accent-green); font-size: clamp(2rem, 6vw, 4rem); margin: 1.5rem 0;'>
                {result_map[prediction]}
            </h1>
            <p style='color: var(--text-secondary); font-size: 1.4rem;'>
                **{home_team} vs {away_team}**
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability metrics
        prob_dict = {result_map[lbl]: p for lbl, p in zip(label_order, proba)}
        col_hw, col_dr, col_aw = st.columns(3)
        col_hw.metric("🏠 Home Win", f"{prob_dict.get('🏠 Home Win', 0):.1%}")
        col_dr.metric("🤝 Draw", f"{prob_dict.get('🤝 Draw', 0):.1%}")
        col_aw.metric("✈️ Away Win", f"{prob_dict.get('✈️ Away Win', 0):.1%}")
        
        # Professional responsive chart
        fig = create_responsive_bar_chart(
            prob_dict, 
            f"{home_team} vs {away_team} - Match Outcome Probabilities",
            "Powered by Logistic Regression AI Model"
        )
        st.pyplot(fig)

# 2. Performance Tab
with tab_perf:
    st.markdown("### 🎯 Model Performance Overview")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("✅ Validation Accuracy", f"{acc:.1%}")
    
    with col2:
        st.markdown("**📈 Precision & Recall by Outcome**")
        report_df = pd.DataFrame(clf_report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True, hide_index=False)
    
    # Professional Confusion Matrix
    st.markdown("### 🔍 Prediction Accuracy Heatmap")
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6), facecolor='white', dpi=100)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", 
                xticklabels=["🏠 Home", "🤝 Draw", "✈️ Away"],
                yticklabels=["🏠 Home", "🤝 Draw", "✈️ Away"],
                ax=ax_cm, cbar_kws={'label': 'Number of Matches', 'shrink': 0.8})
    ax_cm.set_xlabel("Predicted Outcome", fontweight='bold', fontsize=14)
    ax_cm.set_ylabel("Actual Outcome", fontweight='bold', fontsize=14)
    ax_cm.set_title("Confusion Matrix - Model Performance", fontweight='bold', fontsize=18, pad=20)
    plt.tight_layout()
    st.pyplot(fig_cm)

# 3. History Tab
with tab_history:
    st.markdown("### 📜 Filter & Explore Match History")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        season_filter = st.multiselect("📅 Seasons", sorted(df["Season"].unique()))
    with col2:
        home_filter = st.multiselect("🏠 Home Teams", sorted(df["HomeTeam"].unique()))
    with col3:
        away_filter = st.multiselect("✈️ Away Teams", sorted(df["AwayTeam"].unique()))
    
    df_view = df.copy()
    if season_filter: df_view = df_view[df_view["Season"].isin(season_filter)]
    if home_filter: df_view = df_view[df_view["HomeTeam"].isin(home_filter)]
    if away_filter: df_view = df_view[df_view["AwayTeam"].isin(away_filter)]
    
    # Safe column selection
    available_cols = ["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    existing_cols = [col for col in available_cols if col in df_view.columns]
    feature_cols = ["HomeForm", "AwayForm", "HomeGoalsAvg", "AwayGoalsAvg"]
    for col in feature_cols:
        if col in df_view.columns:
            existing_cols.append(col)
    
    st.dataframe(
        df_view[existing_cols].sort_values("Date", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "FTR": st.column_config.SelectboxColumn(
                "Result", 
                options={"H": "🏠 Home Win", "D": "🤝 Draw", "A": "✈️ Away Win"},
                required=True
            )
        }
    )

# 4. Team Stats Tab
with tab_team_stats:
    st.markdown("### 🏆 Team Performance Analytics")
    
    teams_all = sorted(set(df["HomeTeam"].unique()).union(df["AwayTeam"].unique()))
    team_selected = st.selectbox("⚽ Select Team", teams_all)
    
    df_team_home = df[df["HomeTeam"] == team_selected]
    df_team_away = df[df["AwayTeam"] == team_selected]
    total_matches = len(df_team_home) + len(df_team_away)
    
    wins = (df_team_home["FTR"] == "H").sum() + (df_team_away["FTR"] == "A").sum()
    draws = (df_team_home["FTR"] == "D").sum() + (df_team_away["FTR"] == "D").sum()
    losses = total_matches - wins - draws
    
    goals_for = df_team_home["FTHG"].sum() + df_team_away["FTAG"].sum()
    goals_against = df_team_home["FTAG"].sum() + df_team_away["FTHG"].sum()
    
    # Enhanced metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("⚽ Matches", total_matches)
    col_b.metric("🥇 Win Rate", f"{(wins/total_matches)*100:.1f}%", 
                delta=f"+{wins-total_matches//3}" if total_matches else None)
    col_c.metric("⚽ Goals For", f"{goals_for/total_matches:.2f}" if total_matches else "N/A")
    col_d.metric("🛡️ Goals Against", f"{goals_against/total_matches:.2f}" if total_matches else "N/A")
    
    # Professional results chart
    results_data = {"🥇 Wins": wins, "🤝 Draws": draws, "😞 Losses": losses}
    fig_res = create_responsive_bar_chart(
        results_data, 
        f"{team_selected} - Overall Performance",
        f"Total Matches: {total_matches}"
    )
    st.pyplot(fig_res)
    
    # Season win rate trend
    season_stats = []
    for season in sorted(df["Season"].unique()):
        df_s_home = df_team_home[df_team_home["Season"] == season]
        df_s_away = df_team_away[df_team_away["Season"] == season]
        matches_s = len(df_s_home) + len(df_s_away)
        if matches_s > 0:
            wins_s = (df_s_home["FTR"] == "H").sum() + (df_s_away["FTR"] == "A").sum()
            season_stats.append({"Season": season, "WinRate": wins_s / matches_s})
    
    if season_stats:
        season_df = pd.DataFrame(season_stats)
        fig_wr = create_line_chart_responsive(season_df, f"{team_selected} Win Rate Trend", team_selected)
        st.pyplot(fig_wr)
    else:
        st.info(f"📊 No season data available for {team_selected}")

# Professional Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: var(--text-secondary); padding: 2rem; 
           border-top: 1px solid rgba(255,255,255,0.1);'>
    <p style='margin: 0 0 0.5rem 0; font-size: 1.1rem;'>⚽ Built with ❤️ for La Liga fans</p>
    <p style='margin: 0; font-size: 0.9rem;'>Powered by Streamlit • Scikit-learn • Matplotlib</p>
</div>
""", unsafe_allow_html=True)
