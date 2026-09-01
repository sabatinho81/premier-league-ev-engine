import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="PL Hybrid Engine & EV", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Roboto:wght@400;500;700&display=swap');

    .stApp {
        background-color: #3D195B;
        color: #FFFFFF;
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3 { font-family: 'Oswald', sans-serif !important; text-transform: uppercase; }
    .header-title {
        color: #00FF85; font-size: 3rem; font-weight: 700; margin-bottom: 0px; text-shadow: 2px 2px 0px #963CFF;
    }
    .sub-header { color: #FFFFFF; font-size: 1.2rem; margin-bottom: 20px; }
    .metric-card {
        background: #2D1045; padding: 24px; border-radius: 8px; border-left: 5px solid #00FF85; text-align: center; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4); transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover { transform: translateY(-4px); border-left: 5px solid #E90052; }
    .info-card {
        background: #240024; padding: 20px; border-radius: 8px; border: 1px solid #963CFF; margin-bottom: 20px;
    }
    .value-bet { color: #00FF85 !important; font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 1.8rem; }
    .bad-bet { color: #E90052 !important; font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 1.8rem; }
    [data-testid="stSidebar"] { background-color: #240024 !important; }
    hr { border-color: #963CFF; opacity: 0.5; }
    
    /* Style the Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; color: #FFFFFF; font-family: 'Oswald', sans-serif; font-size: 1.2rem; }
    .stTabs [aria-selected="true"] { color: #00FF85 !important; border-bottom: 4px solid #00FF85 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SETTINGS & SESSION STATE MANAGEMENT
# ==========================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'currency' not in st.session_state:
    st.session_state.currency = '£'
if 'bookies_toggle' not in st.session_state:
    st.session_state.bookies_toggle = True
if 'ev_threshold' not in st.session_state:
    st.session_state.ev_threshold = 1.5
if 'home_team_sel' not in st.session_state:
    st.session_state.home_team_sel = ""
if 'away_team_sel' not in st.session_state:
    st.session_state.away_team_sel = ""
if 'odds_h_val' not in st.session_state:
    st.session_state.odds_h_val = None
if 'odds_d_val' not in st.session_state:
    st.session_state.odds_d_val = None
if 'odds_a_val' not in st.session_state:
    st.session_state.odds_a_val = None

DICT = {
    "en": {
        "title": "🦁 HYBRID MATCHDAY ENGINE & EV",
        "subtitle": "Venue tactics, short-term momentum, and real-time Expected Value.",
        "settings": "⚙️ SETTINGS",
        "match_setup": "📋 MATCH SETUP",
        "home": "🏠 HOME CLUB",
        "away": "✈️ AWAY CLUB",
        "odds": "💷 LIVE ODDS",
        "win": "Win",
        "draw": "Draw",
        "h2h": "TACTICAL H2H",
        "momentum": "VENUE MOMENTUM",
        "gd": "NET GOAL DIFF",
        "sh_threat": "2ND HALF THREAT",
        "conversion": "CONVERSION RATE",
        "corners": "CORNERS / MATCH",
        "overall_pts": "OVERALL PTS (LAST 5)",
        "overall_gd": "OVERALL GD (LAST 5)",
        "forecast": "AI FORECAST & EV",
        "true_prob": "TRUE PROB",
        "odds_lbl": "ODDS",
        "tab_match": "🏟️ MATCH CENTER",
        "tab_model": "🧠 INSIDE THE ENGINE",
        "model_title": "How The Hybrid XGBoost Engine Works",
        "ev_title": "Understanding Expected Value (EV)",
        "features_title": "Live Feature Importance Weights",
        "select_teams": "ℹ️ Please select a Home and Away team from the sidebar to begin.",
        "same_teams": "⚠️ Please select two distinct clubs.",
        "missing_odds": "⚠️ Please input all three bookmaker odds in the sidebar.",
        "toggle_odds": "Include Bookmaker Odds",
        "toggle_help": "Turn OFF to calculate probabilities strictly using statistics.",
        "risk_management": "🛡️ RISK MANAGEMENT",
        "min_ev_lbl": "Minimum EV Margin",
        "min_ev_help": "Filters out marginal bets. Only highlights matches where expected profit per 10 wagered exceeds this safety threshold.",
        "stat_glossary_title": "📖 What do these metrics mean?",
        "def_momentum": "The difference between short-term points (last 3 venue games) and long-term average (last 10 venue games). Positive means surging form.",
        "def_gd": "Average goals scored minus goals conceded over the last 10 games at this venue.",
        "def_sh_threat": "Average net goals (scored - conceded) in the second half over the last 10 venue games. Captures fitness and tactical adjustments.",
        "def_conversion": "Percentage of shots on target that result in a goal. High values indicate clinical finishing.",
        "def_overall_pts": "Total points accumulated over the team's last 5 matches across all venues. Captures immediate overall form."
    },
    "pl": {
        "title": "🦁 HYBRYDOWY SILNIK MECZOWY I EV",
        "subtitle": "Taktyka dom/wyjazd, krótkoterminowa forma i oczekiwana wartość (EV).",
        "settings": "⚙️ USTAWIENIA",
        "match_setup": "📋 USTAWIENIA MECZU",
        "home": "🏠 GOSPODARZ",
        "away": "✈️ GOŚĆ",
        "odds": "💷 KURSY BUKMACHERSKIE",
        "win": "Wygrana",
        "draw": "Remis",
        "h2h": "ANALIZA TAKTYCZNA H2H",
        "momentum": "MOMENTUM (DOM/WYJAZD)",
        "gd": "RÓŻNICA BRAMEK (NETTO)",
        "sh_threat": "ZAGROŻENIE W 2. POŁOWIE",
        "conversion": "SKUTECZNOŚĆ STRZAŁÓW",
        "corners": "RZUTY ROŻNE / MECZ",
        "overall_pts": "PUNKTY OGÓŁEM (OST. 5)",
        "overall_gd": "BILANS OGÓŁEM (OST. 5)",
        "forecast": "PROGNOZA AI I EV",
        "true_prob": "PRAWD. AI",
        "odds_lbl": "KURS",
        "tab_match": "🏟️ CENTRUM MECZOWE",
        "tab_model": "🧠 JAK DZIAŁA SILNIK",
        "model_title": "Jak działa hybrydowy model XGBoost",
        "ev_title": "Zrozumieć Oczekiwaną Wartość (EV)",
        "features_title": "Wagi Ważności Cech na Żywo",
        "select_teams": "ℹ️ Wybierz drużynę gospodarzy i gości w panelu bocznym, aby rozpocząć.",
        "same_teams": "⚠️ Wybierz dwa różne kluby.",
        "missing_odds": "⚠️ Wprowadź wszystkie trzy kursy bukmacherskie w panelu bocznym.",
        "toggle_odds": "Uwzględnij Kursy Bukmacherskie",
        "toggle_help": "Wyłącz, aby obliczać prawdopodobieństwo wyłącznie na podstawie statystyk.",
        "risk_management": "🛡️ ZARZĄDZANIE RYZYKIEM",
        "min_ev_lbl": "Margines Bezpieczeństwa EV",
        "min_ev_help": "Filtruje ryzykowne zakłady. Podświetla tylko te mecze, w których oczekiwany zysk na postawione 10 przekracza ten próg.",
        "stat_glossary_title": "📖 Co oznaczają te statystyki?",
        "def_momentum": "Różnica między punktami z ostatnich 3 meczów na danym stadionie, a średnią z ostatnich 10. Wartość dodatnia oznacza wzrost formy.",
        "def_gd": "Średnia różnica między zdobytymi a straconymi bramkami w ostatnich 10 meczach (dom lub wyjazd).",
        "def_sh_threat": "Średni bilans bramek w drugich połowach z ostatnich 10 meczów. Wskazuje na kondycję i adaptację taktyczną.",
        "def_conversion": "Procent strzałów celnych zakończonych golem. Wysoka wartość oznacza wysoką skuteczność.",
        "def_overall_pts": "Łączna liczba punktów zdobytych w ostatnich 5 meczach (dom i wyjazd). Odzwierciedla bieżącą formę zespołu."
    }
}

def t(key):
    return DICT[st.session_state.lang][key]

st.markdown(f"<h1 class='header-title'>{t('title')}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='sub-header'>{t('subtitle')}</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 3. DATA PIPELINE & DUAL MODEL TRAINING
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    seasons = ["2324", "2425", "2526", "2627"]
    dfs = []
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'B365H', 'B365D', 'B365A']

    for season in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
        try:
            df_season = pd.read_csv(url, usecols=cols)
            dfs.append(df_season)
        except Exception:
            try:
                df_season = pd.read_csv(f"E0_{season}.csv", usecols=cols)
                dfs.append(df_season)
            except Exception:
                continue

    if not dfs:
        return None, None, None, None, None, None, None, None

    df = pd.concat(dfs, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by="Date").dropna().reset_index(drop=True)

    conditions = [(df["FTHG"] > df["FTAG"]), (df["FTHG"] == df["FTAG"]), (df["FTHG"] < df["FTAG"])]
    df["Result"] = np.select(conditions, [2, 1, 0], default=1)
    df["Home_Points"] = np.select([(df["Result"] == 2), (df["Result"] == 1)], [3, 1], default=0)
    df["Away_Points"] = np.select([(df["Result"] == 0), (df["Result"] == 1)], [3, 1], default=0)

    df["Home_SHG"] = df["FTHG"] - df["HTHG"] 
    df["Away_SHG"] = df["FTAG"] - df["HTAG"]
    df["Implied_H"] = 1 / df["B365H"]
    df["Implied_D"] = 1 / df["B365D"]
    df["Implied_A"] = 1 / df["B365A"]

    def roll_avg(data, col, metric, window):
        return data.groupby(col)[metric].apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean()).reset_index(level=0, drop=True)

    df["H_Pts_3"], df["H_Pts_10"] = roll_avg(df, "HomeTeam", "Home_Points", 3), roll_avg(df, "HomeTeam", "Home_Points", 10)
    df["H_Momentum"] = df["H_Pts_3"] - df["H_Pts_10"]
    df["A_Pts_3"], df["A_Pts_10"] = roll_avg(df, "AwayTeam", "Away_Points", 3), roll_avg(df, "AwayTeam", "Away_Points", 10)
    df["A_Momentum"] = df["A_Pts_3"] - df["A_Pts_10"]

    w = 10 
    df["H_GS"], df["H_GC"] = roll_avg(df, "HomeTeam", "FTHG", w), roll_avg(df, "HomeTeam", "FTAG", w)
    df["H_SOT"], df["H_Corn"] = roll_avg(df, "HomeTeam", "HST", w), roll_avg(df, "HomeTeam", "HC", w)
    df["A_GS"], df["A_GC"] = roll_avg(df, "AwayTeam", "FTAG", w), roll_avg(df, "AwayTeam", "FTHG", w)
    df["A_SOT"], df["A_Corn"] = roll_avg(df, "AwayTeam", "AST", w), roll_avg(df, "AwayTeam", "AC", w)

    df["H_SH_Threat"], df["A_SH_Threat"] = roll_avg(df, "HomeTeam", "Home_SHG", w), roll_avg(df, "AwayTeam", "Away_SHG", w)
    df["H_GD"], df["A_GD"] = df["H_GS"] - df["H_GC"], df["A_GS"] - df["A_GC"]
    df["H_Conversion"], df["A_Conversion"] = df["H_GS"] / (df["H_SOT"] + 0.01), df["A_GS"] / (df["A_SOT"] + 0.01)

    home_df = df[['Date', 'HomeTeam', 'Home_Points', 'FTHG', 'FTAG']].rename(columns={'HomeTeam': 'Team', 'Home_Points': 'Pts', 'FTHG': 'GS', 'FTAG': 'GC'})
    away_df = df[['Date', 'AwayTeam', 'Away_Points', 'FTAG', 'FTHG']].rename(columns={'AwayTeam': 'Team', 'Away_Points': 'Pts', 'FTAG': 'GS', 'FTHG': 'GC'})
    
    overall_df = pd.concat([home_df, away_df]).sort_values('Date')
    overall_df['GD'] = overall_df['GS'] - overall_df['GC']
    
    overall_df['Overall_Pts_5'] = overall_df.groupby('Team')['Pts'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    overall_df['Overall_GD_5'] = overall_df.groupby('Team')['GD'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    df = df.merge(overall_df[['Date', 'Team', 'Overall_Pts_5', 'Overall_GD_5']], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Overall_Pts_5': 'H_Overall_Pts_5', 'Overall_GD_5': 'H_Overall_GD_5'}).drop(columns=['Team'])
    df = df.merge(overall_df[['Date', 'Team', 'Overall_Pts_5', 'Overall_GD_5']], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Overall_Pts_5': 'A_Overall_Pts_5', 'Overall_GD_5': 'A_Overall_GD_5'}).drop(columns=['Team'])

    clean_df = df.dropna().copy()
    
    pure_features = [
        "H_Momentum", "H_GD", "H_SH_Threat", "H_Conversion", "H_Corn", "H_Overall_Pts_5", "H_Overall_GD_5",
        "A_Momentum", "A_GD", "A_SH_Threat", "A_Conversion", "A_Corn", "A_Overall_Pts_5", "A_Overall_GD_5"
    ]
    hybrid_features = pure_features + ["Implied_H", "Implied_D", "Implied_A"]
    
    y = clean_df["Result"]
    weights = compute_sample_weight(class_weight='balanced', y=y)

    # UPDATED WITH OPTIMAL BACKTEST PARAMETERS (depth=5, estimators=100, subsample=0.9)
    model_pure = xgb.XGBClassifier(objective="multi:softprob", seed=42, max_depth=5, learning_rate=0.05, n_estimators=100, subsample=0.9, colsample_bytree=0.8)
    model_pure.fit(clean_df[pure_features], y, sample_weight=weights)

    model_hybrid = xgb.XGBClassifier(objective="multi:softprob", seed=42, max_depth=5, learning_rate=0.05, n_estimators=100, subsample=0.9, colsample_bytree=0.8)
    model_hybrid.fit(clean_df[hybrid_features], y, sample_weight=weights)

    active_teams = sorted(list(set(clean_df["HomeTeam"].unique()) | set(clean_df["AwayTeam"].unique())))
    latest = {}
    for team in active_teams:
        matches = clean_df[(clean_df["HomeTeam"] == team) | (clean_df["AwayTeam"] == team)].sort_values("Date")
        if not matches.empty:
            last = matches.iloc[-1]
            pfx = "H_" if last["HomeTeam"] == team else "A_"
            latest[team] = {
                "Momentum": last[f"{pfx}Momentum"], "GD": last[f"{pfx}GD"], 
                "SH_Threat": last[f"{pfx}SH_Threat"], "Conversion": last[f"{pfx}Conversion"], "Corn": last[f"{pfx}Corn"],
                "Overall_Pts_5": last[f"{pfx}Overall_Pts_5"], "Overall_GD_5": last[f"{pfx}Overall_GD_5"]
            }

    imps_pure = pd.Series(model_pure.feature_importances_, index=pure_features).sort_values(ascending=False)
    imps_hybrid = pd.Series(model_hybrid.feature_importances_, index=hybrid_features).sort_values(ascending=False)
    
    return model_pure, model_hybrid, latest, active_teams, pure_features, hybrid_features, imps_pure, imps_hybrid

# ==========================================
# 4. VISUAL COMPONENT BUILDER
# ==========================================
def render_stat_bar(label, home_val, away_val, is_percentage=False):
    min_val = min(0, home_val, away_val)
    offset = abs(min_val) if min_val < 0 else 0
    adj_h, adj_a = home_val + offset, away_val + offset
    total = adj_h + adj_a
    p1 = (adj_h / total * 100) if total > 0 else 50
    p2 = (adj_a / total * 100) if total > 0 else 50
    
    h_str = f"{home_val * 100:.1f}%" if is_percentage else f"{home_val:.2f}"
    a_str = f"{away_val * 100:.1f}%" if is_percentage else f"{away_val:.2f}"
    
    html = f"""
    <div style="margin-bottom: 24px; font-family: 'Roboto', sans-serif;">
        <div style="display: flex; justify-content: space-between; font-size: 16px; font-weight: 700; margin-bottom: 8px;">
            <span style="color: #00FF85; min-width: 50px; text-align: left;">{h_str}</span>
            <span style="color: #FFFFFF; font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 1px; font-size: 14px;">{label}</span>
            <span style="color: #E90052; min-width: 50px; text-align: right;">{a_str}</span>
        </div>
        <div style="display: flex; width: 100%; height: 8px; background-color: #240024; border-radius: 0px; overflow: hidden;">
            <div style="width: 50%; display: flex; justify-content: flex-end; padding-right: 1px;"><div style="width: {p1}%; background-color: #00FF85;"></div></div>
            <div style="width: 50%; display: flex; justify-content: flex-start; padding-left: 1px;"><div style="width: {p2}%; background-color: #E90052;"></div></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# 5. APP EXECUTION & UI LAYOUT
# ==========================================
out = load_and_preprocess_data()
if out[0] is None:
    st.error("⚠️ Could not load data from the internet or local backups. Please check your internet connection.")
    st.stop()
model_pure, model_hybrid, latest_stats, active_teams, pure_features, hybrid_features, importances_pure, importances_hybrid = out

with st.sidebar:
    st.header(t("settings"))
    sel_lang = st.selectbox("Language / Język", ["English", "Polski"], index=0 if st.session_state.lang == 'en' else 1)
    sel_curr = st.selectbox("Currency / Waluta", ["GBP (£)", "PLN (zł)"], index=0 if st.session_state.currency == '£' else 1)
    
    st.toggle(t("toggle_odds"), value=st.session_state.bookies_toggle, help=t("toggle_help"), key="bookies_toggle")
    
    new_lang = 'en' if sel_lang == "English" else 'pl'
    new_curr = '£' if sel_curr == "GBP (£)" else 'zł'
    if new_lang != st.session_state.lang or new_curr != st.session_state.currency:
        st.session_state.lang = new_lang
        st.session_state.currency = new_curr
        st.rerun()

    st.markdown("---")
    st.header(t("risk_management"))
    min_ev = st.slider(
        t("min_ev_lbl"), 
        min_value=0.0, 
        max_value=5.0, 
        value=st.session_state.ev_threshold, 
        step=0.1, 
        help=t("min_ev_help"), 
        key="ev_threshold"
    )

    st.markdown("---")
    st.header(t("match_setup"))
    
    placeholder = ""
    dropdown_options = [placeholder] + active_teams
    
    home_idx = dropdown_options.index(st.session_state.home_team_sel) if st.session_state.home_team_sel in dropdown_options else 0
    away_idx = dropdown_options.index(st.session_state.away_team_sel) if st.session_state.away_team_sel in dropdown_options else 0

    home_team = st.selectbox(t("home"), dropdown_options, index=home_idx, key="home_team_sel")
    away_team = st.selectbox(t("away"), dropdown_options, index=away_idx, key="away_team_sel")
    
    st.markdown("---")
    st.header(t("odds"))
    odds_h = st.number_input(f"🏠 {t('win')}", min_value=1.01, value=st.session_state.odds_h_val, step=0.05, placeholder="Enter odds...", key="odds_h_val")
    odds_d = st.number_input(f"🤝 {t('draw')}", min_value=1.01, value=st.session_state.odds_d_val, step=0.05, placeholder="Enter odds...", key="odds_d_val")
    odds_a = st.number_input(f"✈️ {t('win')}", min_value=1.01, value=st.session_state.odds_a_val, step=0.05, placeholder="Enter odds...", key="odds_a_val")

# Setup Main Tabs
tab_match, tab_model = st.tabs([t("tab_match"), t("tab_model")])

# --------------------------
# TAB 1: MATCH CENTER
# --------------------------
with tab_match:
    if home_team == "" or away_team == "":
        st.info(t("select_teams"))
    elif home_team == away_team:
        st.error(t("same_teams"))
    elif st.session_state.bookies_toggle and (odds_h is None or odds_d is None or odds_a is None):
        st.warning(t("missing_odds"))
    else:
        h_stat, a_stat = latest_stats[home_team], latest_stats[away_team]

        if st.session_state.bookies_toggle:
            active_model = model_hybrid
            input_vector = pd.DataFrame([{
                "H_Momentum": h_stat["Momentum"], "H_GD": h_stat["GD"], "H_SH_Threat": h_stat["SH_Threat"], "H_Conversion": h_stat["Conversion"], "H_Corn": h_stat["Corn"],
                "H_Overall_Pts_5": h_stat["Overall_Pts_5"], "H_Overall_GD_5": h_stat["Overall_GD_5"],
                "A_Momentum": a_stat["Momentum"], "A_GD": a_stat["GD"], "A_SH_Threat": a_stat["SH_Threat"], "A_Conversion": a_stat["Conversion"], "A_Corn": a_stat["Corn"],
                "A_Overall_Pts_5": a_stat["Overall_Pts_5"], "A_Overall_GD_5": a_stat["Overall_GD_5"],
                "Implied_H": 1 / odds_h, "Implied_D": 1 / odds_d, "Implied_A": 1 / odds_a
            }])[hybrid_features]
        else:
            active_model = model_pure
            input_vector = pd.DataFrame([{
                "H_Momentum": h_stat["Momentum"], "H_GD": h_stat["GD"], "H_SH_Threat": h_stat["SH_Threat"], "H_Conversion": h_stat["Conversion"], "H_Corn": h_stat["Corn"],
                "H_Overall_Pts_5": h_stat["Overall_Pts_5"], "H_Overall_GD_5": h_stat["Overall_GD_5"],
                "A_Momentum": a_stat["Momentum"], "A_GD": a_stat["GD"], "A_SH_Threat": a_stat["SH_Threat"], "A_Conversion": a_stat["Conversion"], "A_Corn": a_stat["Corn"],
                "A_Overall_Pts_5": a_stat["Overall_Pts_5"], "A_Overall_GD_5": a_stat["Overall_GD_5"]
            }])[pure_features]

        probs = active_model.predict_proba(input_vector)[0]
        
        if odds_h and odds_d and odds_a:
            wager = 10
            ev_h = (probs[2] * (odds_h * wager - wager)) - ((1 - probs[2]) * wager)
            ev_d = (probs[1] * (odds_d * wager - wager)) - ((1 - probs[1]) * wager)
            ev_a = (probs[0] * (odds_a * wager - wager)) - ((1 - probs[0]) * wager)
        else:
            ev_h = ev_d = ev_a = 0.0

        col1, col2 = st.columns([1, 1.2], gap="large")

        with col1:
            st.markdown(f"<h2>{t('h2h')} <span style='color:#00FF85;'>{home_team}</span> vs <span style='color:#E90052;'>{away_team}</span></h2>", unsafe_allow_html=True)
            render_stat_bar(t("momentum"), h_stat['Momentum'], a_stat['Momentum'])
            render_stat_bar(t("gd"), h_stat['GD'], a_stat['GD'])
            render_stat_bar(t("sh_threat"), h_stat['SH_Threat'], a_stat['SH_Threat'])
            render_stat_bar(t("conversion"), h_stat['Conversion'], a_stat['Conversion'], is_percentage=True)
            render_stat_bar(t("corners"), h_stat['Corn'], a_stat['Corn'])
            render_stat_bar(t("overall_pts"), h_stat['Overall_Pts_5'], a_stat['Overall_Pts_5'])
            render_stat_bar(t("overall_gd"), h_stat['Overall_GD_5'], a_stat['Overall_GD_5'])
            
            st.write("") 
            with st.expander(t("stat_glossary_title")):
                st.markdown(f"""
                - **{t('momentum')}**: {t('def_momentum')}
                - **{t('gd')}**: {t('def_gd')}
                - **{t('sh_threat')}**: {t('def_sh_threat')}
                - **{t('conversion')}**: {t('def_conversion')}
                - **{t('overall_pts')}**: {t('def_overall_pts')}
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"<h2>{t('forecast')}</h2>", unsafe_allow_html=True)
            
            def render_ev_card(team_name, prob, odds, ev):
                # Only highlight as a value bet if the EV clears the minimum threshold slider!
                ev_class = "value-bet" if (ev >= st.session_state.ev_threshold and odds is not None) else "bad-bet"
                ev_sign = "+" if ev > 0 else ""
                sym = st.session_state.currency
                ev_string = f"{sym}{ev:.2f}" if sym == "£" else f"{ev:.2f} {sym}"
                
                odds_display = f"{odds:.2f}" if odds else "—"
                ev_display = f"{ev_sign}{ev_string}" if odds else "—"
                
                st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 20px;">
                        <h3 style="margin-top: 0; color: #FFFFFF; font-size: 1.5rem;">{team_name}</h3>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                            <div style="text-align: left;">
                                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 4px; font-weight: 700; font-family: 'Oswald', sans-serif;">{t('true_prob')}</p>
                                <h2 style="margin-top: 0; color: #FFFFFF;">{prob*100:.1f}%</h2>
                            </div>
                            <div style="text-align: center;">
                                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 4px; font-weight: 700; font-family: 'Oswald', sans-serif;">{t('odds_lbl')}</p>
                                <h2 style="margin-top: 0; color: #FFFFFF;">{odds_display}</h2>
                            </div>
                            <div style="text-align: right;">
                                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 4px; font-weight: 700; font-family: 'Oswald', sans-serif;">EV (10 {sym})</p>
                                <h2 class="{ev_class}" style="margin-top: 0;">{ev_display}</h2>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            render_ev_card(f"🏠 {home_team}", probs[2], odds_h, ev_h)
            render_ev_card(f"🤝 {t('draw').upper()}", probs[1], odds_d, ev_d)
            render_ev_card(f"✈️ {away_team}", probs[0], odds_a, ev_a)

# --------------------------
# TAB 2: ABOUT MODEL
# --------------------------
with tab_model:
    st.markdown(f"<h2>{t('model_title')}</h2>", unsafe_allow_html=True)
    
    col_desc1, col_desc2 = st.columns([1, 1])
    
    with col_desc1:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #00FF85; margin-top:0;">🤖 Hybrid Extreme Gradient Boosting (XGBoost)</h3>
            <p>This upgraded model builds <b>100 sequential decision trees</b> utilizing a <b>Hybrid Architecture</b>. It blends long-term venue-specific tactical metrics (how teams play specifically at home vs. away) with short-term overall form (last 5 games total points and goal difference).</p>
            <p>This ensures the model captures structural stadium advantages while reacting instantly to sudden squad crises or managerial bounces.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Engine Parameters")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Trees (Estimators)", "100")
        p2.metric("Learning Rate", "0.05")
        p3.metric("Max Depth", "5")
        p4.metric("Subsample", "0.9")

    with col_desc2:
        st.markdown(f"""
        <div class="info-card">
            <h3 style="color: #E90052; margin-top:0;">{t('ev_title')}</h3>
            <p>Bookmakers price markets based on public sentiment, creating pricing errors. <b>Expected Value (EV)</b> measures your average long-term profit per bet given the true mathematical probability.</p>
            <hr>
            <p style="font-family: monospace; font-size: 1.1rem; text-align: center; color: #00FF85;">
            EV = (True Prob * Profit) - (Loss Prob * Stake)
            </p>
            <hr>
            <p>When the model's true probability exceeds the implied probability from bookmaker odds, it highlights a profitable <span style="color: #00FF85; font-weight: bold;">+EV Value Bet</span>.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<h2>{t('features_title')}</h2>", unsafe_allow_html=True)
    st.write("This chart dynamically updates based on whether you have included Bookmaker Odds via the sidebar toggle.")
    
    active_importances = importances_hybrid if st.session_state.bookies_toggle else importances_pure
    clean_imps = active_importances.rename(index=lambda x: x.replace('H_', 'Home ').replace('A_', 'Away ').replace('_', ' '))
    st.bar_chart(clean_imps, height=400, color="#963CFF")