import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NCAA 2026",
    page_icon="🏀",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* Force light mode regardless of system preference */
:root {
    color-scheme: light only;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #ffffff !important;
    color: #111111 !important;
}

.stApp {
    background-color: #ffffff !important;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
}

.block-container {
    max-width: 640px;
    padding-top: 3rem;
    padding-bottom: 4rem;
    background-color: #ffffff !important;
}

h1, h2, h3, p {
    color: #111111 !important;
}

h1 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.25rem !important;
}

.meta {
    font-size: 0.78rem;
    color: #999999;
    margin-bottom: 2.5rem;
}

label, .stRadio label {
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #999999 !important;
}

.stSelectbox > div > div {
    background-color: #f7f7f7 !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 6px !important;
    color: #111111 !important;
}

hr {
    border: none;
    border-top: 1px solid #eeeeee;
    margin: 1.5rem 0;
}

.result-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 0;
    border-bottom: 1px solid #eeeeee;
}

.result-row:first-child {
    border-top: 1px solid #eeeeee;
}

.t-name {
    font-size: 0.95rem;
    font-weight: 500;
    color: #111111;
}

.t-name-dim {
    font-size: 0.95rem;
    font-weight: 400;
    color: #cccccc;
}

.t-prob {
    font-size: 1.05rem;
    font-weight: 600;
    color: #111111;
}

.t-prob-dim {
    font-size: 1.05rem;
    font-weight: 400;
    color: #cccccc;
}

.tag {
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555555;
    background: #f0f0f0;
    border-radius: 4px;
    padding: 0.12rem 0.45rem;
    margin-left: 0.6rem;
    vertical-align: middle;
}

.bar-wrap {
    margin: 1.25rem 0 2rem 0;
    height: 2px;
    background: #eeeeee;
    border-radius: 2px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: #111111;
    border-radius: 2px;
}

.info-text {
    font-size: 0.75rem;
    color: #aaaaaa;
    line-height: 1.7;
}

.footer-text {
    text-align: center;
    font-size: 0.7rem;
    color: #cccccc;
    margin-top: 3rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    sub = pd.read_csv("submissions/stage2/submission_ensemble_v1.csv")
    mteams = pd.read_csv("data/MTeams.csv")[["TeamID", "TeamName"]]
    wteams = pd.read_csv("data/WTeams.csv")[["TeamID", "TeamName"]]

    parts = sub["ID"].str.split("_", expand=True)
    sub["Season"] = parts[0].astype(int)
    sub["TeamA"]  = parts[1].astype(int)
    sub["TeamB"]  = parts[2].astype(int)

    lookup = dict(zip(zip(sub["TeamA"], sub["TeamB"]), sub["Pred"]))

    s26 = sub[sub["Season"] == 2026]
    all_ids = set(s26["TeamA"]).union(set(s26["TeamB"]))
    m_ids = {t for t in all_ids if t < 2000}
    w_ids = {t for t in all_ids if t >= 2000}

    m_map = mteams[mteams["TeamID"].isin(m_ids)].set_index("TeamID")["TeamName"].to_dict()
    w_map = wteams[wteams["TeamID"].isin(w_ids)].set_index("TeamID")["TeamName"].to_dict()

    return lookup, m_map, w_map


def get_prob(lookup, a, b):
    lo, hi = sorted([a, b])
    p = lookup.get((lo, hi), None)
    if p is None:
        return None
    return p if a == lo else 1 - p


st.markdown('<h1>NCAA March Madness 2026</h1>', unsafe_allow_html=True)
st.markdown('<p class="meta">Win probability · LGB + CatBoost ensemble · CV Brier 0.1579</p>', unsafe_allow_html=True)

try:
    lookup, m_map, w_map = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

gender = st.radio("Division", ["Men's", "Women's"], horizontal=True, label_visibility="collapsed")
team_map = m_map if gender == "Men's" else w_map

sorted_teams = sorted(team_map.items(), key=lambda x: x[1])
name_to_id   = {name: tid for tid, name in sorted_teams}
team_names   = [name for _, name in sorted_teams]

st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    a_name = st.selectbox("Team A", team_names, index=0)
with col2:
    default_b = 1 if team_names[0] == a_name else 0
    b_name = st.selectbox("Team B", team_names, index=default_b)

if a_name == b_name:
    st.warning("Select two different teams.")
    st.stop()

prob_a = get_prob(lookup, name_to_id[a_name], name_to_id[b_name])

if prob_a is None:
    st.error("Matchup not found in 2026 submission.")
    st.stop()

prob_b = 1 - prob_a
a_wins = prob_a >= prob_b

# Build HTML safely without f-string class interpolation issues
a_name_class  = "t-name"    if a_wins  else "t-name-dim"
b_name_class  = "t-name-dim" if a_wins else "t-name"
a_prob_class  = "t-prob"    if a_wins  else "t-prob-dim"
b_prob_class  = "t-prob-dim" if a_wins else "t-prob"
a_tag = '<span class="tag">Favored</span>' if a_wins  else ""
b_tag = '<span class="tag">Favored</span>' if not a_wins else ""

st.markdown(f"""
<div>
  <div class="result-row">
    <span class="{a_name_class}">{a_name}{a_tag}</span>
    <span class="{a_prob_class}">{prob_a*100:.1f}%</span>
  </div>
  <div class="result-row">
    <span class="{b_name_class}">{b_name}{b_tag}</span>
    <span class="{b_prob_class}">{prob_b*100:.1f}%</span>
  </div>
  <div class="bar-wrap">
    <div class="bar-fill" style="width:{prob_a*100:.1f}%"></div>
  </div>
  <p class="info-text">
    Gender-specific LightGBM + CatBoost ensemble trained on NCAA tournament data 2003–2025.
    Features: Elo ratings, seed, strength of schedule, Four Factors, Massey Ordinals.
  </p>
  <p class="footer-text">Xinwei Huang · Haoran Zhang</p>
</div>
""", unsafe_allow_html=True)
