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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff;
    color: #111111;
}

.block-container {
    max-width: 640px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

h1 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #111111 !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.25rem !important;
}

.meta {
    font-size: 0.78rem;
    color: #999999;
    margin-bottom: 2.5rem;
}

label {
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
    font-size: 0.9rem !important;
}

.stRadio > div {
    gap: 0.5rem;
}
.stRadio > div > label {
    background: #f7f7f7 !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 6px !important;
    padding: 0.4rem 1rem !important;
    color: #111111 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
    text-transform: none !important;
}

hr {
    border: none;
    border-top: 1px solid #f0f0f0;
    margin: 1.5rem 0;
}

.result-wrap {
    margin-top: 2rem;
}

.result-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 0;
    border-bottom: 1px solid #f0f0f0;
}

.result-row:first-child {
    border-top: 1px solid #f0f0f0;
}

.team-label {
    font-size: 0.95rem;
    font-weight: 500;
    color: #111111;
}

.team-label.loser {
    color: #bbbbbb;
}

.prob-val {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111111;
}

.prob-val.loser {
    color: #bbbbbb;
    font-weight: 400;
}

.winner-tag {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #111111;
    background: #f0f0f0;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    margin-left: 0.6rem;
    vertical-align: middle;
}

.bar-wrap {
    margin: 1.5rem 0;
    height: 2px;
    background: #f0f0f0;
    border-radius: 2px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: #111111;
    border-radius: 2px;
}

.info {
    font-size: 0.75rem;
    color: #aaaaaa;
    line-height: 1.7;
    margin-top: 2rem;
}

.footer {
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

st.markdown(f"""
<div class="result-wrap">
    <div class="result-row">
        <span class="team-label {'loser' if not a_wins else ''}">
            {a_name}{'<span class="winner-tag">Favored</span>' if a_wins else ''}
        </span>
        <span class="prob-val {'loser' if not a_wins else ''}">{prob_a*100:.1f}%</span>
    </div>
    <div class="result-row">
        <span class="team-label {'loser' if a_wins else ''}">
            {b_name}{'<span class="winner-tag">Favored</span>' if not a_wins else ''}
        </span>
        <span class="prob-val {'loser' if a_wins else ''}">{prob_b*100:.1f}%</span>
    </div>
</div>
<div class="bar-wrap">
    <div class="bar-fill" style="width:{prob_a*100:.1f}%"></div>
</div>
<p class="info">
    Gender-specific LightGBM + CatBoost ensemble trained on NCAA tournament data 2003–2025.
    Features: Elo ratings, seed, strength of schedule, Four Factors, Massey Ordinals.
</p>
<p class="footer">Xinwei Huang · Haoran Zhang</p>
""", unsafe_allow_html=True)
