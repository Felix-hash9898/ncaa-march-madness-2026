import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NCAA 2026 Win Probability",
    page_icon="🏀",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Main container */
.block-container {
    max-width: 720px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Title */
h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3rem !important;
    letter-spacing: 0.05em;
    color: #ffffff !important;
    margin-bottom: 0 !important;
    line-height: 1.1 !important;
}

h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.04em;
    color: #ffffff !important;
}

/* Subtitle */
.subtitle {
    color: #6b6b80;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
    margin-top: 0.25rem;
}

/* Selectbox label */
label {
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6b6b80 !important;
    font-weight: 600 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #13131f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 6px !important;
    color: #e8e8f0 !important;
}

/* Radio */
.stRadio > div {
    gap: 1rem;
}
.stRadio > div > label {
    background: #13131f;
    border: 1px solid #2a2a3d;
    border-radius: 6px;
    padding: 0.5rem 1.2rem !important;
    color: #e8e8f0 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    text-transform: none !important;
    cursor: pointer;
    transition: border-color 0.2s;
}
.stRadio > div > label:hover {
    border-color: #ff6b35;
}

/* Divider */
hr {
    border-color: #1e1e2e;
    margin: 2rem 0;
}

/* Result cards */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1.5rem 0;
}

.team-card {
    background: #13131f;
    border: 1px solid #2a2a3d;
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.team-card.winner {
    border-color: #ff6b35;
    background: linear-gradient(135deg, #1a1025, #13131f);
}

.team-card.winner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff6b35, #ffaa35);
}

.team-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 0.05em;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.prob-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    line-height: 1;
    color: #ff6b35;
}

.prob-number.underdog {
    color: #6b6b80;
}

.prob-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6b80;
    margin-top: 0.25rem;
}

/* Progress bar override */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #ff6b35, #ffaa35) !important;
}

/* Matchup pill */
.vs-pill {
    text-align: center;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.15em;
    color: #3a3a52;
    margin: 0.5rem 0;
}

/* Info box */
.info-box {
    background: #13131f;
    border-left: 3px solid #ff6b35;
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #9999b0;
    margin-top: 1.5rem;
    line-height: 1.6;
}

/* Footer */
.footer {
    text-align: center;
    color: #3a3a52;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    sub = pd.read_csv("submissions/stage2/submission_ensemble_v1.csv")
    mteams = pd.read_csv("data/MTeams.csv")[["TeamID", "TeamName"]]
    wteams = pd.read_csv("data/WTeams.csv")[["TeamID", "TeamName"]]

    # Parse Season, TeamA, TeamB from ID
    parts = sub["ID"].str.split("_", expand=True)
    sub["Season"] = parts[0].astype(int)
    sub["TeamA"]  = parts[1].astype(int)
    sub["TeamB"]  = parts[2].astype(int)

    # Build lookup dict: (teamA, teamB) -> Pred  (teamA < teamB always)
    lookup = dict(zip(zip(sub["TeamA"], sub["TeamB"]), sub["Pred"]))

    # 2026 teams in submission
    s26 = sub[sub["Season"] == 2026]
    m_ids = set(s26["TeamA"]).union(set(s26["TeamB"]))
    m_ids = {t for t in m_ids if t < 2000}
    w_ids = {t for t in m_ids if t >= 2000}
    # re-derive
    all_ids = set(s26["TeamA"]).union(set(s26["TeamB"]))
    m_ids_all = {t for t in all_ids if t < 2000}
    w_ids_all = {t for t in all_ids if t >= 2000}

    m_map = mteams[mteams["TeamID"].isin(m_ids_all)].set_index("TeamID")["TeamName"].to_dict()
    w_map = wteams[wteams["TeamID"].isin(w_ids_all)].set_index("TeamID")["TeamName"].to_dict()

    return lookup, m_map, w_map


def get_prob(lookup, team_a_id, team_b_id):
    """Return P(team_a wins). lookup always has lower ID first."""
    lo, hi = sorted([team_a_id, team_b_id])
    p_lo_wins = lookup.get((lo, hi), None)
    if p_lo_wins is None:
        return None
    return p_lo_wins if team_a_id == lo else 1 - p_lo_wins


# ── App ───────────────────────────────────────────────────────────────────────
st.markdown('<h1>NCAA 2026<br>Win Probability</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">March Machine Learning Mania · Kaggle 2026 · LGB + CatBoost Ensemble</p>', unsafe_allow_html=True)

try:
    lookup, m_map, w_map = load_data()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e}\n\nMake sure `submissions/stage2/submission_ensemble_v1.csv`, `data/MTeams.csv`, and `data/WTeams.csv` are in the repo.")
    st.stop()

# Gender selector
gender = st.radio("Division", ["Men's 🏀", "Women's 🏀"], horizontal=True, label_visibility="collapsed")
is_mens = gender.startswith("Men")
team_map = m_map if is_mens else w_map

if not team_map:
    st.warning("No teams found for this division in the submission file.")
    st.stop()

sorted_teams = sorted(team_map.items(), key=lambda x: x[1])  # sort by name
name_to_id   = {name: tid for tid, name in sorted_teams}
team_names   = [name for _, name in sorted_teams]

st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    team_a_name = st.selectbox("Team A", team_names, index=0)
with col2:
    # Default to a different team
    default_b = 1 if team_names[0] == team_a_name else 0
    team_b_name = st.selectbox("Team B", team_names, index=default_b)

if team_a_name == team_b_name:
    st.warning("Select two different teams.")
    st.stop()

team_a_id = name_to_id[team_a_name]
team_b_id = name_to_id[team_b_name]

prob_a = get_prob(lookup, team_a_id, team_b_id)

if prob_a is None:
    st.error("This matchup isn't in the 2026 submission file. Both teams must be from the 2026 season.")
    st.stop()

prob_b = 1 - prob_a
winner = team_a_name if prob_a >= 0.5 else team_b_name

# ── Results ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"### Predicted Outcome")

a_is_winner = prob_a >= prob_b
card_a_class = "team-card winner" if a_is_winner else "team-card"
card_b_class = "team-card winner" if not a_is_winner else "team-card"
prob_a_class = "prob-number" if a_is_winner else "prob-number underdog"
prob_b_class = "prob-number" if not a_is_winner else "prob-number underdog"

st.markdown(f"""
<div class="result-grid">
    <div class="{card_a_class}">
        <div class="team-name">{team_a_name}</div>
        <div class="{prob_a_class}">{prob_a*100:.1f}%</div>
        <div class="prob-label">Win probability</div>
    </div>
    <div class="{card_b_class}">
        <div class="team-name">{team_b_name}</div>
        <div class="{prob_b_class}">{prob_b*100:.1f}%</div>
        <div class="prob-label">Win probability</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Probability bar
st.markdown(f"**{team_a_name}** &nbsp;←&nbsp; probability split &nbsp;→&nbsp; **{team_b_name}**")
st.progress(float(prob_a))

st.markdown(f"""
<div class="info-box">
    🏆 &nbsp;<strong>Predicted winner: {winner}</strong> &nbsp;({max(prob_a, prob_b)*100:.1f}% confidence)<br>
    Predictions from a gender-specific LightGBM + CatBoost ensemble trained on NCAA tournament data 2003–2025.
    Features include Elo ratings, seed, strength of schedule, Four Factors, and Massey Ordinals rankings.
    CV Brier Score: <strong>0.1579</strong> (vs 0.25 naive baseline).
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Xinwei Huang · Haoran Zhang &nbsp;·&nbsp; Kaggle March Machine Learning Mania 2026
</div>
""", unsafe_allow_html=True)
