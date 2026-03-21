import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="NCAA 2026",
    page_icon="🏀",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #ffffff !important;
    color: #111111 !important;
}
.stApp { background-color: #ffffff !important; }
.block-container {
    max-width: 660px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    background-color: #ffffff !important;
}

.page-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111111;
    margin-bottom: 0.15rem;
}
.page-sub {
    font-size: 0.7rem;
    color: #aaaaaa;
    letter-spacing: 0.05em;
    margin-bottom: 1.8rem;
}

label {
    font-size: 0.67rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: #aaaaaa !important;
}

.stSelectbox > div > div {
    background-color: #f8f8f8 !important;
    border: 1.5px solid #e8e8e8 !important;
    border-radius: 8px !important;
    color: #111111 !important;
}

.divider {
    border: none;
    border-top: 1px solid #efefef;
    margin: 1.25rem 0;
}

.result-card {
    background: #ffffff;
    border: 1.5px solid #ebebeb;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.25rem;
}

.team-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-top: 1px solid #f3f3f3;
}

.dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 7px;
    vertical-align: middle;
}

.t-name { font-size: 0.85rem; font-weight: 500; color: #111111; }
.t-name-dim { font-size: 0.85rem; font-weight: 400; color: #cccccc; }
.t-pct { font-size: 0.9rem; font-weight: 600; color: #111111; }
.t-pct-dim { font-size: 0.9rem; font-weight: 400; color: #cccccc; }

.fav-tag {
    display: inline-block;
    font-size: 0.58rem; font-weight: 700;
    letter-spacing: 0.09em; text-transform: uppercase;
    color: #fff; background: #111111;
    border-radius: 20px; padding: 0.1rem 0.45rem;
    margin-left: 0.4rem; vertical-align: middle;
}

.info-text {
    font-size: 0.7rem; color: #bbbbbb;
    line-height: 1.7; margin-top: 1rem;
    padding-top: 0.85rem; border-top: 1px solid #f3f3f3;
}

.footer-text {
    text-align: center; font-size: 0.67rem;
    color: #cccccc; margin-top: 2.5rem;
}

/* Style the primary/secondary buttons better */
.stButton > button {
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.3rem 1.1rem !important;
    border: 1.5px solid #e0e0e0 !important;
}
.stButton > button[kind="primary"] {
    background: #111111 !important;
    color: #ffffff !important;
    border-color: #111111 !important;
}
.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #555555 !important;
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


def make_donut(prob_a, prob_b, name_a, name_b):
    COLOR_A = "#2563eb"
    COLOR_B = "#e11d48"

    # Shorten long names
    def short(n): return n[:14] + "…" if len(n) > 14 else n

    fig = go.Figure(go.Pie(
        values=[prob_a, prob_b],
        labels=[short(name_a), short(name_b)],
        hole=0.68,
        marker_colors=[COLOR_A, COLOR_B],
        textinfo="none",
        hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        sort=False,
        direction="clockwise",
        rotation=90,
    ))

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        width=220, height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text="VS",
            x=0.5, y=0.5,
            font=dict(size=13, color="#aaaaaa", family="Inter"),
            showarrow=False,
        )],
    )
    return fig


# ── Page ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">NCAA March Madness 2026</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">WIN PROBABILITY · LGB + CATBOOST ENSEMBLE · CV BRIER 0.1579</p>', unsafe_allow_html=True)

try:
    lookup, m_map, w_map = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

# Gender toggle with session state
if "gender" not in st.session_state:
    st.session_state.gender = "Men's"
if "team_a_idx" not in st.session_state:
    st.session_state.team_a_idx = 0
if "team_b_idx" not in st.session_state:
    st.session_state.team_b_idx = 1

col_m, col_w, _ = st.columns([1.1, 1.3, 4])
with col_m:
    if st.button("Men's",
                 type="primary" if st.session_state.gender == "Men's" else "secondary",
                 use_container_width=True):
        st.session_state.gender = "Men's"
        st.rerun()
with col_w:
    if st.button("Women's",
                 type="primary" if st.session_state.gender == "Women's" else "secondary",
                 use_container_width=True):
        st.session_state.gender = "Women's"
        st.rerun()

team_map = m_map if st.session_state.gender == "Men's" else w_map
sorted_teams = sorted(team_map.items(), key=lambda x: x[1])
name_to_id   = {name: tid for tid, name in sorted_teams}
team_names   = [name for _, name in sorted_teams]

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    a_idx = min(st.session_state.team_a_idx, len(team_names) - 1)
    a_name = st.selectbox("Team A", team_names, index=a_idx)
with col2:
    b_idx = min(st.session_state.team_b_idx, len(team_names) - 1)
    b_name = st.selectbox("Team B", team_names, index=b_idx)

st.session_state.team_a_idx = team_names.index(a_name)
st.session_state.team_b_idx = team_names.index(b_name)

if a_name == b_name:
    st.warning("Select two different teams.")
    st.stop()

prob_a = get_prob(lookup, name_to_id[a_name], name_to_id[b_name])
if prob_a is None:
    st.error("Matchup not found in 2026 submission.")
    st.stop()

prob_b = 1 - prob_a
a_wins = prob_a >= prob_b

# Donut chart centered
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    fig = make_donut(prob_a, prob_b, a_name, b_name)
    st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})

# Result rows
a_tag  = '<span class="fav-tag">Favored</span>' if a_wins else ""
b_tag  = '<span class="fav-tag">Favored</span>' if not a_wins else ""

st.markdown(f"""
<div class="result-card">
  <div class="team-row">
    <span>
      <span class="dot" style="background:#2563eb"></span>
      <span class="{'t-name' if a_wins else 't-name-dim'}">{a_name}{a_tag}</span>
    </span>
    <span class="{'t-pct' if a_wins else 't-pct-dim'}">{prob_a*100:.1f}%</span>
  </div>
  <div class="team-row">
    <span>
      <span class="dot" style="background:#e11d48"></span>
      <span class="{'t-name' if not a_wins else 't-name-dim'}">{b_name}{b_tag}</span>
    </span>
    <span class="{'t-pct' if not a_wins else 't-pct-dim'}">{prob_b*100:.1f}%</span>
  </div>
  <p class="info-text">
    Gender-specific LightGBM + CatBoost ensemble · NCAA tournament data 2003–2025 ·
    Features: Elo, seed, SOS, Four Factors, Massey Ordinals
  </p>
</div>
<p class="footer-text">Xinwei Huang · Haoran Zhang</p>
""", unsafe_allow_html=True)
