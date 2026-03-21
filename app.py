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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Force white everywhere */
html, body, [class*="css"], .stApp,
.stApp > div, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #ffffff !important;
    color: #111111 !important;
}

/* Hide Streamlit header/toolbar completely */
[data-testid="stHeader"],
[data-testid="stToolbar"],
.stDeployButton,
header { display: none !important; }

#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }

.block-container {
    max-width: 680px;
    padding-top: 2.5rem !important;
    padding-bottom: 4rem;
}

/* Page title */
.page-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #111111;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
}
.page-sub {
    font-size: 0.68rem;
    color: #bbbbbb;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

/* Gender buttons */
.stButton > button {
    border-radius: 20px !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    padding: 0.28rem 1rem !important;
    border: 1.5px solid #e0e0e0 !important;
    transition: all 0.15s !important;
}
.stButton > button[kind="primary"] {
    background: #111111 !important;
    color: #ffffff !important;
    border-color: #111111 !important;
}
.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #666666 !important;
}

/* Labels */
label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #aaaaaa !important;
}

/* Selectboxes */
.stSelectbox > div > div {
    background-color: #f8f8f8 !important;
    border: 1.5px solid #e8e8e8 !important;
    border-radius: 8px !important;
    color: #111111 !important;
    font-size: 0.88rem !important;
}

.divider {
    border: none;
    border-top: 1px solid #f0f0f0;
    margin: 1.25rem 0;
}

/* ESPN-style probability display */
.espn-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}
.espn-pct {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    color: #111111;
}
.espn-pct-dim {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    color: #dddddd;
}
.espn-name {
    font-size: 0.72rem;
    font-weight: 600;
    color: #555555;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.espn-name-dim {
    font-size: 0.72rem;
    font-weight: 500;
    color: #cccccc;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.side-left  { text-align: left; }
.side-right { text-align: right; }

.fav-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #111111;
    margin-right: 4px;
    vertical-align: middle;
}

/* Bottom info */
.info-text {
    font-size: 0.68rem;
    color: #bbbbbb;
    line-height: 1.7;
    text-align: center;
    margin-top: 1.5rem;
}
.footer-text {
    text-align: center;
    font-size: 0.65rem;
    color: #cccccc;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    sub     = pd.read_csv("submissions/stage2/submission_ensemble_v1.csv")
    mteams  = pd.read_csv("data/MTeams.csv")[["TeamID", "TeamName"]]
    wteams  = pd.read_csv("data/WTeams.csv")[["TeamID", "TeamName"]]

    parts       = sub["ID"].str.split("_", expand=True)
    sub["Season"] = parts[0].astype(int)
    sub["TeamA"]  = parts[1].astype(int)
    sub["TeamB"]  = parts[2].astype(int)

    lookup  = dict(zip(zip(sub["TeamA"], sub["TeamB"]), sub["Pred"]))

    s26     = sub[sub["Season"] == 2026]
    all_ids = set(s26["TeamA"]) | set(s26["TeamB"])
    m_ids   = {t for t in all_ids if t < 2000}
    w_ids   = {t for t in all_ids if t >= 2000}

    m_map   = mteams[mteams["TeamID"].isin(m_ids)].set_index("TeamID")["TeamName"].to_dict()
    w_map   = wteams[wteams["TeamID"].isin(w_ids)].set_index("TeamID")["TeamName"].to_dict()
    return lookup, m_map, w_map


def get_prob(lookup, a, b):
    lo, hi = sorted([a, b])
    p = lookup.get((lo, hi))
    if p is None:
        return None
    return p if a == lo else 1 - p


def make_donut(prob_a, prob_b, color_a="#2563eb", color_b="#e11d48"):
    fig = go.Figure(go.Pie(
        values=[prob_a, prob_b],
        hole=0.72,
        marker_colors=[color_a, color_b],
        textinfo="none",
        hovertemplate="%{value:.1%}<extra></extra>",
        sort=False,
        direction="clockwise",
        rotation=90,
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        width=180, height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text="VS",
            x=0.5, y=0.5,
            font=dict(size=12, color="#cccccc", family="Inter"),
            showarrow=False,
        )],
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">NCAA March Madness 2026</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Win probability · LGB + CatBoost ensemble · CV Brier 0.1579</p>', unsafe_allow_html=True)

try:
    lookup, m_map, w_map = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

# Session state
for key, val in [("gender", "Men's"), ("team_a_idx", 0), ("team_b_idx", 1)]:
    if key not in st.session_state:
        st.session_state[key] = val

# Gender toggle
c1, c2, _ = st.columns([1.1, 1.3, 4.5])
with c1:
    if st.button("Men's",
                 type="primary" if st.session_state.gender == "Men's" else "secondary",
                 use_container_width=True):
        st.session_state.gender = "Men's"; st.rerun()
with c2:
    if st.button("Women's",
                 type="primary" if st.session_state.gender == "Women's" else "secondary",
                 use_container_width=True):
        st.session_state.gender = "Women's"; st.rerun()

team_map    = m_map if st.session_state.gender == "Men's" else w_map
sorted_teams = sorted(team_map.items(), key=lambda x: x[1])
name_to_id  = {n: tid for tid, n in sorted_teams}
team_names  = [n for _, n in sorted_teams]

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    a_name = st.selectbox("Team A", team_names,
                          index=min(st.session_state.team_a_idx, len(team_names)-1))
with col2:
    b_name = st.selectbox("Team B", team_names,
                          index=min(st.session_state.team_b_idx, len(team_names)-1))

st.session_state.team_a_idx = team_names.index(a_name)
st.session_state.team_b_idx = team_names.index(b_name)

if a_name == b_name:
    st.warning("Select two different teams.")
    st.stop()

prob_a = get_prob(lookup, name_to_id[a_name], name_to_id[b_name])
if prob_a is None:
    st.error("Matchup not found.")
    st.stop()

prob_b  = 1 - prob_a
a_wins  = prob_a >= prob_b
COLOR_A = "#2563eb"
COLOR_B = "#e11d48"

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── ESPN layout: pct LEFT | donut CENTER | pct RIGHT ─────────────────────────
col_l, col_c, col_r = st.columns([2, 2, 2])

# Left side — Team A
with col_l:
    pct_class  = "espn-pct"     if a_wins else "espn-pct-dim"
    name_class = "espn-name"    if a_wins else "espn-name-dim"
    dot        = f'<span class="fav-dot" style="background:{COLOR_A}"></span>' if a_wins else \
                 f'<span class="fav-dot" style="background:{COLOR_A};opacity:0.25"></span>'
    st.markdown(f"""
    <div class="side-left">
      <div class="{pct_class}" style="color:{'#111111' if a_wins else '#dddddd'}">{prob_a*100:.0f}%</div>
      <div class="{name_class}">{dot}{a_name}</div>
    </div>
    """, unsafe_allow_html=True)

# Center — donut
with col_c:
    fig = make_donut(prob_a, prob_b, COLOR_A, COLOR_B)
    st.plotly_chart(fig, use_container_width=False,
                    config={"displayModeBar": False})

# Right side — Team B
with col_r:
    pct_class  = "espn-pct"  if not a_wins else "espn-pct-dim"
    name_class = "espn-name" if not a_wins else "espn-name-dim"
    dot        = f'<span class="fav-dot" style="background:{COLOR_B}"></span>' if not a_wins else \
                 f'<span class="fav-dot" style="background:{COLOR_B};opacity:0.25"></span>'
    st.markdown(f"""
    <div class="side-right">
      <div class="{pct_class}" style="color:{'#111111' if not a_wins else '#dddddd'}">{prob_b*100:.0f}%</div>
      <div class="{name_class}">{b_name}{dot}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<p class="info-text">
  Gender-specific LightGBM + CatBoost ensemble · NCAA tournament data 2003–2025<br>
  Features: Elo, seed, SOS, Four Factors, Massey Ordinals
</p>
<p class="footer-text">Xinwei Huang · Haoran Zhang</p>
""", unsafe_allow_html=True)
