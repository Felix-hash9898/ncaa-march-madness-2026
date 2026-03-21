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

html, body, [class*="css"], .stApp,
.stApp > div, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #ffffff !important;
    color: #111111 !important;
}
.stApp { background-color: #ffffff !important; }

[data-testid="stHeader"], [data-testid="stToolbar"],
.stDeployButton, header { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }

.block-container {
    max-width: 680px;
    padding-top: 2.5rem !important;
    padding-bottom: 4rem;
}

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
    letter-spacing: 0.04em;
    margin-bottom: 2rem;
}

.stButton > button {
    border-radius: 20px !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    padding: 0.28rem 1rem !important;
    border: 1.5px solid #e0e0e0 !important;
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

label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #aaaaaa !important;
}
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

.pct-left {
    text-align: right;
    padding-right: 0.5rem;
}
.pct-right {
    text-align: left;
    padding-left: 0.5rem;
}
.pct-num {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1;
    color: #111111;
}
.pct-num-dim {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1;
    color: #e0e0e0;
}
.pct-name {
    font-size: 0.68rem;
    font-weight: 600;
    color: #555555;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}
.pct-name-dim {
    font-size: 0.68rem;
    font-weight: 500;
    color: #cccccc;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}

.info-text {
    font-size: 0.68rem;
    color: #bbbbbb;
    line-height: 1.7;
    text-align: center;
    margin-top: 1.25rem;
}
.footer-text {
    text-align: center;
    font-size: 0.65rem;
    color: #cccccc;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    sub    = pd.read_csv("submissions/stage2/submission_ensemble_v1.csv")
    mteams = pd.read_csv("data/MTeams.csv")[["TeamID", "TeamName"]]
    wteams = pd.read_csv("data/WTeams.csv")[["TeamID", "TeamName"]]

    parts        = sub["ID"].str.split("_", expand=True)
    sub["Season"] = parts[0].astype(int)
    sub["TeamA"]  = parts[1].astype(int)
    sub["TeamB"]  = parts[2].astype(int)

    lookup  = dict(zip(zip(sub["TeamA"], sub["TeamB"]), sub["Pred"]))
    s26     = sub[sub["Season"] == 2026]
    all_ids = set(s26["TeamA"]) | set(s26["TeamB"])
    m_ids   = {t for t in all_ids if t < 2000}
    w_ids   = {t for t in all_ids if t >= 2000}

    m_map = mteams[mteams["TeamID"].isin(m_ids)].set_index("TeamID")["TeamName"].to_dict()
    w_map = wteams[wteams["TeamID"].isin(w_ids)].set_index("TeamID")["TeamName"].to_dict()
    return lookup, m_map, w_map


def get_prob(lookup, a, b):
    lo, hi = sorted([a, b])
    p = lookup.get((lo, hi))
    if p is None:
        return None
    return p if a == lo else 1 - p


def make_espn_donut(prob_a, prob_b, name_a, name_b,
                    color_a="#2563eb", color_b="#e11d48"):
    """
    ESPN-style ring:
    - Starts at 12 o'clock
    - Team A fills counterclockwise (left side)
    - Team B fills clockwise (right side)
    - Thin vertical line at center (50/50 marker)
    - Team names flanking inside the ring
    """
    def short(n, k=11):
        return n[:k] + "…" if len(n) > k else n

    fig = go.Figure()

    # Pie: counterclockwise from top → Team A goes left, Team B goes right
    fig.add_trace(go.Pie(
        values=[prob_a, prob_b],
        hole=0.65,
        marker_colors=[color_a, color_b],
        marker=dict(line=dict(color="#ffffff", width=2)),
        textinfo="none",
        hovertemplate="%{value:.1%}<extra></extra>",
        sort=False,
        direction="counterclockwise",
        rotation=90,   # start at 12 o'clock
    ))

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        width=240, height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            # Team A name — left inside ring
            dict(
                text=f"<b>{short(name_a)}</b>",
                x=0.28, y=0.55,
                font=dict(size=9, color=color_a, family="Inter"),
                showarrow=False,
                align="center",
            ),
            # Team B name — right inside ring
            dict(
                text=f"<b>{short(name_b)}</b>",
                x=0.72, y=0.55,
                font=dict(size=9, color=color_b, family="Inter"),
                showarrow=False,
                align="center",
            ),
        ],
        shapes=[
            # Vertical center line (50/50 indicator) — top half
            dict(
                type="line",
                x0=0.5, y0=0.82,
                x1=0.5, y1=0.93,
                xref="paper", yref="paper",
                line=dict(color="#dddddd", width=1.5),
            ),
            # Vertical center line — bottom half
            dict(
                type="line",
                x0=0.5, y0=0.07,
                x1=0.5, y1=0.18,
                xref="paper", yref="paper",
                line=dict(color="#dddddd", width=1.5),
            ),
        ],
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

for key, val in [("gender", "Men's"), ("team_a_idx", 0), ("team_b_idx", 1)]:
    if key not in st.session_state:
        st.session_state[key] = val

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

team_map     = m_map if st.session_state.gender == "Men's" else w_map
sorted_teams = sorted(team_map.items(), key=lambda x: x[1])
name_to_id   = {n: tid for tid, n in sorted_teams}
team_names   = [n for _, n in sorted_teams]

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

# ESPN layout: big % LEFT | donut CENTER | big % RIGHT
col_l, col_c, col_r = st.columns([2, 2.2, 2])

with col_l:
    pc = "pct-num" if a_wins else "pct-num-dim"
    nc = "pct-name" if a_wins else "pct-name-dim"
    st.markdown(f"""
    <div class="pct-left">
      <div class="{pc}" style="color:{COLOR_A if a_wins else '#e0e0e0'}">{prob_a*100:.0f}%</div>
      <div class="{nc}">{a_name}</div>
    </div>""", unsafe_allow_html=True)

with col_c:
    fig = make_espn_donut(prob_a, prob_b, a_name, b_name, COLOR_A, COLOR_B)
    st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})

with col_r:
    pc = "pct-num" if not a_wins else "pct-num-dim"
    nc = "pct-name" if not a_wins else "pct-name-dim"
    st.markdown(f"""
    <div class="pct-right">
      <div class="{pc}" style="color:{COLOR_B if not a_wins else '#e0e0e0'}">{prob_b*100:.0f}%</div>
      <div class="{nc}">{b_name}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<p class="info-text">
  Gender-specific LightGBM + CatBoost ensemble · NCAA tournament data 2003–2025<br>
  Features: Elo, seed, SOS, Four Factors, Massey Ordinals
</p>
<p class="footer-text">Xinwei Huang · Haoran Zhang</p>
""", unsafe_allow_html=True)
