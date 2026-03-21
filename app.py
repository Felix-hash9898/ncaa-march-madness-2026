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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

html, body, [class*="css"], .stApp,
.stApp > div, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f5f5f5 !important;
    color: #0d0d0d !important;
}
.stApp { background-color: #f5f5f5 !important; }
[data-testid="stHeader"], [data-testid="stToolbar"],
.stDeployButton, header { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }

.block-container {
    max-width: 700px;
    padding-top: 3rem !important;
    padding-bottom: 4rem;
}

.page-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0d0d0d;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
    line-height: 1.1;
}
.page-sub {
    font-size: 0.68rem;
    color: #888888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Gender toggle */
.stButton > button {
    border-radius: 4px !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.3rem 1.1rem !important;
    border: 1px solid #dddddd !important;
    transition: all 0.15s !important;
}
.stButton > button[kind="primary"] {
    background: #0d0d0d !important;
    color: #0d0d0d !important;
    border-color: #0d0d0d !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #999999 !important;
}

/* Labels */
label {
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #999999 !important;
}

/* Selectboxes */
.stSelectbox > div > div {
    background-color: #ffffff !important;
    border: 1px solid #dddddd !important;
    border-radius: 4px !important;
    color: #0d0d0d !important;
    font-size: 0.88rem !important;
}

.divider {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 1.5rem 0;
}

/* Probability display */
.pct-left  { text-align: right; padding-right: 0.5rem; }
.pct-right { text-align: left;  padding-left:  0.5rem; }

.pct-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.01em;
}
.pct-name {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    color: #aaaaaa;
}

.info-text {
    font-size: 0.65rem;
    color: #999999;
    line-height: 1.8;
    text-align: center;
    margin-top: 1.5rem;
    letter-spacing: 0.02em;
}
.footer-text {
    text-align: center;
    font-size: 0.62rem;
    color: #bbbbbb;
    margin-top: 1.5rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    sub    = pd.read_csv("submissions/stage2/submission_ensemble_v1.csv")
    mteams = pd.read_csv("data/MTeams.csv")[["TeamID", "TeamName"]]
    wteams = pd.read_csv("data/WTeams.csv")[["TeamID", "TeamName"]]
    parts         = sub["ID"].str.split("_", expand=True)
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


def ring_polygon(t_start_deg, t_end_deg, r_in, r_out, n=400):
    thetas = np.linspace(np.radians(t_start_deg), np.radians(t_end_deg), n)
    xo = r_out * np.cos(thetas)
    yo = r_out * np.sin(thetas)
    xi = r_in  * np.cos(thetas[::-1])
    yi = r_in  * np.sin(thetas[::-1])
    return (np.concatenate([xo, xi, [xo[0]]]),
            np.concatenate([yo, yi, [yo[0]]]))


def make_espn_donut(prob_a, prob_b, color_a="#2563eb", color_b="#e11d48"):
    r_out = 1.0
    r_in  = 0.62
    seam  = 2.5
    deg_a = prob_a * 360
    bottom_deg = 90 + deg_a

    fig = go.Figure()

    # Full ring in color_b
    xb, yb = ring_polygon(90 - (prob_b * 360) - 1, 90 + 1, r_in, r_out, 600)
    fig.add_trace(go.Scatter(
        x=xb, y=yb, fill="toself", fillcolor=color_b,
        line=dict(color=color_b, width=0),
        hoverinfo="skip", showlegend=False,
    ))

    # Team A arc on top
    xa, ya = ring_polygon(90 + seam, 90 + deg_a - seam, r_in, r_out, 400)
    fig.add_trace(go.Scatter(
        x=xa, y=ya, fill="toself", fillcolor=color_a,
        line=dict(color=color_a, width=0),
        hoverinfo="skip", showlegend=False,
    ))

    # Seam lines at 12 o'clock and bottom
    for angle in [90, bottom_deg]:
        a = np.radians(angle)
        fig.add_trace(go.Scatter(
            x=[r_in * np.cos(a), r_out * np.cos(a)],
            y=[r_in * np.sin(a), r_out * np.sin(a)],
            mode="lines",
            line=dict(color="#f5f5f5", width=4),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        width=260, height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-1.15, 1.15], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-1.15, 1.15]),
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">NCAA March Madness 2026</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Win Probability · LGB + CatBoost Ensemble · CV Brier 0.1579</p>', unsafe_allow_html=True)

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
COLOR_A = "#2563eb"
COLOR_B = "#e11d48"

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_l, col_c, col_r = st.columns([2, 2.4, 2])

with col_l:
    st.markdown(f"""
    <div class="pct-left">
      <div class="pct-num" style="color:{COLOR_A}">{prob_a*100:.0f}%</div>
      <div class="pct-name">{a_name}</div>
    </div>""", unsafe_allow_html=True)

with col_c:
    fig = make_espn_donut(prob_a, prob_b, COLOR_A, COLOR_B)
    st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})

with col_r:
    st.markdown(f"""
    <div class="pct-right">
      <div class="pct-num" style="color:{COLOR_B}">{prob_b*100:.0f}%</div>
      <div class="pct-name">{b_name}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<p class="info-text">
  Gender-specific LightGBM + CatBoost ensemble · NCAA tournament data 2003–2025<br>
  Features: Elo · Seed · Strength of Schedule · Four Factors · Massey Ordinals
</p>
<p class="footer-text">XINWEI HUANG · HAORAN ZHANG</p>
""", unsafe_allow_html=True)
