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
    font-size: 0.72rem; font-weight: 700; color: #111111;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.1rem;
}
.page-sub {
    font-size: 0.68rem; color: #bbbbbb;
    letter-spacing: 0.04em; margin-bottom: 2rem;
}
.stButton > button {
    border-radius: 20px !important; font-size: 0.76rem !important;
    font-weight: 500 !important; padding: 0.28rem 1rem !important;
    border: 1.5px solid #e0e0e0 !important;
}
.stButton > button[kind="primary"] {
    background: #111111 !important; color: #ffffff !important;
    border-color: #111111 !important;
}
.stButton > button[kind="secondary"] {
    background: #ffffff !important; color: #666666 !important;
}
label {
    font-size: 0.65rem !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: #aaaaaa !important;
}
.stSelectbox > div > div {
    background-color: #f8f8f8 !important; border: 1.5px solid #e8e8e8 !important;
    border-radius: 8px !important; color: #111111 !important; font-size: 0.88rem !important;
}
.divider { border: none; border-top: 1px solid #f0f0f0; margin: 1.25rem 0; }
.pct-left  { text-align: right; padding-right: 0.75rem; }
.pct-right { text-align: left;  padding-left:  0.75rem; }
.pct-num { font-size: 2.6rem; font-weight: 700; line-height: 1; }
.pct-name { font-size: 0.67rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; margin-top: 0.35rem; }
.info-text { font-size: 0.68rem; color: #bbbbbb; line-height: 1.7; text-align: center; margin-top: 1.25rem; }
.footer-text { text-align: center; font-size: 0.65rem; color: #cccccc; margin-top: 1.5rem; }
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


def arc_xy(theta_start_deg, theta_end_deg, r, n=300):
    thetas = np.linspace(np.radians(theta_start_deg), np.radians(theta_end_deg), n)
    return r * np.cos(thetas), r * np.sin(thetas)


def make_espn_donut(prob_a, prob_b, color_a="#2563eb", color_b="#e11d48"):
    r_outer = 1.0
    r_inner = 0.62
    n = 300

    deg_a = prob_a * 360
    deg_b = prob_b * 360

    fig = go.Figure()

    # Background ring
    t = np.linspace(0, 2*np.pi, 600)
    xo, yo = r_outer * np.cos(t), r_outer * np.sin(t)
    xi, yi = r_inner * np.cos(t[::-1]), r_inner * np.sin(t[::-1])
    fig.add_trace(go.Scatter(
        x=np.concatenate([xo, xi, [xo[0]]]),
        y=np.concatenate([yo, yi, [yo[0]]]),
        fill="toself", fillcolor="#f2f2f2",
        line=dict(color="#f2f2f2", width=0),
        hoverinfo="skip", showlegend=False,
    ))

    def filled_arc(t_start, t_end, r_in, r_out, color):
        thetas = np.linspace(np.radians(t_start), np.radians(t_end), n)
        xo2 = r_out * np.cos(thetas)
        yo2 = r_out * np.sin(thetas)
        xi2 = r_in  * np.cos(thetas[::-1])
        yi2 = r_in  * np.sin(thetas[::-1])
        return (
            np.concatenate([xo2, xi2, [xo2[0]]]),
            np.concatenate([yo2, yi2, [yo2[0]]]),
        )

    # Team B: clockwise from 12 o'clock → theta goes 90 → 90-deg_b
    xb, yb = filled_arc(90, 90 - deg_b, r_inner, r_outer, color_b)
    fig.add_trace(go.Scatter(
        x=xb, y=yb, fill="toself", fillcolor=color_b,
        line=dict(color=color_b, width=0),
        hoverinfo="skip", showlegend=False,
    ))

    # Team A: counterclockwise from 12 o'clock → theta goes 90 → 90+deg_a
    xa, ya = filled_arc(90, 90 + deg_a, r_inner, r_outer, color_a)
    fig.add_trace(go.Scatter(
        x=xa, y=ya, fill="toself", fillcolor=color_a,
        line=dict(color=color_a, width=0),
        hoverinfo="skip", showlegend=False,
    ))

    # White gap lines at 12 o'clock and 6 o'clock (seam lines)
    for angle_deg in [90, -90]:
        ang = np.radians(angle_deg)
        fig.add_trace(go.Scatter(
            x=[r_inner * np.cos(ang), r_outer * np.cos(ang)],
            y=[r_inner * np.sin(ang), r_outer * np.sin(ang)],
            mode="lines", line=dict(color="#ffffff", width=3),
            hoverinfo="skip", showlegend=False,
        ))

    # Center divider: subtle line ONLY inside the ring (top & bottom arc sections)
    # Top: from inner top to outer top
    for y0, y1 in [(r_inner, r_outer), (-r_outer, -r_inner)]:
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[y0, y1],
            mode="lines", line=dict(color="#ffffff", width=2),
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
COLOR_A = "#2563eb"
COLOR_B = "#e11d48"

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_l, col_c, col_r = st.columns([2, 2.4, 2])

with col_l:
    st.markdown(f"""
    <div class="pct-left">
      <div class="pct-num" style="color:{COLOR_A}">{prob_a*100:.0f}%</div>
      <div class="pct-name" style="color:#555555">{a_name}</div>
    </div>""", unsafe_allow_html=True)

with col_c:
    fig = make_espn_donut(prob_a, prob_b, COLOR_A, COLOR_B)
    st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})

with col_r:
    st.markdown(f"""
    <div class="pct-right">
      <div class="pct-num" style="color:{COLOR_B}">{prob_b*100:.0f}%</div>
      <div class="pct-name" style="color:#555555">{b_name}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<p class="info-text">
  Gender-specific LightGBM + CatBoost ensemble · NCAA tournament data 2003–2025<br>
  Features: Elo, seed, SOS, Four Factors, Massey Ordinals
</p>
<p class="footer-text">Xinwei Huang · Haoran Zhang</p>
""", unsafe_allow_html=True)
