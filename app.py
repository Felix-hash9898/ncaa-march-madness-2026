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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #fafafa !important;
    color: #111111 !important;
}

.stApp { background-color: #fafafa !important; }

.block-container {
    max-width: 680px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    background-color: #fafafa !important;
}

/* Header */
.page-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111111;
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem;
}
.page-sub {
    font-size: 0.72rem;
    color: #aaaaaa;
    letter-spacing: 0.04em;
    margin-bottom: 2rem;
}

/* Gender toggle */
.gender-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}
.g-btn {
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    border: 1.5px solid #e0e0e0;
    background: #ffffff;
    color: #555555;
    transition: all 0.15s;
    text-decoration: none;
}
.g-btn.active {
    background: #111111;
    color: #ffffff;
    border-color: #111111;
}

/* Selects */
label {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: #aaaaaa !important;
}
.stSelectbox > div > div {
    background-color: #ffffff !important;
    border: 1.5px solid #e8e8e8 !important;
    border-radius: 8px !important;
    color: #111111 !important;
    font-size: 0.88rem !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #111111 !important;
    box-shadow: none !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #ebebeb;
    margin: 1.5rem 0;
}

/* Result section */
.result-section {
    background: #ffffff;
    border: 1.5px solid #ebebeb;
    border-radius: 14px;
    padding: 2rem 2rem 1.5rem 2rem;
    margin-top: 1.5rem;
}

/* Circles row */
.circles-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 1.75rem;
}

.circle-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
}

.circle-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #111111;
    text-align: center;
    max-width: 120px;
    line-height: 1.3;
}

.circle-name.dim {
    color: #cccccc;
    font-weight: 400;
}

.vs-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #cccccc;
    letter-spacing: 0.1em;
    margin-top: -0.5rem;
}

/* Tag */
.favored-tag {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #ffffff;
    background: #111111;
    border-radius: 20px;
    padding: 0.15rem 0.5rem;
    margin-left: 0.4rem;
    vertical-align: middle;
}

/* Bottom rows */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-top: 1px solid #f2f2f2;
}
.stat-label { font-size: 0.78rem; color: #aaaaaa; }
.stat-val   { font-size: 0.82rem; font-weight: 600; color: #111111; }

.info-text {
    font-size: 0.72rem;
    color: #bbbbbb;
    line-height: 1.7;
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid #f2f2f2;
}

.footer-text {
    text-align: center;
    font-size: 0.68rem;
    color: #cccccc;
    margin-top: 2.5rem;
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


def make_ring_svg(prob_a, prob_b, name_a, name_b, color_a="#2563eb", color_b="#e11d48"):
    """SVG ring chart: two arcs showing win probabilities."""
    r = 52
    cx, cy = 80, 80
    stroke = 10
    circumference = 2 * np.pi * r

    # Arc lengths
    arc_a = circumference * prob_a
    arc_b = circumference * prob_b
    gap = 3  # gap between arcs in px

    # Team A arc: starts at top (-90deg), goes clockwise
    dash_a = f"{max(arc_a - gap, 0):.1f} {circumference:.1f}"
    # Team B arc: starts where A ends
    offset_b = -(arc_a)
    dash_b = f"{max(arc_b - gap, 0):.1f} {circumference:.1f}"
    rotate_b = 360 * prob_a

    pct_a = f"{prob_a*100:.0f}%"
    pct_b = f"{prob_b*100:.0f}%"

    svg = f"""
    <svg width="160" height="160" viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg">
      <style>
        .arc-a {{
          stroke-dasharray: {dash_a};
          stroke-dashoffset: 0;
          transform-origin: 80px 80px;
          transform: rotate(-90deg);
          animation: growA 0.9s cubic-bezier(0.4,0,0.2,1) forwards;
        }}
        .arc-b {{
          stroke-dasharray: {dash_b};
          stroke-dashoffset: 0;
          transform-origin: 80px 80px;
          transform: rotate({rotate_b - 90:.1f}deg);
          animation: growB 0.9s cubic-bezier(0.4,0,0.2,1) forwards;
        }}
        @keyframes growA {{
          from {{ stroke-dasharray: 0 {circumference:.1f}; }}
          to   {{ stroke-dasharray: {dash_a}; }}
        }}
        @keyframes growB {{
          from {{ stroke-dasharray: 0 {circumference:.1f}; }}
          to   {{ stroke-dasharray: {dash_b}; }}
        }}
      </style>

      <!-- Background track -->
      <circle cx="{cx}" cy="{cy}" r="{r}"
              fill="none" stroke="#f0f0f0" stroke-width="{stroke}"/>

      <!-- Team A arc -->
      <circle cx="{cx}" cy="{cy}" r="{r}"
              fill="none" stroke="{color_a}" stroke-width="{stroke}"
              stroke-linecap="round"
              class="arc-a"/>

      <!-- Team B arc -->
      <circle cx="{cx}" cy="{cy}" r="{r}"
              fill="none" stroke="{color_b}" stroke-width="{stroke}"
              stroke-linecap="round"
              class="arc-b"/>

      <!-- Center text -->
      <text x="{cx}" y="{cy - 6}" text-anchor="middle"
            font-family="Inter, sans-serif" font-size="11" fill="#aaaaaa" font-weight="500"
            letter-spacing="1">VS</text>

      <!-- Color legend dots -->
      <circle cx="{cx - 22}" cy="{cy + 18}" r="4" fill="{color_a}"/>
      <text x="{cx - 14}" y="{cy + 22}" font-family="Inter, sans-serif"
            font-size="9" fill="#aaaaaa">{pct_a}</text>

      <circle cx="{cx + 8}" cy="{cy + 18}" r="4" fill="{color_b}"/>
      <text x="{cx + 16}" y="{cy + 22}" font-family="Inter, sans-serif"
            font-size="9" fill="#aaaaaa">{pct_b}</text>
    </svg>
    """
    return svg


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">NCAA March Madness 2026</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">WIN PROBABILITY · LGB + CATBOOST ENSEMBLE · CV BRIER 0.1579</p>', unsafe_allow_html=True)

try:
    lookup, m_map, w_map = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

# ── Gender toggle (session state, no reset) ───────────────────────────────────
if "gender" not in st.session_state:
    st.session_state.gender = "Men's"

col_m, col_w, _ = st.columns([1.2, 1.2, 5])
with col_m:
    if st.button("Men's", use_container_width=True,
                 type="primary" if st.session_state.gender == "Men's" else "secondary"):
        st.session_state.gender = "Men's"
with col_w:
    if st.button("Women's", use_container_width=True,
                 type="primary" if st.session_state.gender == "Women's" else "secondary"):
        st.session_state.gender = "Women's"

team_map = m_map if st.session_state.gender == "Men's" else w_map
sorted_teams = sorted(team_map.items(), key=lambda x: x[1])
name_to_id   = {name: tid for tid, name in sorted_teams}
team_names   = [name for _, name in sorted_teams]

# ── Team selectors ────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Preserve selections across gender switches using session state
if "team_a_idx" not in st.session_state:
    st.session_state.team_a_idx = 0
if "team_b_idx" not in st.session_state:
    st.session_state.team_b_idx = min(1, len(team_names) - 1)

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
    st.error("Matchup not found in 2026 submission.")
    st.stop()

prob_b = 1 - prob_a
a_wins = prob_a >= prob_b

COLOR_A = "#2563eb"  # blue
COLOR_B = "#e11d48"  # red

# ── Result card ───────────────────────────────────────────────────────────────
a_tag = '<span class="favored-tag">Favored</span>' if a_wins else ""
b_tag = '<span class="favored-tag">Favored</span>' if not a_wins else ""
a_dim = "" if a_wins else " dim"
b_dim = "" if not a_wins else " dim"

svg = make_ring_svg(prob_a, prob_b, a_name, b_name, COLOR_A, COLOR_B)

st.markdown(f"""
<div class="result-section">

  <!-- Ring chart -->
  <div style="display:flex;justify-content:center;margin-bottom:1.5rem;">
    {svg}
  </div>

  <!-- Team rows -->
  <div class="stat-row">
    <span class="stat-label">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                   background:{COLOR_A};margin-right:6px;vertical-align:middle;"></span>
      <span class="circle-name{a_dim}">{a_name}{a_tag}</span>
    </span>
    <span class="stat-val" style="color:{'#111111' if a_wins else '#cccccc'}">{prob_a*100:.1f}%</span>
  </div>

  <div class="stat-row">
    <span class="stat-label">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                   background:{COLOR_B};margin-right:6px;vertical-align:middle;"></span>
      <span class="circle-name{b_dim}">{b_name}{b_tag}</span>
    </span>
    <span class="stat-val" style="color:{'#111111' if not a_wins else '#cccccc'}">{prob_b*100:.1f}%</span>
  </div>

  <p class="info-text">
    Gender-specific LightGBM + CatBoost ensemble · NCAA tournament data 2003–2025 ·
    Features: Elo, seed, SOS, Four Factors, Massey Ordinals
  </p>

</div>

<p class="footer-text">Xinwei Huang · Haoran Zhang</p>
""", unsafe_allow_html=True)
