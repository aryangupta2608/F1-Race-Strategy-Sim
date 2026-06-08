"""
app.py — F1 Race Strategy Simulator
====================================
Streamlit web app that ties together all three modules:
  - data_fetcher.py  → pulls FastF1 race data
  - tyre_model.py    → fits tyre degradation curves
  - strategy_engine.py → finds optimal pit stop strategy

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

import fastf1
import os

# Set up FastF1 cache — uses a temp dir on Streamlit Cloud (no persistent disk)
# This means first load per session downloads fresh data (~30s), which is expected
_CACHE_DIR = "/tmp/f1_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(_CACHE_DIR)

from data_fetcher import get_race_schedule, load_race_session, get_driver_list, get_pit_stops
from tyre_model import build_tyre_model, predict_lap_time
from strategy_engine import find_optimal_strategy, simulate_strategy, compare_actual_vs_optimal, get_pit_loss_time


# ── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Strategy Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS — dark F1-themed styling ──────────────────────────────────────

st.markdown("""
<style>
    /* Import F1-style fonts */
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

    /* Global background */
    .stApp {
        background-color: #0a0a0a;
        color: #f0f0f0;
        font-family: 'Barlow', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #e10600;
    }

    /* Header */
    .f1-header {
        background: linear-gradient(135deg, #e10600 0%, #9b0400 60%, #0a0a0a 100%);
        padding: 2rem 2.5rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #ffffff;
    }
    .f1-header h1 {
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #ffffff;
        margin: 0;
        line-height: 1;
    }
    .f1-header p {
        font-family: 'Barlow', sans-serif;
        font-weight: 300;
        color: rgba(255,255,255,0.75);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-top: 3px solid #e10600;
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #aaa;
        margin-top: 0.3rem;
    }

    /* Strategy pills */
    .compound-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        margin: 2px;
    }
    .pill-SOFT   { background: #cc0000; color: white; }
    .pill-MEDIUM { background: #cc9900; color: black; }
    .pill-HARD   { background: #888888; color: black; }

    /* Section headers */
    .section-header {
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #e10600;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Delta display */
    .delta-positive { color: #ff4444; font-weight: 600; }
    .delta-neutral  { color: #44ff88; font-weight: 600; }

    /* Streamlit overrides */
    .stSelectbox label, .stSlider label {
        font-family: 'Barlow Condensed', sans-serif !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        color: #aaa !important;
    }
    div[data-testid="stMetric"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-top: 3px solid #e10600;
        padding: 1rem;
        border-radius: 4px;
    }
    .stButton > button {
        background: #e10600 !important;
        color: white !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 3px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: #ff1a0e !important;
    }
</style>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="f1-header">
    <h1>🏎️ F1 Strategy Simulator</h1>
    <p>Pit stop optimizer powered by real FastF1 timing data</p>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR — Race Selection ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Race Configuration")
    st.markdown("---")

    # Year selector
    current_year = 2026  # update this each season
    year = st.selectbox(
        "Season",
        options=list(range(current_year, 2018, -1)),
        index=0
    )

    # Load schedule for chosen year (cached so it's fast on re-runs)
    @st.cache_data
    def fetch_schedule(yr):
        return get_race_schedule(yr)

    with st.spinner("Loading calendar..."):
        schedule = fetch_schedule(year)

    # Race selector — show race names, map back to round number
    race_names   = schedule['EventName'].tolist()
    selected_race = st.selectbox("Grand Prix", options=race_names)
    round_number  = int(schedule[schedule['EventName'] == selected_race]['RoundNumber'].values[0])

    st.markdown("---")
    st.markdown("### 🏎️ Driver Comparison")

    # Load session (cached by year + round so switching races reloads correctly)
    # We define this outside cache_resource so Streamlit Cloud handles it cleanly
    @st.cache_resource(show_spinner=False)
    def fetch_session(yr, rnd):
        sess = fastf1.get_session(yr, rnd, "R")
        sess.load(telemetry=False, weather=False, messages=False)
        return sess

    with st.spinner(f"Loading {selected_race} — first load may take ~30s..."):
        try:
            session = fetch_session(year, round_number)
            # Verify laps loaded correctly before proceeding
            _ = session.laps
        except Exception as e:
            st.error(f"Failed to load session data: {e}")
            st.stop()

    drivers = get_driver_list(session)
    driver1 = st.selectbox("Driver 1", options=drivers, index=drivers.index("VER") if "VER" in drivers else 0)
    driver2 = st.selectbox("Driver 2", options=drivers, index=drivers.index("HAM") if "HAM" in drivers else 1)

    st.markdown("---")
    st.markdown("### 🔧 Simulation Settings")
    # Auto-fetch total laps from the session — no manual input needed
    try:
        total_laps = int(session.laps["LapNumber"].max())
    except Exception:
        total_laps = 57  # safe fallback
    st.markdown(f"**Race distance:** {total_laps} laps")
    _pit_preview = get_pit_loss_time(selected_race)
    st.markdown(f"**Pit lane delta:** {_pit_preview}s")
    max_stops  = st.selectbox("Max Pit Stops to Simulate", options=[1, 2], index=1)

    st.markdown("---")
    run_btn = st.button("▶  RUN SIMULATION")


# ── MAIN CONTENT ─────────────────────────────────────────────────────────────

if not run_btn:
    # Landing state — show instructions
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #555;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">🏁</div>
        <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.5rem;
                    letter-spacing: 0.15em; text-transform: uppercase; color: #888;">
            Select a race and click Run Simulation
        </div>
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #555; max-width: 500px; margin: 1rem auto 0;">
            The simulator loads real lap timing data, fits a tyre degradation model
            per compound, then brute-forces every possible pit stop strategy to find
            the fastest one.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── RUN THE SIMULATION ────────────────────────────────────────────────────

    with st.spinner("Building tyre degradation model..."):
        model = build_tyre_model(session, total_laps)

    with st.spinner("Simulating all strategies..."):
        event_name = session.event["EventName"]
        pit_loss   = get_pit_loss_time(event_name)
        optimal = find_optimal_strategy(model, total_laps, max_stops, event_name)

    if not optimal:
        st.error("Not enough tyre data to simulate strategies for this race.")
        st.stop()

    best      = optimal['best_overall']
    best_1    = optimal['best_by_stops'].get(1)
    best_2    = optimal['best_by_stops'].get(2)
    n_sim     = optimal['total_simulated']

    # ── SECTION 1: Top-line summary metrics ──────────────────────────────────

    st.markdown('<div class="section-header">Optimal Strategy</div>', unsafe_allow_html=True)

    # Figure out which stop count actually won overall
    winning_stops = best['n_stops']

    col1, col2, col3, col4 = st.columns(4)

    best_mins = int(best['total_time'] // 60)
    best_secs = best['total_time'] % 60

    with col1:
        st.metric("Best Race Time", f"{best_mins}m {best_secs:.1f}s")
    with col2:
        st.metric("Optimal Stop Count", f"{best['n_stops']}-stop")
    with col3:
        st.metric("Pit Loss Time", f"{best['pit_time']:.0f}s")
    with col4:
        st.metric("Strategies Tested", f"{n_sim:,}")

    # Show the best strategy as compound pills
    st.markdown("<br>", unsafe_allow_html=True)
    pill_colours = {'SOFT': 'pill-SOFT', 'MEDIUM': 'pill-MEDIUM', 'HARD': 'pill-HARD'}

    stint_html = " &nbsp;→&nbsp; ".join([
        f'<span class="compound-pill {pill_colours.get(c, "")}">{c} — {l} laps</span>'
        for c, l in best['strategy']
    ])
    st.markdown(
        f'<div style="margin: 0.5rem 0 1.5rem;">{stint_html}</div>',
        unsafe_allow_html=True
    )

    # ── SECTION 2: 1-stop vs 2-stop comparison table ─────────────────────────

    st.markdown('<div class="section-header">Strategy Comparison</div>', unsafe_allow_html=True)

    compare_rows = []
    for n, strat in optimal['best_by_stops'].items():
        mins = int(strat['total_time'] // 60)
        secs = strat['total_time'] % 60
        strategy_str = " → ".join([f"{c} ({l} laps)" for c, l in strat['strategy']])
        gap = strat['total_time'] - best['total_time']
        compare_rows.append({
            'Stops':        f"{n}-stop",
            'Strategy':     strategy_str,
            'Pit Laps':     str(strat['pit_laps']),
            'Race Time':    f"{mins}m {secs:.2f}s",
            'Gap to Best':  f"+{gap:.2f}s" if gap > 0 else "OPTIMAL"
        })

    st.dataframe(
        pd.DataFrame(compare_rows),
        use_container_width=True,
        hide_index=True
    )

    # ── SECTION 3: Driver comparison ─────────────────────────────────────────

    st.markdown('<div class="section-header">Driver Analysis</div>', unsafe_allow_html=True)

    dcol1, dcol2 = st.columns(2)

    def render_driver_card(col, driver_code, session, model, total_laps, optimal):
        stints_df = get_pit_stops(session, total_laps)
        driver_stints = stints_df[stints_df['Driver'] == driver_code]

        if driver_stints.empty:
            col.warning(f"No data for {driver_code}")
            return

        actual_strategy = [
            (row['Compound'], int(row['StintLength']))
            for _, row in driver_stints.iterrows()
        ]
        actual_result = simulate_strategy(model, total_laps, actual_strategy, get_pit_loss_time(session.event["EventName"]))
        best          = optimal['best_overall']

        if not actual_result['valid']:
            col.warning(f"Could not simulate {driver_code}'s strategy")
            return

        delta    = actual_result['total_time'] - best['total_time']
        a_mins   = int(actual_result['total_time'] // 60)
        a_secs   = actual_result['total_time'] % 60
        faster   = delta <= 0

        with col:
            st.markdown(f"#### 🏎️ {driver_code}")

            # Actual strategy pills
            pills = " → ".join([
                f'<span class="compound-pill {pill_colours.get(c,"")}">{c} {l}L</span>'
                for c, l in actual_strategy
            ])
            st.markdown(pills, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Race Time", f"{a_mins}m {a_secs:.1f}s")
            colour = "🟢" if faster else "🔴"
            m2.metric(
                "vs Optimal",
                f"{colour} {abs(delta):.1f}s {'faster' if faster else 'slower'}"
            )

            # Pit lap annotation
            st.caption(f"Pitted on laps: {actual_result['pit_laps']}")

    render_driver_card(dcol1, driver1, session, model, total_laps, optimal)
    render_driver_card(dcol2, driver2, session, model, total_laps, optimal)

    # ── SECTION 4: Tyre Degradation Model ────────────────────────────────────

    st.markdown('<div class="section-header">Tyre Degradation Model</div>', unsafe_allow_html=True)

    tcol1, tcol2 = st.columns([2, 1])

    with tcol1:
        # Plot degradation curves
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#111111')

        colours    = {'SOFT': '#e10600', 'MEDIUM': '#e6b800', 'HARD': '#aaaaaa'}
        max_age    = min(40, total_laps)
        tyre_ages  = np.arange(1, max_age + 1)

        for compound, colour in colours.items():
            comp_data = model.get(compound)
            if comp_data is None or comp_data['deg_per_lap'] is None:
                continue
            lap_times = [predict_lap_time(model, compound, age) for age in tyre_ages]
            label = (f"{compound}  (+{comp_data['deg_per_lap']:.3f}s/lap, "
                     f"R²={comp_data['r_squared']:.2f})")
            ax.plot(tyre_ages, lap_times, color=colour, linewidth=2.5, label=label)

        ax.set_xlabel("Tyre Age (laps)", color='#888', fontsize=10)
        ax.set_ylabel("Predicted Lap Time (s)", color='#888', fontsize=10)
        ax.set_title("Degradation Curves by Compound", color='#ffffff', fontsize=12, pad=12)
        ax.tick_params(colors='#666')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax.grid(alpha=0.15, color='#444')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tcol2:
        # Deg model table
        model_rows = []
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            d = model.get(compound, {})
            if d and d.get('deg_per_lap') is not None:
                model_rows.append({
                    'Compound':   compound,
                    'Base Pace':  f"{d['base_pace']:.2f}s",
                    'Deg/lap':    f"+{d['deg_per_lap']:.4f}s",
                    'R²':         f"{d['r_squared']:.2f}",
                    'Laps Used':  d['sample_size']
                })
            else:
                model_rows.append({
                    'Compound': compound,
                    'Base Pace': '—',
                    'Deg/lap':   '—',
                    'R²':        '—',
                    'Laps Used': 0
                })
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
        st.caption("Base pace = 25th pct raw lap time. Deg rate from linear regression on normalised, fuel-corrected lap times.")

    # ── SECTION 5: Strategy lap time chart ───────────────────────────────────

    st.markdown('<div class="section-header">Lap-by-Lap Pace Comparison</div>', unsafe_allow_html=True)

    # Build results for both drivers and the optimal strategy
    to_plot   = []
    plot_labels = []

    for driver_code in [driver1, driver2]:
        stints_df     = get_pit_stops(session, total_laps)
        driver_stints = stints_df[stints_df['Driver'] == driver_code]
        if driver_stints.empty:
            continue
        actual_strategy = [(r['Compound'], int(r['StintLength'])) for _, r in driver_stints.iterrows()]
        result = simulate_strategy(model, total_laps, actual_strategy, get_pit_loss_time(session.event["EventName"]))
        if result['valid']:
            to_plot.append((result, driver_code, '#00aaff'))

    if best_1:
        to_plot.append((best_1, 'Best 1-stop', '#e10600'))
    if best_2:
        to_plot.append((best_2, 'Best 2-stop', '#ff8800'))

    if to_plot:
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        fig2.patch.set_facecolor('#0a0a0a')
        ax2.set_facecolor('#111111')

        driver_colours = {'VER': '#3671C6', 'HAM': '#00D2BE', 'LEC': '#E8002D',
                          'SAI': '#E8002D', 'NOR': '#FF8000', 'PIA': '#FF8000',
                          'RUS': '#00D2BE', 'ALO': '#358C75', 'STR': '#358C75'}
        line_styles = ['-', '--', '-.', ':']

        for i, (result, label, fallback_colour) in enumerate(to_plot):
            lap_times = result['lap_times']
            laps      = list(range(1, len(lap_times) + 1))
            colour    = driver_colours.get(label, fallback_colour)
            ax2.plot(laps, lap_times,
                     color=colour,
                     linewidth=2,
                     linestyle=line_styles[i % len(line_styles)],
                     label=label,
                     alpha=0.9)
            for pit_lap in result['pit_laps']:
                ax2.axvline(x=pit_lap, color=colour, linestyle=':', alpha=0.3, linewidth=1)

        ax2.set_xlabel("Lap Number", color='#888', fontsize=10)
        ax2.set_ylabel("Predicted Lap Time (s)", color='#888', fontsize=10)
        ax2.set_title("Predicted Lap Times — Actual vs Optimal Strategies", color='#fff', fontsize=12, pad=12)
        ax2.tick_params(colors='#666')
        ax2.spines['bottom'].set_color('#333')
        ax2.spines['left'].set_color('#333')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax2.grid(alpha=0.15, color='#444')
        st.caption("Vertical dotted lines = pit stops for each strategy")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── FOOTER ───────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#444; font-size:0.8rem; '
        'font-family: Barlow Condensed, sans-serif; letter-spacing: 0.1em;">'
        'DATA: FASTF1 OFFICIAL F1 TIMING FEED &nbsp;|&nbsp; '
        'MODEL: LINEAR REGRESSION ON FUEL-CORRECTED, DRIVER-NORMALISED LAP TIMES'
        '</div>',
        unsafe_allow_html=True
    )st.set_page_config(
    page_title="F1 Strategy Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS — dark F1-themed styling ──────────────────────────────────────

st.markdown("""
<style>
    /* Import F1-style fonts */
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

    /* Global background */
    .stApp {
        background-color: #0a0a0a;
        color: #f0f0f0;
        font-family: 'Barlow', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #e10600;
    }

    /* Header */
    .f1-header {
        background: linear-gradient(135deg, #e10600 0%, #9b0400 60%, #0a0a0a 100%);
        padding: 2rem 2.5rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #ffffff;
    }
    .f1-header h1 {
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #ffffff;
        margin: 0;
        line-height: 1;
    }
    .f1-header p {
        font-family: 'Barlow', sans-serif;
        font-weight: 300;
        color: rgba(255,255,255,0.75);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-top: 3px solid #e10600;
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #aaa;
        margin-top: 0.3rem;
    }

    /* Strategy pills */
    .compound-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        margin: 2px;
    }
    .pill-SOFT   { background: #cc0000; color: white; }
    .pill-MEDIUM { background: #cc9900; color: black; }
    .pill-HARD   { background: #888888; color: black; }

    /* Section headers */
    .section-header {
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #e10600;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Delta display */
    .delta-positive { color: #ff4444; font-weight: 600; }
    .delta-neutral  { color: #44ff88; font-weight: 600; }

    /* Streamlit overrides */
    .stSelectbox label, .stSlider label {
        font-family: 'Barlow Condensed', sans-serif !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        color: #aaa !important;
    }
    div[data-testid="stMetric"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-top: 3px solid #e10600;
        padding: 1rem;
        border-radius: 4px;
    }
    .stButton > button {
        background: #e10600 !important;
        color: white !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 3px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: #ff1a0e !important;
    }
</style>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="f1-header">
    <h1>🏎️ F1 Strategy Simulator</h1>
    <p>Pit stop optimizer powered by real FastF1 timing data</p>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR — Race Selection ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Race Configuration")
    st.markdown("---")

    # Year selector
    current_year = 2026  # update this each season
    year = st.selectbox(
        "Season",
        options=list(range(current_year, 2018, -1)),
        index=0
    )

    # Load schedule for chosen year (cached so it's fast on re-runs)
    @st.cache_data
    def fetch_schedule(yr):
        return get_race_schedule(yr)

    with st.spinner("Loading calendar..."):
        schedule = fetch_schedule(year)

    # Race selector — show race names, map back to round number
    race_names   = schedule['EventName'].tolist()
    selected_race = st.selectbox("Grand Prix", options=race_names)
    round_number  = int(schedule[schedule['EventName'] == selected_race]['RoundNumber'].values[0])

    st.markdown("---")
    st.markdown("### 🏎️ Driver Comparison")

    # Load session (cached by year + round so switching races reloads correctly)
    # We define this outside cache_resource so Streamlit Cloud handles it cleanly
    @st.cache_resource(show_spinner=False)
    def fetch_session(yr, rnd):
        sess = fastf1.get_session(yr, rnd, "R")
        sess.load(telemetry=False, weather=False, messages=False)
        return sess

    with st.spinner(f"Loading {selected_race} — first load may take ~30s..."):
        try:
            session = fetch_session(year, round_number)
            # Verify laps loaded correctly before proceeding
            _ = session.laps
        except Exception as e:
            st.error(f"Failed to load session data: {e}")
            st.stop()

    drivers = get_driver_list(session)
    driver1 = st.selectbox("Driver 1", options=drivers, index=drivers.index("VER") if "VER" in drivers else 0)
    driver2 = st.selectbox("Driver 2", options=drivers, index=drivers.index("HAM") if "HAM" in drivers else 1)

    st.markdown("---")
    st.markdown("### 🔧 Simulation Settings")
    # Auto-fetch total laps from the session — no manual input needed
    try:
        total_laps = int(session.laps["LapNumber"].max())
    except Exception:
        total_laps = 57  # safe fallback
    st.markdown(f"**Race distance:** {total_laps} laps")
    _pit_preview = get_pit_loss_time(selected_race)
    st.markdown(f"**Pit lane delta:** {_pit_preview}s")
    max_stops  = st.selectbox("Max Pit Stops to Simulate", options=[1, 2], index=1)

    st.markdown("---")
    run_btn = st.button("▶  RUN SIMULATION")


# ── MAIN CONTENT ─────────────────────────────────────────────────────────────

if not run_btn:
    # Landing state — show instructions
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #555;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">🏁</div>
        <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.5rem;
                    letter-spacing: 0.15em; text-transform: uppercase; color: #888;">
            Select a race and click Run Simulation
        </div>
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #555; max-width: 500px; margin: 1rem auto 0;">
            The simulator loads real lap timing data, fits a tyre degradation model
            per compound, then brute-forces every possible pit stop strategy to find
            the fastest one.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── RUN THE SIMULATION ────────────────────────────────────────────────────

    with st.spinner("Building tyre degradation model..."):
        model = build_tyre_model(session, total_laps)

    with st.spinner("Simulating all strategies..."):
        event_name = session.event["EventName"]
        pit_loss   = get_pit_loss_time(event_name)
        optimal = find_optimal_strategy(model, total_laps, max_stops, event_name)

    if not optimal:
        st.error("Not enough tyre data to simulate strategies for this race.")
        st.stop()

    best      = optimal['best_overall']
    best_1    = optimal['best_by_stops'].get(1)
    best_2    = optimal['best_by_stops'].get(2)
    n_sim     = optimal['total_simulated']

    # ── SECTION 1: Top-line summary metrics ──────────────────────────────────

    st.markdown('<div class="section-header">Optimal Strategy</div>', unsafe_allow_html=True)

    # Figure out which stop count actually won overall
    winning_stops = best['n_stops']

    col1, col2, col3, col4 = st.columns(4)

    best_mins = int(best['total_time'] // 60)
    best_secs = best['total_time'] % 60

    with col1:
        st.metric("Best Race Time", f"{best_mins}m {best_secs:.1f}s")
    with col2:
        st.metric("Optimal Stop Count", f"{best['n_stops']}-stop")
    with col3:
        st.metric("Pit Loss Time", f"{best['pit_time']:.0f}s")
    with col4:
        st.metric("Strategies Tested", f"{n_sim:,}")

    # Show the best strategy as compound pills
    st.markdown("<br>", unsafe_allow_html=True)
    pill_colours = {'SOFT': 'pill-SOFT', 'MEDIUM': 'pill-MEDIUM', 'HARD': 'pill-HARD'}

    stint_html = " &nbsp;→&nbsp; ".join([
        f'<span class="compound-pill {pill_colours.get(c, "")}">{c} — {l} laps</span>'
        for c, l in best['strategy']
    ])
    st.markdown(
        f'<div style="margin: 0.5rem 0 1.5rem;">{stint_html}</div>',
        unsafe_allow_html=True
    )

    # ── SECTION 2: 1-stop vs 2-stop comparison table ─────────────────────────

    st.markdown('<div class="section-header">Strategy Comparison</div>', unsafe_allow_html=True)

    compare_rows = []
    for n, strat in optimal['best_by_stops'].items():
        mins = int(strat['total_time'] // 60)
        secs = strat['total_time'] % 60
        strategy_str = " → ".join([f"{c} ({l} laps)" for c, l in strat['strategy']])
        gap = strat['total_time'] - best['total_time']
        compare_rows.append({
            'Stops':        f"{n}-stop",
            'Strategy':     strategy_str,
            'Pit Laps':     str(strat['pit_laps']),
            'Race Time':    f"{mins}m {secs:.2f}s",
            'Gap to Best':  f"+{gap:.2f}s" if gap > 0 else "OPTIMAL"
        })

    st.dataframe(
        pd.DataFrame(compare_rows),
        use_container_width=True,
        hide_index=True
    )

    # ── SECTION 3: Driver comparison ─────────────────────────────────────────

    st.markdown('<div class="section-header">Driver Analysis</div>', unsafe_allow_html=True)

    dcol1, dcol2 = st.columns(2)

    def render_driver_card(col, driver_code, session, model, total_laps, optimal):
        stints_df = get_pit_stops(session, total_laps)
        driver_stints = stints_df[stints_df['Driver'] == driver_code]

        if driver_stints.empty:
            col.warning(f"No data for {driver_code}")
            return

        actual_strategy = [
            (row['Compound'], int(row['StintLength']))
            for _, row in driver_stints.iterrows()
        ]
        actual_result = simulate_strategy(model, total_laps, actual_strategy, get_pit_loss_time(session.event["EventName"]))
        best          = optimal['best_overall']

        if not actual_result['valid']:
            col.warning(f"Could not simulate {driver_code}'s strategy")
            return

        delta    = actual_result['total_time'] - best['total_time']
        a_mins   = int(actual_result['total_time'] // 60)
        a_secs   = actual_result['total_time'] % 60
        faster   = delta <= 0

        with col:
            st.markdown(f"#### 🏎️ {driver_code}")

            # Actual strategy pills
            pills = " → ".join([
                f'<span class="compound-pill {pill_colours.get(c,"")}">{c} {l}L</span>'
                for c, l in actual_strategy
            ])
            st.markdown(pills, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Race Time", f"{a_mins}m {a_secs:.1f}s")
            colour = "🟢" if faster else "🔴"
            m2.metric(
                "vs Optimal",
                f"{colour} {abs(delta):.1f}s {'faster' if faster else 'slower'}"
            )

            # Pit lap annotation
            st.caption(f"Pitted on laps: {actual_result['pit_laps']}")

    render_driver_card(dcol1, driver1, session, model, total_laps, optimal)
    render_driver_card(dcol2, driver2, session, model, total_laps, optimal)

    # ── SECTION 4: Tyre Degradation Model ────────────────────────────────────

    st.markdown('<div class="section-header">Tyre Degradation Model</div>', unsafe_allow_html=True)

    tcol1, tcol2 = st.columns([2, 1])

    with tcol1:
        # Plot degradation curves
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#111111')

        colours    = {'SOFT': '#e10600', 'MEDIUM': '#e6b800', 'HARD': '#aaaaaa'}
        max_age    = min(40, total_laps)
        tyre_ages  = np.arange(1, max_age + 1)

        for compound, colour in colours.items():
            comp_data = model.get(compound)
            if comp_data is None or comp_data['deg_per_lap'] is None:
                continue
            lap_times = [predict_lap_time(model, compound, age) for age in tyre_ages]
            label = (f"{compound}  (+{comp_data['deg_per_lap']:.3f}s/lap, "
                     f"R²={comp_data['r_squared']:.2f})")
            ax.plot(tyre_ages, lap_times, color=colour, linewidth=2.5, label=label)

        ax.set_xlabel("Tyre Age (laps)", color='#888', fontsize=10)
        ax.set_ylabel("Predicted Lap Time (s)", color='#888', fontsize=10)
        ax.set_title("Degradation Curves by Compound", color='#ffffff', fontsize=12, pad=12)
        ax.tick_params(colors='#666')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax.grid(alpha=0.15, color='#444')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tcol2:
        # Deg model table
        model_rows = []
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            d = model.get(compound, {})
            if d and d.get('deg_per_lap') is not None:
                model_rows.append({
                    'Compound':   compound,
                    'Base Pace':  f"{d['base_pace']:.2f}s",
                    'Deg/lap':    f"+{d['deg_per_lap']:.4f}s",
                    'R²':         f"{d['r_squared']:.2f}",
                    'Laps Used':  d['sample_size']
                })
            else:
                model_rows.append({
                    'Compound': compound,
                    'Base Pace': '—',
                    'Deg/lap':   '—',
                    'R²':        '—',
                    'Laps Used': 0
                })
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
        st.caption("Base pace = 25th pct raw lap time. Deg rate from linear regression on normalised, fuel-corrected lap times.")

    # ── SECTION 5: Strategy lap time chart ───────────────────────────────────

    st.markdown('<div class="section-header">Lap-by-Lap Pace Comparison</div>', unsafe_allow_html=True)

    # Build results for both drivers and the optimal strategy
    to_plot   = []
    plot_labels = []

    for driver_code in [driver1, driver2]:
        stints_df     = get_pit_stops(session, total_laps)
        driver_stints = stints_df[stints_df['Driver'] == driver_code]
        if driver_stints.empty:
            continue
        actual_strategy = [(r['Compound'], int(r['StintLength'])) for _, r in driver_stints.iterrows()]
        result = simulate_strategy(model, total_laps, actual_strategy, get_pit_loss_time(session.event["EventName"]))
        if result['valid']:
            to_plot.append((result, driver_code, '#00aaff'))

    if best_1:
        to_plot.append((best_1, 'Best 1-stop', '#e10600'))
    if best_2:
        to_plot.append((best_2, 'Best 2-stop', '#ff8800'))

    if to_plot:
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        fig2.patch.set_facecolor('#0a0a0a')
        ax2.set_facecolor('#111111')

        driver_colours = {'VER': '#3671C6', 'HAM': '#00D2BE', 'LEC': '#E8002D',
                          'SAI': '#E8002D', 'NOR': '#FF8000', 'PIA': '#FF8000',
                          'RUS': '#00D2BE', 'ALO': '#358C75', 'STR': '#358C75'}
        line_styles = ['-', '--', '-.', ':']

        for i, (result, label, fallback_colour) in enumerate(to_plot):
            lap_times = result['lap_times']
            laps      = list(range(1, len(lap_times) + 1))
            colour    = driver_colours.get(label, fallback_colour)
            ax2.plot(laps, lap_times,
                     color=colour,
                     linewidth=2,
                     linestyle=line_styles[i % len(line_styles)],
                     label=label,
                     alpha=0.9)
            for pit_lap in result['pit_laps']:
                ax2.axvline(x=pit_lap, color=colour, linestyle=':', alpha=0.3, linewidth=1)

        ax2.set_xlabel("Lap Number", color='#888', fontsize=10)
        ax2.set_ylabel("Predicted Lap Time (s)", color='#888', fontsize=10)
        ax2.set_title("Predicted Lap Times — Actual vs Optimal Strategies", color='#fff', fontsize=12, pad=12)
        ax2.tick_params(colors='#666')
        ax2.spines['bottom'].set_color('#333')
        ax2.spines['left'].set_color('#333')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax2.grid(alpha=0.15, color='#444')
        st.caption("Vertical dotted lines = pit stops for each strategy")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── FOOTER ───────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#444; font-size:0.8rem; '
        'font-family: Barlow Condensed, sans-serif; letter-spacing: 0.1em;">'
        'DATA: FASTF1 OFFICIAL F1 TIMING FEED &nbsp;|&nbsp; '
        'MODEL: LINEAR REGRESSION ON FUEL-CORRECTED, DRIVER-NORMALISED LAP TIMES'
        '</div>',
        unsafe_allow_html=True
    )st.markdown("""
<style>
    /* Import F1-style fonts */
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

    /* Global background */
    .stApp {
        background-color: #0a0a0a;
        color: #f0f0f0;
        font-family: 'Barlow', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #e10600;
    }

    /* Header */
    .f1-header {
        background: linear-gradient(135deg, #e10600 0%, #9b0400 60%, #0a0a0a 100%);
        padding: 2rem 2.5rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #ffffff;
    }
    .f1-header h1 {
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #ffffff;
        margin: 0;
        line-height: 1;
    }
    .f1-header p {
        font-family: 'Barlow', sans-serif;
        font-weight: 300;
        color: rgba(255,255,255,0.75);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-top: 3px solid #e10600;
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #aaa;
        margin-top: 0.3rem;
    }

    /* Strategy pills */
    .compound-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        margin: 2px;
    }
    .pill-SOFT   { background: #cc0000; color: white; }
    .pill-MEDIUM { background: #cc9900; color: black; }
    .pill-HARD   { background: #888888; color: black; }

    /* Section headers */
    .section-header {
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #e10600;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Delta display */
    .delta-positive { color: #ff4444; font-weight: 600; }
    .delta-neutral  { color: #44ff88; font-weight: 600; }

    /* Streamlit overrides */
    .stSelectbox label, .stSlider label {
        font-family: 'Barlow Condensed', sans-serif !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        color: #aaa !important;
    }
    div[data-testid="stMetric"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-top: 3px solid #e10600;
        padding: 1rem;
        border-radius: 4px;
    }
    .stButton > button {
        background: #e10600 !important;
        color: white !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 3px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: #ff1a0e !important;
    }
</style>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="f1-header">
    <h1>🏎️ F1 Strategy Simulator</h1>
    <p>Pit stop optimizer powered by real FastF1 timing data</p>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR — Race Selection ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Race Configuration")
    st.markdown("---")

    # Year selector
    current_year = 2026  # update this each season
    year = st.selectbox(
        "Season",
        options=list(range(current_year, 2018, -1)),
        index=0
    )

    # Load schedule for chosen year (cached so it's fast on re-runs)
    @st.cache_data
    def fetch_schedule(yr):
        return get_race_schedule(yr)

    with st.spinner("Loading calendar..."):
        schedule = fetch_schedule(year)

    # Race selector — show race names, map back to round number
    race_names   = schedule['EventName'].tolist()
    selected_race = st.selectbox("Grand Prix", options=race_names)
    round_number  = int(schedule[schedule['EventName'] == selected_race]['RoundNumber'].values[0])

    st.markdown("---")
    st.markdown("### 🏎️ Driver Comparison")

    # Load session (cached)
    @st.cache_resource
    def fetch_session(yr, rnd):
        return load_race_session(yr, rnd)

    with st.spinner(f"Loading {selected_race}..."):
        session = fetch_session(year, round_number)

    drivers = get_driver_list(session)
    driver1 = st.selectbox("Driver 1", options=drivers, index=drivers.index('VER') if 'VER' in drivers else 0)
    driver2 = st.selectbox("Driver 2", options=drivers, index=drivers.index('HAM') if 'HAM' in drivers else 1)

    st.markdown("---")
    st.markdown("### 🔧 Simulation Settings")
    # Auto-fetch total laps from the session data so it's always correct
    total_laps = int(session.laps['LapNumber'].max())
    st.markdown(f'**Race distance:** {total_laps} laps')
    max_stops  = st.selectbox("Max Pit Stops to Simulate", options=[1, 2], index=1)

    st.markdown("---")
    run_btn = st.button("▶  RUN SIMULATION")


# ── MAIN CONTENT ─────────────────────────────────────────────────────────────

if not run_btn:
    # Landing state — show instructions
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #555;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">🏁</div>
        <div style="font-family: 'Barlow Condensed', sans-serif; font-size: 1.5rem;
                    letter-spacing: 0.15em; text-transform: uppercase; color: #888;">
            Select a race and click Run Simulation
        </div>
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #555; max-width: 500px; margin: 1rem auto 0;">
            The simulator loads real lap timing data, fits a tyre degradation model
            per compound, then brute-forces every possible pit stop strategy to find
            the fastest one.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── RUN THE SIMULATION ────────────────────────────────────────────────────

    with st.spinner("Building tyre degradation model..."):
        model = build_tyre_model(session, total_laps)

    with st.spinner("Simulating all strategies..."):
        optimal = find_optimal_strategy(model, total_laps, max_stops)

    if not optimal:
        st.error("Not enough tyre data to simulate strategies for this race.")
        st.stop()

    best      = optimal['best_overall']
    best_1    = optimal['best_by_stops'].get(1)
    best_2    = optimal['best_by_stops'].get(2)
    n_sim     = optimal['total_simulated']

    # ── SECTION 1: Top-line summary metrics ──────────────────────────────────

    st.markdown('<div class="section-header">Optimal Strategy</div>', unsafe_allow_html=True)

    # Figure out which stop count actually won overall
    winning_stops = best['n_stops']

    col1, col2, col3, col4 = st.columns(4)

    best_mins = int(best['total_time'] // 60)
    best_secs = best['total_time'] % 60

    with col1:
        st.metric("Best Race Time", f"{best_mins}m {best_secs:.1f}s")
    with col2:
        st.metric("Optimal Stop Count", f"{best['n_stops']}-stop")
    with col3:
        st.metric("Pit Loss Time", f"{best['pit_time']:.0f}s")
    with col4:
        st.metric("Strategies Tested", f"{n_sim:,}")

    # Show the best strategy as compound pills
    st.markdown("<br>", unsafe_allow_html=True)
    pill_colours = {'SOFT': 'pill-SOFT', 'MEDIUM': 'pill-MEDIUM', 'HARD': 'pill-HARD'}

    stint_html = " &nbsp;→&nbsp; ".join([
        f'<span class="compound-pill {pill_colours.get(c, "")}">{c} — {l} laps</span>'
        for c, l in best['strategy']
    ])
    st.markdown(
        f'<div style="margin: 0.5rem 0 1.5rem;">{stint_html}</div>',
        unsafe_allow_html=True
    )

    # ── SECTION 2: 1-stop vs 2-stop comparison table ─────────────────────────

    st.markdown('<div class="section-header">Strategy Comparison</div>', unsafe_allow_html=True)

    compare_rows = []
    for n, strat in optimal['best_by_stops'].items():
        mins = int(strat['total_time'] // 60)
        secs = strat['total_time'] % 60
        strategy_str = " → ".join([f"{c} ({l} laps)" for c, l in strat['strategy']])
        gap = strat['total_time'] - best['total_time']
        compare_rows.append({
            'Stops':        f"{n}-stop",
            'Strategy':     strategy_str,
            'Pit Laps':     str(strat['pit_laps']),
            'Race Time':    f"{mins}m {secs:.2f}s",
            'Gap to Best':  f"+{gap:.2f}s" if gap > 0 else "OPTIMAL"
        })

    st.dataframe(
        pd.DataFrame(compare_rows),
        use_container_width=True,
        hide_index=True
    )

    # ── SECTION 3: Driver comparison ─────────────────────────────────────────

    st.markdown('<div class="section-header">Driver Analysis</div>', unsafe_allow_html=True)

    dcol1, dcol2 = st.columns(2)

    def render_driver_card(col, driver_code, session, model, total_laps, optimal):
        stints_df = get_pit_stops(session, total_laps)
        driver_stints = stints_df[stints_df['Driver'] == driver_code]

        if driver_stints.empty:
            col.warning(f"No data for {driver_code}")
            return

        actual_strategy = [
            (row['Compound'], int(row['StintLength']))
            for _, row in driver_stints.iterrows()
        ]
        actual_result = simulate_strategy(model, total_laps, actual_strategy)
        best          = optimal['best_overall']

        if not actual_result['valid']:
            col.warning(f"Could not simulate {driver_code}'s strategy")
            return

        delta    = actual_result['total_time'] - best['total_time']
        a_mins   = int(actual_result['total_time'] // 60)
        a_secs   = actual_result['total_time'] % 60
        faster   = delta <= 0

        with col:
            st.markdown(f"#### 🏎️ {driver_code}")

            # Actual strategy pills
            pills = " → ".join([
                f'<span class="compound-pill {pill_colours.get(c,"")}">{c} {l}L</span>'
                for c, l in actual_strategy
            ])
            st.markdown(pills, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Race Time", f"{a_mins}m {a_secs:.1f}s")
            colour = "🟢" if faster else "🔴"
            m2.metric(
                "vs Optimal",
                f"{colour} {abs(delta):.1f}s {'faster' if faster else 'slower'}"
            )

            # Pit lap annotation
            st.caption(f"Pitted on laps: {actual_result['pit_laps']}")

    render_driver_card(dcol1, driver1, session, model, total_laps, optimal)
    render_driver_card(dcol2, driver2, session, model, total_laps, optimal)

    # ── SECTION 4: Tyre Degradation Model ────────────────────────────────────

    st.markdown('<div class="section-header">Tyre Degradation Model</div>', unsafe_allow_html=True)

    tcol1, tcol2 = st.columns([2, 1])

    with tcol1:
        # Plot degradation curves
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#111111')

        colours    = {'SOFT': '#e10600', 'MEDIUM': '#e6b800', 'HARD': '#aaaaaa'}
        max_age    = min(40, total_laps)
        tyre_ages  = np.arange(1, max_age + 1)

        for compound, colour in colours.items():
            comp_data = model.get(compound)
            if comp_data is None or comp_data['deg_per_lap'] is None:
                continue
            lap_times = [predict_lap_time(model, compound, age) for age in tyre_ages]
            label = (f"{compound}  (+{comp_data['deg_per_lap']:.3f}s/lap, "
                     f"R²={comp_data['r_squared']:.2f})")
            ax.plot(tyre_ages, lap_times, color=colour, linewidth=2.5, label=label)

        ax.set_xlabel("Tyre Age (laps)", color='#888', fontsize=10)
        ax.set_ylabel("Predicted Lap Time (s)", color='#888', fontsize=10)
        ax.set_title("Degradation Curves by Compound", color='#ffffff', fontsize=12, pad=12)
        ax.tick_params(colors='#666')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax.grid(alpha=0.15, color='#444')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tcol2:
        # Deg model table
        model_rows = []
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            d = model.get(compound, {})
            if d and d.get('deg_per_lap') is not None:
                model_rows.append({
                    'Compound':   compound,
                    'Base Pace':  f"{d['base_pace']:.2f}s",
                    'Deg/lap':    f"+{d['deg_per_lap']:.4f}s",
                    'R²':         f"{d['r_squared']:.2f}",
                    'Laps Used':  d['sample_size']
                })
            else:
                model_rows.append({
                    'Compound': compound,
                    'Base Pace': '—',
                    'Deg/lap':   '—',
                    'R²':        '—',
                    'Laps Used': 0
                })
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
        st.caption("Base pace = 25th pct raw lap time. Deg rate from linear regression on normalised, fuel-corrected lap times.")

    # ── SECTION 5: Strategy lap time chart ───────────────────────────────────

    st.markdown('<div class="section-header">Lap-by-Lap Pace Comparison</div>', unsafe_allow_html=True)

    # Build results for both drivers and the optimal strategy
    to_plot   = []
    plot_labels = []

    for driver_code in [driver1, driver2]:
        stints_df     = get_pit_stops(session, total_laps)
        driver_stints = stints_df[stints_df['Driver'] == driver_code]
        if driver_stints.empty:
            continue
        actual_strategy = [(r['Compound'], int(r['StintLength'])) for _, r in driver_stints.iterrows()]
        result = simulate_strategy(model, total_laps, actual_strategy)
        if result['valid']:
            to_plot.append((result, driver_code, '#00aaff'))

    if best_1:
        to_plot.append((best_1, 'Best 1-stop', '#e10600'))
    if best_2:
        to_plot.append((best_2, 'Best 2-stop', '#ff8800'))

    if to_plot:
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        fig2.patch.set_facecolor('#0a0a0a')
        ax2.set_facecolor('#111111')

        driver_colours = {'VER': '#3671C6', 'HAM': '#00D2BE', 'LEC': '#E8002D',
                          'SAI': '#E8002D', 'NOR': '#FF8000', 'PIA': '#FF8000',
                          'RUS': '#00D2BE', 'ALO': '#358C75', 'STR': '#358C75'}
        line_styles = ['-', '--', '-.', ':']

        for i, (result, label, fallback_colour) in enumerate(to_plot):
            lap_times = result['lap_times']
            laps      = list(range(1, len(lap_times) + 1))
            colour    = driver_colours.get(label, fallback_colour)
            ax2.plot(laps, lap_times,
                     color=colour,
                     linewidth=2,
                     linestyle=line_styles[i % len(line_styles)],
                     label=label,
                     alpha=0.9)
            for pit_lap in result['pit_laps']:
                ax2.axvline(x=pit_lap, color=colour, linestyle=':', alpha=0.3, linewidth=1)

        ax2.set_xlabel("Lap Number", color='#888', fontsize=10)
        ax2.set_ylabel("Predicted Lap Time (s)", color='#888', fontsize=10)
        ax2.set_title("Predicted Lap Times — Actual vs Optimal Strategies", color='#fff', fontsize=12, pad=12)
        ax2.tick_params(colors='#666')
        ax2.spines['bottom'].set_color('#333')
        ax2.spines['left'].set_color('#333')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')
        ax2.grid(alpha=0.15, color='#444')
        st.caption("Vertical dotted lines = pit stops for each strategy")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── FOOTER ───────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#444; font-size:0.8rem; '
        'font-family: Barlow Condensed, sans-serif; letter-spacing: 0.1em;">'
        'DATA: FASTF1 OFFICIAL F1 TIMING FEED &nbsp;|&nbsp; '
        'MODEL: LINEAR REGRESSION ON FUEL-CORRECTED, DRIVER-NORMALISED LAP TIMES'
        '</div>',
        unsafe_allow_html=True
    )
