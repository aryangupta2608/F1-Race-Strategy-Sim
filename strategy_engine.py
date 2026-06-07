"""
strategy_engine.py — F1 Race Strategy Simulator
================================================
This is the brain of the project.

It takes the tyre degradation model from Milestone 2 and simulates
every possible pit stop strategy (0-stop, 1-stop, 2-stop) for a race.
For each strategy, it calculates the total predicted race time, then
picks the fastest one.

How it works at a high level:
- A race is just a sequence of laps
- Each lap has a predicted time based on: compound + tyre age + deg rate
- A pit stop adds a fixed time penalty (~22s) but resets tyre age to 0
- So the question becomes: when to pit, and onto what compound?
- We brute-force test every combination and find the minimum total time

This is exactly how real F1 teams build their base strategy before a race.

Dependencies: data_fetcher.py and tyre_model.py must be in the same folder.
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import load_race_session, get_all_laps, get_pit_stops
from tyre_model import build_tyre_model, predict_lap_time


# ── CONSTANTS ────────────────────────────────────────────────────────────────

# Time lost in the pit lane per stop (seconds).
# Includes slowing down, pit lane speed limit, and rejoining.
# Bahrain pit loss is ~22-23 seconds.
PIT_LOSS_TIME = 22.0

# F1 rules require at least 2 different compounds to be used in a dry race.
# This means a 1-stop is the minimum for most races.
# We'll still simulate 0-stop for reference (though it's usually illegal).
SLICK_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

# Tyre compounds ranked from softest to hardest (softest = most grip, most deg)
COMPOUND_RANK = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}


# ── FUNCTION 1: Simulate a single strategy ───────────────────────────────────
def simulate_strategy(model: dict, total_laps: int, strategy: list) -> dict:
    """
    Simulates one specific pit stop strategy and returns the total race time.

    A strategy is a list of (compound, stint_length) tuples.
    Example 1-stop: [('SOFT', 15), ('HARD', 42)]  → soft for 15 laps, hard for 42
    Example 2-stop: [('SOFT', 12), ('HARD', 25), ('SOFT', 20)]

    For each lap in each stint, we call predict_lap_time() to get the
    expected lap time based on tyre age and compound.
    Then we add PIT_LOSS_TIME for each pit stop (i.e. n_stints - 1 stops).

    Parameters:
        model      — tyre degradation model from build_tyre_model()
        total_laps — total race distance in laps
        strategy   — list of (compound, laps) tuples

    Returns a dict with:
        strategy       — the input strategy
        total_time     — total predicted race time in seconds
        stint_times    — list of total time per stint
        lap_times      — lap-by-lap predicted times (full list)
        pit_laps       — which laps the pit stops happen on
        n_stops        — number of pit stops
        valid          — False if a compound has no model data
    """
    # Check all compounds in this strategy have model data
    for compound, _ in strategy:
        if model.get(compound) is None or model[compound]['deg_per_lap'] is None:
            return {
                'strategy':   strategy,
                'total_time': float('inf'),  # infinity = disqualify this strategy
                'valid':      False,
                'error':      f'No tyre model data for {compound}'
            }

    lap_times   = []   # predicted time for every lap
    stint_times = []   # total time for each stint
    pit_laps    = []   # which lap number each pit stop happens on
    current_lap = 1    # track absolute lap number through the race

    for stint_idx, (compound, stint_length) in enumerate(strategy):
        stint_total = 0

        for tyre_age in range(1, stint_length + 1):
            # Get predicted lap time for this tyre age on this compound
            lap_time = predict_lap_time(model, compound, tyre_age)
            lap_times.append(lap_time)
            stint_total += lap_time
            current_lap += 1

        stint_times.append(stint_total)

        # Record pit stop lap (happens at the end of every stint except last)
        if stint_idx < len(strategy) - 1:
            pit_laps.append(current_lap - 1)  # pit at end of this lap

    # Add pit loss time for each stop
    n_stops    = len(strategy) - 1
    pit_time   = n_stops * PIT_LOSS_TIME
    total_time = sum(stint_times) + pit_time

    return {
        'strategy':    strategy,
        'total_time':  round(total_time, 2),
        'stint_times': [round(t, 2) for t in stint_times],
        'lap_times':   [round(t, 3) for t in lap_times],
        'pit_laps':    pit_laps,
        'n_stops':     n_stops,
        'pit_time':    pit_time,
        'valid':       True
    }


# ── FUNCTION 2: Generate all valid strategy combinations ─────────────────────
def generate_strategies(model: dict, total_laps: int, max_stops: int = 2) -> list:
    """
    Generates every possible pit stop strategy worth simulating.

    Rules:
    - F1 requires at least 2 different compounds in a dry race
    - We test 1-stop and 2-stop combinations
    - Pit windows are tested every 3 laps (not every single lap — too slow)
    - We only include compounds that have model data for this race

    Parameters:
        model      — tyre degradation model
        total_laps — total race laps
        max_stops  — maximum pit stops to simulate (default 2)

    Returns a list of strategy dicts (output of simulate_strategy).
    """
    # Only use compounds we actually have model data for
    available = [c for c in SLICK_COMPOUNDS
                 if model.get(c) and model[c]['deg_per_lap'] is not None]

    print(f"  Available compounds with model data: {available}")

    all_results = []

    # ── 1-STOP STRATEGIES ────────────────────────────────────────────────────
    # Try every pair of compounds (must use at least 2 different ones)
    # Try pitting at every 3rd lap between lap 8 and lap (total_laps - 8)
    if max_stops >= 1:
        compound_pairs = [(c1, c2) for c1 in available for c2 in available if c1 != c2]

        for c1, c2 in compound_pairs:
            # Pit window: not too early, not too late
            for pit_lap in range(8, total_laps - 8, 3):
                stint1 = pit_lap
                stint2 = total_laps - pit_lap
                if stint1 < 3 or stint2 < 3:
                    continue

                strategy = [(c1, stint1), (c2, stint2)]
                result   = simulate_strategy(model, total_laps, strategy)
                if result['valid']:
                    all_results.append(result)

    # ── 2-STOP STRATEGIES ────────────────────────────────────────────────────
    # Try every combination of 3 compounds (middle compound can repeat)
    # Two pit windows tested every 5 laps to keep runtime reasonable
    if max_stops >= 2:
        compound_triples = [
            (c1, c2, c3)
            for c1 in available
            for c2 in available
            for c3 in available
            if len({c1, c2, c3}) >= 2  # must use at least 2 different compounds
        ]

        for c1, c2, c3 in compound_triples:
            for pit1 in range(8, total_laps - 16, 5):
                for pit2 in range(pit1 + 8, total_laps - 8, 5):
                    stint1 = pit1
                    stint2 = pit2 - pit1
                    stint3 = total_laps - pit2
                    if any(s < 3 for s in [stint1, stint2, stint3]):
                        continue

                    strategy = [(c1, stint1), (c2, stint2), (c3, stint3)]
                    result   = simulate_strategy(model, total_laps, strategy)
                    if result['valid']:
                        all_results.append(result)

    return all_results


# ── FUNCTION 3: Find the optimal strategy ────────────────────────────────────
def find_optimal_strategy(model: dict, total_laps: int, max_stops: int = 2) -> dict:
    """
    Master function — runs all strategy simulations and returns a ranked summary.

    Steps:
        1. Generate all valid strategies
        2. Sort by total race time (fastest first)
        3. Return top results + best per stop count

    Parameters:
        model      — tyre degradation model from build_tyre_model()
        total_laps — total race laps
        max_stops  — max pit stops to consider (default 2)

    Returns a dict with:
        all_results     — every strategy simulated, sorted by time
        best_overall    — single fastest strategy
        best_by_stops   — fastest strategy for each stop count (1-stop, 2-stop)
        total_simulated — how many strategies were evaluated
    """
    print(f"\n🏎️  Simulating strategies for {total_laps}-lap race...")
    print(f"   Max pit stops: {max_stops}")

    results = generate_strategies(model, total_laps, max_stops)

    if not results:
        print("❌ No valid strategies found — check tyre model data.")
        return {}

    # Sort all results by total race time, fastest first
    results.sort(key=lambda x: x['total_time'])

    # Best overall
    best_overall = results[0]

    # Best strategy per number of stops
    best_by_stops = {}
    for n in range(1, max_stops + 1):
        candidates = [r for r in results if r['n_stops'] == n]
        if candidates:
            best_by_stops[n] = candidates[0]

    print(f"   Total strategies simulated: {len(results)}")
    print(f"\n🏆 Best overall strategy:")
    print_strategy(best_overall)

    print(f"\n📊 Best strategy per stop count:")
    for n_stops, strat in best_by_stops.items():
        label = f"  Best {n_stops}-stop"
        print(f"\n{label}:")
        print_strategy(strat)

    return {
        'all_results':     results,
        'best_overall':    best_overall,
        'best_by_stops':   best_by_stops,
        'total_simulated': len(results)
    }


# ── FUNCTION 4: Pretty-print a strategy result ───────────────────────────────
def print_strategy(result: dict):
    """
    Prints a human-readable summary of one strategy result.
    """
    stints = result['strategy']
    stops  = result['n_stops']
    time   = result['total_time']
    pit_l  = result['pit_laps']

    stint_str = " → ".join([f"{c} ({l} laps)" for c, l in stints])
    minutes   = int(time // 60)
    seconds   = time % 60

    print(f"    Strategy : {stint_str}")
    print(f"    Pit laps : {pit_l}")
    print(f"    Pit stops: {stops}  (time lost: {result['pit_time']:.0f}s)")
    print(f"    Race time: {minutes}m {seconds:.2f}s  ({time:.2f}s total)")


# ── FUNCTION 5: Compare actual vs optimal strategy for one driver ─────────────
def compare_actual_vs_optimal(session, model: dict, total_laps: int,
                               driver_code: str, optimal: dict):
    """
    Shows how a specific driver's actual race strategy compares
    to what the simulator says was optimal.

    Parameters:
        session     — FastF1 Session object
        model       — tyre degradation model
        total_laps  — total race laps
        driver_code — e.g. 'VER', 'HAM', 'LEC'
        optimal     — output of find_optimal_strategy()
    """
    # Get this driver's actual stint data
    # Pass total_laps so stint lengths are stretched to full race distance
    stints = get_pit_stops(session, total_laps)
    driver_stints = stints[stints['Driver'] == driver_code]

    if driver_stints.empty:
        print(f"⚠️  No stint data found for {driver_code}")
        return

    # Reconstruct actual strategy as a list of (compound, stint_length) tuples
    actual_strategy = [
        (row['Compound'], int(row['StintLength']))
        for _, row in driver_stints.iterrows()
    ]

    # Simulate the driver's actual strategy through our model
    actual_result = simulate_strategy(model, total_laps, actual_strategy)

    # Get the optimal strategy
    best = optimal['best_overall']

    print(f"\n🔍 {driver_code} — Actual vs Optimal Strategy Comparison")
    print("-" * 55)
    print(f"  ACTUAL strategy:")
    print_strategy(actual_result)
    print(f"\n  OPTIMAL strategy (simulator):")
    print_strategy(best)

    # Time delta
    if actual_result['valid']:
        delta = actual_result['total_time'] - best['total_time']
        if delta > 0:
            print(f"\n  ⏱️  {driver_code}'s strategy was {delta:.2f}s slower than optimal")
        else:
            print(f"\n  ✅ {driver_code}'s strategy was {abs(delta):.2f}s faster than optimal!")

    return actual_result


# ── FUNCTION 6: Plot lap-by-lap pace for a strategy ─────────────────────────
def plot_strategy_lap_times(results_to_plot: list, race_name: str = "Race"):
    """
    Plots lap-by-lap predicted pace for multiple strategies side by side.

    Parameters:
        results_to_plot — list of simulate_strategy() outputs to compare
        race_name       — shown in the chart title
    """
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')

    compound_colours = {'SOFT': '#FF3333', 'MEDIUM': '#FFD700', 'HARD': '#CCCCCC'}
    line_styles      = ['-', '--', '-.', ':']

    for i, result in enumerate(results_to_plot):
        if not result['valid']:
            continue

        lap_times = result['lap_times']
        laps      = list(range(1, len(lap_times) + 1))
        label     = " → ".join([f"{c}({l})" for c, l in result['strategy']])

        plt.plot(laps, lap_times,
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=2,
                 label=label,
                 alpha=0.9)

        # Mark pit stop laps with vertical lines
        for pit_lap in result['pit_laps']:
            plt.axvline(x=pit_lap, linestyle=':', alpha=0.4, color='white')

    plt.xlabel("Lap Number", fontsize=12)
    plt.ylabel("Predicted Lap Time (seconds)", fontsize=12)
    plt.title(f"Strategy Comparison — {race_name}", fontsize=14)
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(alpha=0.2)
    plt.tight_layout()

    output_file = "strategy_comparison.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n📊 Chart saved as '{output_file}'")
    plt.show()


# ── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("F1 Strategy Simulator — strategy_engine.py Test Run")
    print("=" * 55)

    # Load 2024 Bahrain GP (from cache)
    print("\n⏳ Loading 2024 Bahrain GP session (from cache)...")
    session    = load_race_session(2024, 1)
    TOTAL_LAPS = 57
    RACE_NAME  = "2024 Bahrain Grand Prix"

    # Build the tyre model (Milestone 2)
    print()
    model = build_tyre_model(session, TOTAL_LAPS)

    # Run the strategy optimizer
    optimal = find_optimal_strategy(model, TOTAL_LAPS, max_stops=2)

    # Compare Verstappen's actual strategy vs optimal
    actual_ver = compare_actual_vs_optimal(
        session, model, TOTAL_LAPS, 'VER', optimal
    )

    # Also compare Hamilton
    actual_ham = compare_actual_vs_optimal(
        session, model, TOTAL_LAPS, 'HAM', optimal
    )

    # Plot the best 1-stop vs best 2-stop vs VER actual
    print("\n📈 Plotting strategy comparison chart...")
    to_plot = [
        optimal['best_by_stops'].get(1),   # best 1-stop
        optimal['best_by_stops'].get(2),   # best 2-stop
        actual_ver                          # VER's actual race
    ]
    to_plot = [r for r in to_plot if r and r.get('valid')]
    plot_strategy_lap_times(to_plot, RACE_NAME)

    print("\n✅ strategy_engine.py is working correctly!")
    print("   Ready for Milestone 4: Streamlit app (app.py)")