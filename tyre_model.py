"""
tyre_model.py — F1 Race Strategy Simulator
===========================================
This file figures out how much slower each tyre compound gets
as it wears out over a stint.

The core idea:
- Put a driver on fresh tyres → they do fast laps
- As the tyres age, laps get progressively slower
- We measure that slowdown rate using linear regression
- Output: "SOFT degrades at +0.08s per lap, HARD at +0.04s per lap"

That degradation rate is the foundation of the strategy engine —
it lets us predict total race time for any pit stop strategy.

Install dependencies:
    pip install fastf1 pandas numpy scipy matplotlib
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress noisy FastF1 warnings

# We import our own data fetcher from Milestone 1
from data_fetcher import load_race_session, get_all_laps


# ── CONSTANTS ────────────────────────────────────────────────────────────────

# Approximate fuel burn rate in F1: cars carry ~100kg at race start,
# burning ~1.8kg per lap. Lighter car = faster lap time.
# Each kg of fuel costs roughly 0.03 seconds per lap.
# So as fuel burns off, the car naturally gets faster — we need to
# account for this so we don't confuse "fuel effect" with "tyre deg".
FUEL_EFFECT_PER_LAP = 0.03  # seconds gained per lap as fuel burns

# Tyre compounds we care about (ignore rain tyres for the simulator)
SLICK_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

# Minimum number of laps a stint needs to have for us to trust the regression.
# A 3-lap stint isn't enough data to fit a reliable degradation curve.
MIN_STINT_LAPS = 5


# ── FUNCTION 1: Fuel-correct the lap times ───────────────────────────────────
def apply_fuel_correction(laps_df: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    """
    Removes the fuel load effect from lap times so we can isolate tyre deg.

    As the race goes on, the car gets lighter and naturally faster.
    If we don't correct for this, our degradation numbers will be wrong
    (we'd think tyres are faster than they are because the car is lighter).

    Formula:
        corrected_time = raw_time - (laps_remaining * FUEL_EFFECT_PER_LAP)

    Parameters:
        laps_df    — DataFrame from get_all_laps()
        total_laps — total number of laps in the race

    Returns the same DataFrame with a new 'CorrectedLapTime' column.
    """
    df = laps_df.copy()

    # How many laps of fuel have been burned by this lap
    laps_elapsed = df['LapNumber']

    # Early laps are slower because the car is heavy with fuel.
    # As fuel burns, the car gets lighter and faster.
    # We SUBTRACT the fuel benefit to make late-race laps comparable to early laps.
    # i.e. we artificially make the car "heavy" again on every lap,
    # so that any remaining lap time increase is purely tyre deg.
    df['CorrectedLapTime'] = df['LapTime'] + (laps_elapsed * FUEL_EFFECT_PER_LAP)

    return df


# ── FUNCTION 2: Fit degradation curve for one compound ───────────────────────
def fit_compound_degradation(laps_df: pd.DataFrame, compound: str) -> dict:
    """
    For a given tyre compound, fits a linear regression of:
        CorrectedLapTime ~ TyreLife

    The slope of this line = degradation rate (seconds per lap of tyre age).

    Example output:
        {
            'compound':    'SOFT',
            'base_pace':   95.2,   # expected lap time on a brand new tyre
            'deg_per_lap': 0.082,  # gets 0.082s slower per lap
            'r_squared':   0.74,   # how well the line fits the data
            'sample_size': 87      # how many laps were used
        }

    Parameters:
        laps_df  — fuel-corrected DataFrame (from apply_fuel_correction)
        compound — 'SOFT', 'MEDIUM', or 'HARD'
    """
    # Filter to just this compound
    comp_laps = laps_df[
        (laps_df['Compound'] == compound) &
        (laps_df['TyreLife'] >= 1)
    ].copy()

    if len(comp_laps) < MIN_STINT_LAPS:
        return {
            'compound':    compound,
            'base_pace':   None,
            'deg_per_lap': None,
            'r_squared':   None,
            'sample_size': len(comp_laps),
            'error':       f'Not enough data (only {len(comp_laps)} laps)'
        }

    # ── KEY FIX: normalize each driver's laps relative to their own pace ──
    # Problem: Mercedes laps at ~97s, Williams at ~101s. If we pool all
    # drivers raw, the spread in car speed completely drowns out the small
    # tyre degradation signal, giving us a near-zero R².
    # Fix: for each driver, subtract their median lap time for this compound.
    # Now every driver is on a "0 = their normal pace" scale,
    # and we only measure the tyre age effect on top of that baseline.
    comp_laps['NormalisedLapTime'] = comp_laps.groupby('Driver')['CorrectedLapTime'].transform(
        lambda x: x - x.median()
    )

    # Remove outlier laps — more than 2.5s off each driver's own median
    # (covers yellow flags, traffic, mistakes)
    comp_laps = comp_laps[abs(comp_laps['NormalisedLapTime']) < 2.5]

    if len(comp_laps) < MIN_STINT_LAPS:
        return {
            'compound':    compound,
            'base_pace':   None,
            'deg_per_lap': None,
            'r_squared':   None,
            'sample_size': len(comp_laps),
            'error':       f'Not enough data after normalisation (only {len(comp_laps)} laps)'
        }

    # Run linear regression: TyreLife (x) vs NormalisedLapTime (y)
    # Slope = how many seconds slower per lap of tyre age (this is deg rate)
    x = comp_laps['TyreLife'].values
    y = comp_laps['NormalisedLapTime'].values

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Base pace = average of each driver's median pace on this compound
    # Gives a realistic absolute lap time anchor for predictions
    base_pace = comp_laps.groupby('Driver')['CorrectedLapTime'].median().mean()

    return {
        'compound':    compound,
        'base_pace':   round(base_pace, 3),    # realistic median lap time on this compound
        'deg_per_lap': round(slope, 4),         # seconds slower per lap of tyre age
        'r_squared':   round(r_value ** 2, 3),  # fit quality (1.0 = perfect)
        'sample_size': len(comp_laps)
    }


# ── FUNCTION 3: Fit degradation for all compounds in one race ────────────────
def build_tyre_model(session, total_laps: int) -> dict:
    """
    Master function — builds the full tyre degradation model for a race.

    Steps:
        1. Load all lap data
        2. Apply fuel correction
        3. Fit degradation curves for SOFT, MEDIUM, HARD
        4. Return a clean dictionary of results

    Parameters:
        session    — FastF1 Session object (from load_race_session)
        total_laps — total race laps (e.g. 57 for Bahrain 2024)

    Returns a dict like:
        {
            'SOFT':   { 'base_pace': 95.2, 'deg_per_lap': 0.082, ... },
            'MEDIUM': { 'base_pace': 96.1, 'deg_per_lap': 0.051, ... },
            'HARD':   { 'base_pace': 96.8, 'deg_per_lap': 0.034, ... }
        }
    """
    print("🔧 Building tyre degradation model...")

    # Step 1: Get all clean lap data
    laps = get_all_laps(session)

    # Step 2: Filter to slick compounds only
    laps = laps[laps['Compound'].isin(SLICK_COMPOUNDS)].copy()

    # Step 3: Apply fuel correction
    laps = apply_fuel_correction(laps, total_laps)

    # Step 4: Fit a regression for each compound
    model = {}
    for compound in SLICK_COMPOUNDS:
        result = fit_compound_degradation(laps, compound)
        model[compound] = result

        # Print a readable summary
        if result['deg_per_lap'] is not None:
            print(f"  {compound:8s} → base pace: {result['base_pace']:.2f}s | "
                  f"deg: +{result['deg_per_lap']:.4f}s/lap | "
                  f"R²: {result['r_squared']:.2f} | "
                  f"n={result['sample_size']} laps")
        else:
            print(f"  {compound:8s} → ⚠️  {result.get('error', 'Unknown error')}")

    return model


# ── FUNCTION 4: Predict lap time at a given tyre age ─────────────────────────
def predict_lap_time(model: dict, compound: str, tyre_age: int) -> float:
    """
    Given the tyre model, predicts the lap time for a specific tyre age.

    This is what the strategy engine will call repeatedly to simulate
    total race time for different pit stop strategies.

    Parameters:
        model     — output of build_tyre_model()
        compound  — 'SOFT', 'MEDIUM', or 'HARD'
        tyre_age  — how many laps old the tyre is (0 = brand new)

    Returns predicted lap time in seconds.

    Example:
        predict_lap_time(model, 'SOFT', 10)  → ~96.3s
        predict_lap_time(model, 'HARD', 25)  → ~98.1s
    """
    comp_data = model.get(compound)

    if comp_data is None or comp_data['deg_per_lap'] is None:
        raise ValueError(f"No degradation data available for {compound}")

    predicted = comp_data["base_pace"] + (comp_data["deg_per_lap"] * (tyre_age - 1))
    return round(predicted, 3)


# ── FUNCTION 5: Plot the degradation curves ───────────────────────────────────
def plot_degradation_curves(model: dict, total_laps: int, race_name: str = "Race"):
    """
    Draws a chart showing predicted lap time vs tyre age for each compound.
    Soft = red line, Medium = yellow line, Hard = white/grey line.

    Saves the chart as 'tyre_degradation.png' in the current folder.

    Parameters:
        model      — output of build_tyre_model()
        total_laps — used to set the x-axis range
        race_name  — shown in the chart title
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')

    # F1-style colours for each compound
    colours = {'SOFT': '#FF3333', 'MEDIUM': '#FFD700', 'HARD': '#CCCCCC'}

    for compound in SLICK_COMPOUNDS:
        comp_data = model.get(compound)
        if comp_data is None or comp_data['deg_per_lap'] is None:
            continue

        # Generate x values (tyre ages 1 to 40 laps)
        max_age = min(40, total_laps)
        tyre_ages = np.arange(1, max_age + 1)

        # Predict lap time at each tyre age
        lap_times = [predict_lap_time(model, compound, age) for age in tyre_ages]

        label = (f"{compound} "
                 f"(+{comp_data['deg_per_lap']:.3f}s/lap, "
                 f"R²={comp_data['r_squared']:.2f})")

        plt.plot(tyre_ages, lap_times,
                 color=colours[compound],
                 linewidth=2.5,
                 label=label)

    plt.xlabel("Tyre Age (laps)", fontsize=12)
    plt.ylabel("Predicted Lap Time (seconds)", fontsize=12)
    plt.title(f"Tyre Degradation Model — {race_name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = "tyre_degradation.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n📊 Chart saved as '{output_file}'")
    plt.show()


# ── QUICK TEST ────────────────────────────────────────────────────────────────
# Run this file directly to test on 2024 Bahrain GP.
# Make sure data_fetcher.py is in the same folder.

if __name__ == "__main__":
    print("=" * 55)
    print("F1 Strategy Simulator — tyre_model.py Test Run")
    print("=" * 55)

    # Load the same session from Milestone 1 (uses cache, so it's fast)
    print("\n⏳ Loading 2024 Bahrain GP session (from cache)...")
    session = load_race_session(2024, 1)

    # Bahrain 2024 had 57 laps
    TOTAL_LAPS = 57
    RACE_NAME  = "2024 Bahrain Grand Prix"

    # Build the tyre model
    print()
    model = build_tyre_model(session, TOTAL_LAPS)

    # Show some predictions
    print("\n🔮 Sample lap time predictions:")
    print(f"  SOFT  at lap  5 of stint: {predict_lap_time(model, 'SOFT',  5):.2f}s")
    print(f"  SOFT  at lap 15 of stint: {predict_lap_time(model, 'SOFT', 15):.2f}s")
    print(f"  HARD  at lap  5 of stint: {predict_lap_time(model, 'HARD',  5):.2f}s")
    print(f"  HARD  at lap 25 of stint: {predict_lap_time(model, 'HARD', 25):.2f}s")

    # Plot the degradation curves
    plot_degradation_curves(model, TOTAL_LAPS, RACE_NAME)

    print("\n✅ tyre_model.py is working correctly!")
    print("   Ready to move to Milestone 3: strategy_engine.py")