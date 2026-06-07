"""
data_fetcher.py — F1 Race Strategy Simulator
=============================================
This file handles all data loading from the FastF1 library.
It pulls lap times, tyre compounds, stint info, and pit stops
for any race you choose. Results are cached locally so you
don't re-download data every time you run the app.

Install dependencies before running:
    pip install fastf1 pandas
"""

import fastf1
import pandas as pd
import os

# ── CACHE SETUP ──────────────────────────────────────────────────────────────
# FastF1 saves downloaded data to a local folder so repeated runs are fast.
# This folder will be created automatically if it doesn't exist.

CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# ── FUNCTION 1: Get the full race calendar for a given year ──────────────────
def get_race_schedule(year: int) -> pd.DataFrame:
    """
    Returns a DataFrame with all races in a given F1 season.
    Columns include: RoundNumber, EventName, Country, Location, EventDate.

    Example:
        schedule = get_race_schedule(2024)
        print(schedule[['RoundNumber', 'EventName']])
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    # Keep only the columns we actually need
    cols = ['RoundNumber', 'EventName', 'Country', 'Location', 'EventDate']
    return schedule[cols].reset_index(drop=True)


# ── FUNCTION 2: Load a full race session ────────────────────────────────────
def load_race_session(year: int, round_number: int):
    """
    Loads and returns a FastF1 Session object for a specific race.
    Calling session.load() downloads lap data, telemetry, and tyre info.

    Parameters:
        year         — e.g. 2024
        round_number — e.g. 1 for Bahrain, 2 for Saudi Arabia, etc.

    Returns:
        A FastF1 Session object with all data attached.
    """
    session = fastf1.get_session(year, round_number, 'R')  # 'R' = Race
    session.load(telemetry=False, weather=False, messages=False)
    # We disable telemetry and weather to keep loading fast.
    # We only need lap-level data for the strategy simulator.
    return session


# ── FUNCTION 3: Get lap-by-lap data for all drivers ─────────────────────────
def get_all_laps(session) -> pd.DataFrame:
    """
    Extracts clean lap-by-lap data for every driver in a session.

    Key columns returned:
        Driver      — 3-letter code (e.g. 'VER', 'HAM', 'LEC')
        LapNumber   — which lap it was
        LapTime     — lap time as a float in seconds
        Compound    — tyre type: SOFT, MEDIUM, HARD, INTERMEDIATE, WET
        TyreLife    — how many laps old the tyre is
        Stint       — which stint the driver is on (resets after each pit stop)
        PitInTime   — time of pit entry (NaT if no pit on this lap)
        PitOutTime  — time of pit exit (NaT if no pit-out on this lap)
        IsAccurate  — FastF1 flag: False means the lap time is unreliable

    Returns a cleaned pandas DataFrame.
    """
    laps = session.laps.copy()

    # Convert LapTime from timedelta to seconds (easier to work with)
    laps['LapTime'] = laps['LapTime'].dt.total_seconds()

    # Keep only laps FastF1 considers accurate (removes in/out laps, SC laps etc.)
    laps = laps[laps['IsAccurate'] == True].copy()

    # Select only the columns we need
    cols = [
        'Driver', 'LapNumber', 'LapTime',
        'Compound', 'TyreLife', 'Stint',
        'PitInTime', 'PitOutTime'
    ]
    laps = laps[cols].reset_index(drop=True)

    return laps


# ── FUNCTION 4: Get lap data for a single driver ─────────────────────────────
def get_driver_laps(session, driver_code: str) -> pd.DataFrame:
    """
    Returns lap-by-lap data for one specific driver.

    Parameters:
        session     — FastF1 Session object (from load_race_session)
        driver_code — 3-letter code e.g. 'VER', 'NOR', 'LEC'

    Returns a cleaned DataFrame for just that driver.
    """
    laps = get_all_laps(session)
    driver_laps = laps[laps['Driver'] == driver_code].copy()

    if driver_laps.empty:
        print(f"⚠️  No data found for driver: {driver_code}")
        print(f"Available drivers: {laps['Driver'].unique().tolist()}")

    return driver_laps.reset_index(drop=True)


# ── FUNCTION 5: Get pit stop summary for all drivers ────────────────────────
def get_pit_stops(session, total_laps: int = None) -> pd.DataFrame:
    """
    Returns a summary of when each driver pitted and which compounds they used.

    Each row = one stint.
    Columns: Driver, Stint, Compound, StartLap, EndLap, StintLength

    Uses RAW session laps (not just accurate ones) so stint lengths reflect
    the full race distance. If total_laps is provided, the last stint for
    each driver is extended to reach the end of the race — so stints always
    add up to total_laps and the strategy comparison is apples-to-apples.
    """
    # Use raw session laps so we don't lose in/out laps from the count
    laps = session.laps.copy()
    laps['LapTime'] = laps['LapTime'].dt.total_seconds()

    # Drop laps with no compound info (rare edge case)
    laps = laps[laps['Compound'].notna() & (laps['Compound'] != '')].copy()

    # Group by driver and stint to find compound used and start/end laps
    stints = (
        laps.groupby(['Driver', 'Stint', 'Compound'])
        .agg(
            StartLap=('LapNumber', 'min'),
            EndLap=('LapNumber', 'max')
        )
        .reset_index()
    )

    # Calculate how many laps each stint lasted
    stints['StintLength'] = stints['EndLap'] - stints['StartLap'] + 1

    # If total_laps given, stretch the last stint of each driver to race end.
    # This corrects for any laps dropped in the data so stints sum correctly.
    if total_laps is not None:
        for driver in stints['Driver'].unique():
            mask     = stints['Driver'] == driver
            last_idx = stints[mask].index[-1]
            last_end = stints.loc[last_idx, 'EndLap']
            if last_end < total_laps:
                extra = total_laps - last_end
                stints.loc[last_idx, 'StintLength'] += extra
                stints.loc[last_idx, 'EndLap']       = total_laps

    return stints.sort_values(['Driver', 'Stint']).reset_index(drop=True)


# ── FUNCTION 6: Get list of drivers in a session ────────────────────────────
def get_driver_list(session) -> list:
    """
    Returns a sorted list of all driver codes in a session.
    Useful for populating dropdown menus in the Streamlit app.

    Example output: ['ALB', 'ALO', 'BOT', 'GAS', 'HAM', ...]
    """
    laps = session.laps.copy()
    return sorted(laps['Driver'].dropna().unique().tolist())


# ── QUICK TEST ───────────────────────────────────────────────────────────────
# Run this file directly to make sure everything works.
# It will load the 2024 Bahrain GP and print a summary.

if __name__ == "__main__":
    print("=" * 55)
    print("F1 Strategy Simulator — data_fetcher.py Test Run")
    print("=" * 55)

    # Step 1: Print the 2024 race calendar
    print("\n📅 Loading 2024 F1 Calendar...")
    schedule = get_race_schedule(2024)
    print(schedule[['RoundNumber', 'EventName']].to_string(index=False))

    # Step 2: Load the 2024 Bahrain GP (Round 1)
    print("\n⏳ Loading 2024 Bahrain Grand Prix session (Round 1)...")
    print("   (First run will download data — this takes ~30 seconds)")
    session = load_race_session(2024, 1)

    # Step 3: Print list of drivers
    drivers = get_driver_list(session)
    print(f"\n🏎️  Drivers in this race: {drivers}")

    # Step 4: Print lap data for Verstappen
    print("\n📊 Lap data for VER (Verstappen):")
    ver_laps = get_driver_laps(session, 'VER')
    print(ver_laps.head(10).to_string(index=False))

    # Step 5: Print pit stop / stint summary for all drivers
    print("\n🔧 Pit stop & stint summary (all drivers):")
    stints = get_pit_stops(session)
    print(stints.to_string(index=False))

    print("\n✅ data_fetcher.py is working correctly!")
    print(f"   Cached data saved to: ./{CACHE_DIR}/")