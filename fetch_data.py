#!/usr/bin/env python3
"""
MLB Underestimated Player Analyzer — Data Pipeline
Fetches real Statcast data from Baseball Savant via pybaseball,
computes rolling wOBA/xwOBA differentials, and outputs player_data.json.
"""

import json
import sys
import os
from datetime import datetime, date

import pandas as pd
import numpy as np
from pybaseball import (
    cache,
    statcast,
    statcast_batter_expected_stats,
    statcast_batter_exitvelo_barrels,
)

cache.enable()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "player_data.json")

ROLLING_WINDOWS = [50, 100, 250]
TREND_POINTS = 20  # number of points in trend arrays (more for retro visualization)
MIN_PA = 50


def determine_season():
    """Try 2025 first; fall back to 2024 if no data."""
    today = date.today()
    year = today.year
    # MLB season typically starts late March
    # Try current year first
    for try_year in [year, year - 1]:
        print(f"[INFO] Trying season {try_year}...")
        try:
            df = statcast_batter_expected_stats(try_year, minPA=1)
            if df is not None and len(df) > 50:
                print(f"[INFO] Found {len(df)} batters for {try_year}")
                return try_year
        except Exception as e:
            print(f"[WARN] No data for {try_year}: {e}")
    # Hard fallback
    print("[INFO] Falling back to 2024")
    return 2024


def get_season_dates(year):
    """Return approximate season start/end for fetching pitch data."""
    # Regular season windows
    start = f"{year}-03-20"
    end = min(f"{year}-10-05", date.today().strftime("%Y-%m-%d"))
    return start, end


def fetch_expected_stats(year):
    """Fetch season-level wOBA, xwOBA, xBA, xSLG for all batters."""
    print("[STEP 1] Fetching expected statistics...")
    df = statcast_batter_expected_stats(year, minPA=1)
    print(f"  -> {len(df)} batters retrieved")
    return df


def fetch_ev_barrels(year):
    """Fetch exit velocity, hard hit %, barrel % for all batters."""
    print("[STEP 2] Fetching exit velocity & barrels...")
    df = statcast_batter_exitvelo_barrels(year, minBBE=1)
    print(f"  -> {len(df)} batters retrieved")
    return df


def fetch_pitch_level(year):
    """Fetch pitch-level Statcast data for the full season."""
    start, end = get_season_dates(year)
    print(f"[STEP 3] Fetching pitch-level data from {start} to {end}...")
    print("  (this may take several minutes due to the volume of data)")
    df = statcast(start_dt=start, end_dt=end)
    print(f"  -> {len(df)} total pitches retrieved")
    return df


def extract_batter_teams(pitch_df):
    """
    Derive each batter's most recent team from pitch-level data.
    Top of inning = away team batting, bottom = home team batting.
    Returns dict: {batter_id: team_abbrev}
    """
    print("[STEP 3b] Extracting batter team affiliations...")
    pa_df = pitch_df[pitch_df["events"].notna()].copy()
    pa_df["bat_team"] = pa_df.apply(
        lambda r: r["away_team"] if r["inning_topbot"] == "Top" else r["home_team"],
        axis=1,
    )
    # Use most recent game to determine current team
    pa_df = pa_df.sort_values("game_date")
    team_map = pa_df.groupby("batter")["bat_team"].last().to_dict()
    print(f"  -> {len(team_map)} batter-team mappings extracted")
    return team_map


def compute_rolling_metrics(pitch_df):
    """
    From pitch-level data, compute rolling wOBA/xwOBA for each batter
    at 50/100/250 PA windows.
    Returns a dict keyed by batter ID with rolling data.
    """
    print("[STEP 4] Computing rolling metrics...")

    # Filter to plate-appearance-ending events only
    pa_df = pitch_df[pitch_df["events"].notna()].copy()
    pa_df = pa_df.sort_values(["game_date", "at_bat_number"]).reset_index(drop=True)

    # Ensure numeric columns
    pa_df["woba_value"] = pd.to_numeric(pa_df["woba_value"], errors="coerce").fillna(0)
    pa_df["woba_denom"] = pd.to_numeric(pa_df["woba_denom"], errors="coerce").fillna(0)
    pa_df["estimated_woba_using_speedangle"] = pd.to_numeric(
        pa_df["estimated_woba_using_speedangle"], errors="coerce"
    )

    # For xwOBA: use estimated_woba_using_speedangle for batted balls,
    # fall back to woba_value for non-batted-ball events (K, BB, HBP)
    pa_df["xwoba_value"] = pa_df["estimated_woba_using_speedangle"].fillna(
        pa_df["woba_value"]
    )

    # Group by batter
    batter_col = "batter"
    results = {}
    grouped = pa_df.groupby(batter_col)
    total_batters = len(grouped)
    processed = 0

    for batter_id, batter_pa in grouped:
        batter_pa = batter_pa.sort_values(["game_date", "at_bat_number"]).reset_index(
            drop=True
        )
        total_pa = len(batter_pa)

        if total_pa < MIN_PA:
            continue

        batter_result = {"total_pa": total_pa, "windows": {}}

        for window in ROLLING_WINDOWS:
            if total_pa < window:
                continue

            # Compute rolling sums
            woba_num = batter_pa["woba_value"].rolling(window=window, min_periods=window).sum()
            woba_den = batter_pa["woba_denom"].rolling(window=window, min_periods=window).sum()
            xwoba_num = batter_pa["xwoba_value"].rolling(window=window, min_periods=window).sum()

            rolling_woba = woba_num / woba_den
            rolling_xwoba = xwoba_num / woba_den
            rolling_diff = rolling_woba - rolling_xwoba

            # Current (latest) values
            latest_woba = rolling_woba.iloc[-1] if not pd.isna(rolling_woba.iloc[-1]) else None
            latest_xwoba = rolling_xwoba.iloc[-1] if not pd.isna(rolling_xwoba.iloc[-1]) else None
            latest_diff = rolling_diff.iloc[-1] if not pd.isna(rolling_diff.iloc[-1]) else None

            # Trend: last TREND_POINTS valid values, evenly spaced
            valid_woba = rolling_woba.dropna()
            valid_xwoba = rolling_xwoba.dropna()

            if len(valid_woba) >= TREND_POINTS:
                indices = np.linspace(0, len(valid_woba) - 1, TREND_POINTS, dtype=int)
                trend_woba = [round(float(valid_woba.iloc[i]), 3) for i in indices]
                trend_xwoba = [round(float(valid_xwoba.iloc[i]), 3) for i in indices]
            elif len(valid_woba) > 0:
                trend_woba = [round(float(v), 3) for v in valid_woba.tolist()]
                trend_xwoba = [round(float(v), 3) for v in valid_xwoba.tolist()]
            else:
                trend_woba = []
                trend_xwoba = []

            batter_result["windows"][str(window)] = {
                "rolling_woba": round(float(latest_woba), 3) if latest_woba is not None else None,
                "rolling_xwoba": round(float(latest_xwoba), 3) if latest_xwoba is not None else None,
                "diff_rolling_OBA": round(float(latest_diff), 3) if latest_diff is not None else None,
                "trend_woba": trend_woba,
                "trend_xwoba": trend_xwoba,
                "trend_diff": [round(float(trend_woba[i] - trend_xwoba[i]), 3) for i in range(len(trend_woba))],
            }

        if batter_result["windows"]:
            results[int(batter_id)] = batter_result

        processed += 1
        if processed % 50 == 0:
            print(f"  -> Processed {processed}/{total_batters} batters...")

    print(f"  -> {len(results)} batters with {MIN_PA}+ PA computed")
    return results


def build_output(expected_df, ev_df, rolling_data, team_map, year):
    """Merge all data sources and produce final JSON."""
    print("[STEP 5] Building output JSON...")

    # The expected stats column is literally "last_name, first_name" (one column)
    NAME_COL = "last_name, first_name"

    players = []

    for _, row in expected_df.iterrows():
        player_id = int(row.get("player_id", 0))
        pa = int(row.get("pa", 0))

        if pa < MIN_PA:
            continue

        # Parse the combined name column: "Lastname, Firstname" -> "Firstname Lastname"
        raw_name = str(row.get(NAME_COL, "")).strip()
        if "," in raw_name:
            parts = raw_name.split(",", 1)
            name = f"{parts[1].strip()} {parts[0].strip()}"
        else:
            name = raw_name

        # Get team from pitch-level data
        team = team_map.get(player_id, "")

        player = {
            "player_id": player_id,
            "name": name,
            "team": team,
            "pa": pa,
            "batting_avg": safe_round(row.get("ba"), 3),
            "wOBA": safe_round(row.get("woba"), 3),
            "xwOBA": safe_round(row.get("est_woba"), 3),
            "diff_season": safe_round(row.get("est_woba_minus_woba_diff"), 3),
            "xBA": safe_round(row.get("est_ba"), 3),
            "xSLG": safe_round(row.get("est_slg"), 3),
        }

        # Merge exit velocity / barrel data
        if ev_df is not None:
            ev_row = ev_df[ev_df["player_id"] == player_id]
            if len(ev_row) > 0:
                ev_row = ev_row.iloc[0]
                player["exit_velocity"] = safe_round(ev_row.get("avg_hit_speed"), 1)
                player["launch_angle"] = safe_round(ev_row.get("avg_hit_angle"), 1)
                player["hard_hit_pct"] = safe_round(ev_row.get("ev95percent"), 1)
                player["barrel_pct"] = safe_round(ev_row.get("brl_percent"), 1)
                player["max_exit_velocity"] = safe_round(ev_row.get("max_hit_speed"), 1)

        # Merge rolling data
        if player_id in rolling_data:
            player["rolling"] = rolling_data[player_id]["windows"]
            player["total_pa_events"] = rolling_data[player_id]["total_pa"]

            # Use 100 PA window as the primary diff if available, else 50
            for w in [100, 50, 250]:
                wkey = str(w)
                if wkey in rolling_data[player_id]["windows"]:
                    wd = rolling_data[player_id]["windows"][wkey]
                    if wd.get("diff_rolling_OBA") is not None:
                        player["diff_rolling_OBA"] = wd["diff_rolling_OBA"]
                        break

        # Fallback: use season-level diff if no rolling diff
        if "diff_rolling_OBA" not in player:
            # est_woba_minus_woba_diff is (xwOBA - wOBA) from savant, we want (wOBA - xwOBA)
            season_diff = row.get("est_woba_minus_woba_diff")
            if pd.notna(season_diff):
                player["diff_rolling_OBA"] = safe_round(-float(season_diff), 3)
            else:
                player["diff_rolling_OBA"] = 0.0

        players.append(player)

    # Sort by diff_rolling_OBA ascending (most underestimated first)
    players.sort(key=lambda p: p.get("diff_rolling_OBA", 0))

    output = {
        "generated_at": datetime.now().isoformat(),
        "season": year,
        "total_players": len(players),
        "min_pa": MIN_PA,
        "rolling_windows": ROLLING_WINDOWS,
        "players": players,
    }

    return output


def safe_round(val, decimals=3):
    """Safely round a value, handling None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


def main():
    print("=" * 60)
    print("MLB Underestimated Player Analyzer — Data Pipeline")
    print("=" * 60)

    year = determine_season()
    print(f"\n>>> Using season: {year}\n")

    # Step 1: Season-level expected stats
    expected_df = fetch_expected_stats(year)

    # Step 2: Exit velocity & barrels
    try:
        ev_df = fetch_ev_barrels(year)
    except Exception as e:
        print(f"[WARN] Could not fetch EV/barrel data: {e}")
        ev_df = None

    # Step 3: Pitch-level data for rolling calculations
    try:
        pitch_df = fetch_pitch_level(year)
        # Step 3b: Extract team affiliations
        team_map = extract_batter_teams(pitch_df)
        # Step 4: Compute rolling metrics
        rolling_data = compute_rolling_metrics(pitch_df)
    except Exception as e:
        print(f"[WARN] Could not fetch/process pitch-level data: {e}")
        print("[WARN] Will use season-level stats only (no rolling windows)")
        rolling_data = {}
        team_map = {}

    # Step 5: Build and save output
    output = build_output(expected_df, ev_df, rolling_data, team_map, year)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[DONE] Saved {output['total_players']} players to {OUTPUT_FILE}")
    print(f"       Season: {year}")
    print(f"       Generated: {output['generated_at']}")


if __name__ == "__main__":
    main()
