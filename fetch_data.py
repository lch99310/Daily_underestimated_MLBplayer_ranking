#!/usr/bin/env python3
"""
MLB Underestimated Player Analyzer — Data Pipeline
Fetches real Statcast data from Baseball Savant via pybaseball,
computes rolling wOBA/xwOBA differentials, and outputs player_data.json.

Supports "transition mode" at the start of a new season: blends previous
season data with early new-season data until enough players qualify.
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
TRANSITION_THRESHOLD = 10  # min players with MIN_PA in new season to exit transition


def determine_season_mode():
    """
    Determine data mode:
      - "current": enough players in current year, use it alone
      - "transition": new season has started but <TRANSITION_THRESHOLD players
        meet MIN_PA — blend previous season as fallback
      - "fallback": no current year data at all, use previous year
    Returns (mode, current_year) or (mode, current_year) where mode encodes behaviour.
    """
    today = date.today()
    year = today.year

    # Try fetching current year stats
    curr_df = None
    try:
        curr_df = statcast_batter_expected_stats(year, minPA=1)
    except Exception as e:
        print(f"[WARN] Could not fetch {year} expected stats: {e}")

    if curr_df is not None and len(curr_df) > 0:
        qualified = curr_df[curr_df["pa"] >= MIN_PA] if "pa" in curr_df.columns else pd.DataFrame()
        n_qualified = len(qualified)

        if n_qualified >= TRANSITION_THRESHOLD:
            print(f"[INFO] {n_qualified} players with {MIN_PA}+ PA in {year} — full current season mode")
            return "current", year
        else:
            print(f"[INFO] Only {n_qualified} players with {MIN_PA}+ PA in {year} — transition mode")
            return "transition", year

    # No current year data at all, try previous years
    for fallback_year in [year - 1, year - 2]:
        try:
            df = statcast_batter_expected_stats(fallback_year, minPA=1)
            if df is not None and len(df) > 50:
                print(f"[INFO] No {year} data available, using {fallback_year}")
                return "current", fallback_year
        except Exception as e:
            print(f"[WARN] No data for {fallback_year}: {e}")

    print("[INFO] Hard fallback to 2024")
    return "current", 2024


def get_season_dates(year):
    """Return approximate season start/end for fetching pitch data."""
    # Regular season windows
    start = f"{year}-03-20"
    end = min(f"{year}-10-05", date.today().strftime("%Y-%m-%d"))
    return start, end


def fetch_expected_stats(year):
    """Fetch season-level wOBA, xwOBA, xBA, xSLG for all batters."""
    print(f"  Fetching expected statistics for {year}...")
    df = statcast_batter_expected_stats(year, minPA=1)
    print(f"  -> {len(df)} batters retrieved")
    return df


def fetch_ev_barrels(year):
    """Fetch exit velocity, hard hit %, barrel % for all batters."""
    print(f"  Fetching exit velocity & barrels for {year}...")
    df = statcast_batter_exitvelo_barrels(year, minBBE=1)
    print(f"  -> {len(df)} batters retrieved")
    return df


def fetch_pitch_level(year):
    """Fetch pitch-level Statcast data for the full season."""
    start, end = get_season_dates(year)
    print(f"  Fetching pitch-level data from {start} to {end}...")
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
    print("  Extracting batter team affiliations...")
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
    print("  Computing rolling metrics...")

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


def fetch_season_data(year):
    """Fetch all data sources for a given season. Returns a dict with all components."""
    print(f"\n{'='*50}")
    print(f"  Fetching {year} season data")
    print(f"{'='*50}")

    expected_df = fetch_expected_stats(year)

    try:
        ev_df = fetch_ev_barrels(year)
    except Exception as e:
        print(f"  [WARN] Could not fetch EV/barrel data for {year}: {e}")
        ev_df = None

    rolling_data = {}
    team_map = {}
    try:
        pitch_df = fetch_pitch_level(year)
        team_map = extract_batter_teams(pitch_df)
        rolling_data = compute_rolling_metrics(pitch_df)
    except Exception as e:
        print(f"  [WARN] Could not fetch/process pitch-level data for {year}: {e}")
        print(f"  [WARN] Will use season-level stats only (no rolling windows)")

    return {
        "expected_df": expected_df,
        "ev_df": ev_df,
        "rolling_data": rolling_data,
        "team_map": team_map,
    }


NAME_COL = "last_name, first_name"


def build_player(row, ev_df, rolling_data, team_map):
    """Build a single player dict from a row of expected stats + auxiliary data."""
    player_id = int(row.get("player_id", 0))
    pa = int(row.get("pa", 0))

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
        season_diff = row.get("est_woba_minus_woba_diff")
        if pd.notna(season_diff):
            player["diff_rolling_OBA"] = safe_round(-float(season_diff), 3)
        else:
            player["diff_rolling_OBA"] = 0.0

    return player


def build_output(expected_df, ev_df, rolling_data, team_map, year):
    """Merge all data sources and produce final JSON (single-season mode)."""
    print("[BUILD] Building output JSON (current season mode)...")

    players = []
    for _, row in expected_df.iterrows():
        if int(row.get("pa", 0)) < MIN_PA:
            continue
        player = build_player(row, ev_df, rolling_data, team_map)
        player["data_season"] = year
        players.append(player)

    players.sort(key=lambda p: p.get("diff_rolling_OBA", 0))

    return {
        "generated_at": datetime.now().isoformat(),
        "mode": "current",
        "season": year,
        "total_players": len(players),
        "min_pa": MIN_PA,
        "rolling_windows": ROLLING_WINDOWS,
        "players": players,
    }


def build_transition_output(prev_data, prev_year, curr_data, curr_year):
    """
    Build output in transition mode:
    - Players who meet MIN_PA in the new season → use new season data (data_season = curr_year)
    - Remaining players from previous season → keep prev data (data_season = prev_year)
    """
    print("[BUILD] Building output JSON (transition mode)...")

    curr_expected = curr_data["expected_df"]
    curr_ev = curr_data["ev_df"]
    curr_rolling = curr_data["rolling_data"]
    curr_teams = curr_data["team_map"]

    prev_expected = prev_data["expected_df"]
    prev_ev = prev_data["ev_df"]
    prev_rolling = prev_data["rolling_data"]
    prev_teams = prev_data["team_map"]

    # Identify players who qualify in the current season
    curr_qualified_ids = set()
    if curr_expected is not None and len(curr_expected) > 0:
        for _, row in curr_expected.iterrows():
            if int(row.get("pa", 0)) >= MIN_PA:
                curr_qualified_ids.add(int(row.get("player_id", 0)))

    players = []
    seen_ids = set()

    # 1) Add current-season qualified players
    if curr_expected is not None:
        for _, row in curr_expected.iterrows():
            pid = int(row.get("player_id", 0))
            if pid not in curr_qualified_ids:
                continue
            player = build_player(row, curr_ev, curr_rolling, curr_teams)
            player["data_season"] = curr_year
            players.append(player)
            seen_ids.add(pid)

    # 2) Add previous-season players (as fallback)
    for _, row in prev_expected.iterrows():
        pid = int(row.get("player_id", 0))
        if pid in seen_ids or int(row.get("pa", 0)) < MIN_PA:
            continue
        player = build_player(row, prev_ev, prev_rolling, prev_teams)
        player["data_season"] = prev_year
        players.append(player)
        seen_ids.add(pid)

    players.sort(key=lambda p: p.get("diff_rolling_OBA", 0))

    n_curr = len(curr_qualified_ids)
    n_prev = len(players) - n_curr

    print(f"  -> {n_curr} players from {curr_year} (new season)")
    print(f"  -> {n_prev} players from {prev_year} (fallback)")
    print(f"  -> {len(players)} total")

    return {
        "generated_at": datetime.now().isoformat(),
        "mode": "transition",
        "season": curr_year,
        "fallback_season": prev_year,
        "current_season_players": n_curr,
        "total_players": len(players),
        "min_pa": MIN_PA,
        "rolling_windows": ROLLING_WINDOWS,
        "transition_threshold": TRANSITION_THRESHOLD,
        "players": players,
    }


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

    mode, year = determine_season_mode()

    if mode == "transition":
        prev_year = year - 1
        print(f"\n>>> TRANSITION MODE: blending {prev_year} (fallback) + {year} (new season)\n")

        # Fetch previous season (full dataset as fallback)
        prev_data = fetch_season_data(prev_year)

        # Fetch current season (may be sparse early on)
        curr_data = fetch_season_data(year)

        # Build merged output
        output = build_transition_output(prev_data, prev_year, curr_data, year)

    else:
        # Normal single-season mode
        print(f"\n>>> Using season: {year}\n")
        season_data = fetch_season_data(year)
        output = build_output(
            season_data["expected_df"],
            season_data["ev_df"],
            season_data["rolling_data"],
            season_data["team_map"],
            year,
        )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[DONE] Saved {output['total_players']} players to {OUTPUT_FILE}")
    print(f"       Mode: {output['mode']}")
    print(f"       Season: {output['season']}")
    if output["mode"] == "transition":
        print(f"       Fallback season: {output['fallback_season']}")
        print(f"       New season players: {output['current_season_players']}")
    print(f"       Generated: {output['generated_at']}")


if __name__ == "__main__":
    main()
