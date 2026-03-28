#!/usr/bin/env python3
"""
MLB Underestimated Player Analyzer — Data Pipeline
Fetches real Statcast data from Baseball Savant via pybaseball,
computes rolling wOBA/xwOBA differentials, and outputs player_data.json.

Supports cross-season rolling windows at the start of a new season:
pitch-level data from both seasons is concatenated so rolling windows
physically span the season boundary (e.g. last 30 PA from 2025 + first
20 PA from 2026 = 50 PA window).  Once enough players qualify in the
new season alone, the pipeline switches to single-season mode.
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
MIN_PA = 50
TRANSITION_THRESHOLD = 10  # new-season players needed to exit transition


# ═══════════════════════════════════════════════════════════════════
# SEASON MODE DETECTION
# ═══════════════════════════════════════════════════════════════════

def determine_season_mode():
    """
    Returns (mode, year):
      "current", year  — enough players qualify in year, single-season
      "transition", year — new season started, <TRANSITION_THRESHOLD qualify
    """
    today = date.today()
    year = today.year

    curr_df = None
    try:
        curr_df = statcast_batter_expected_stats(year, minPA=1)
    except Exception as e:
        print(f"[WARN] Could not fetch {year} expected stats: {e}")

    if curr_df is not None and len(curr_df) > 0:
        qualified = curr_df[curr_df["pa"] >= MIN_PA] if "pa" in curr_df.columns else pd.DataFrame()
        n = len(qualified)
        if n >= TRANSITION_THRESHOLD:
            print(f"[INFO] {n} players with {MIN_PA}+ PA in {year} — single-season mode")
            return "current", year
        else:
            print(f"[INFO] Only {n} players with {MIN_PA}+ PA in {year} — transition mode")
            return "transition", year

    for fallback in [year - 1, year - 2]:
        try:
            df = statcast_batter_expected_stats(fallback, minPA=1)
            if df is not None and len(df) > 50:
                print(f"[INFO] No {year} data, using {fallback}")
                return "current", fallback
        except Exception as e:
            print(f"[WARN] No data for {fallback}: {e}")

    print("[INFO] Hard fallback to 2024")
    return "current", 2024


# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════

def get_season_dates(year):
    start = f"{year}-03-20"
    end = min(f"{year}-10-05", date.today().strftime("%Y-%m-%d"))
    return start, end


def fetch_expected_stats(year):
    print(f"  Fetching expected statistics for {year}...")
    df = statcast_batter_expected_stats(year, minPA=1)
    print(f"  -> {len(df)} batters retrieved")
    return df


def fetch_ev_barrels(year):
    print(f"  Fetching exit velocity & barrels for {year}...")
    df = statcast_batter_exitvelo_barrels(year, minBBE=1)
    print(f"  -> {len(df)} batters retrieved")
    return df


def fetch_pitch_level(year):
    start, end = get_season_dates(year)
    print(f"  Fetching pitch-level data from {start} to {end}...")
    print("  (this may take several minutes)")
    df = statcast(start_dt=start, end_dt=end)
    print(f"  -> {len(df)} total pitches retrieved")
    return df


def extract_batter_teams(pitch_df):
    print("  Extracting batter team affiliations...")
    pa_df = pitch_df[pitch_df["events"].notna()].copy()
    pa_df["bat_team"] = pa_df.apply(
        lambda r: r["away_team"] if r["inning_topbot"] == "Top" else r["home_team"],
        axis=1,
    )
    pa_df = pa_df.sort_values("game_date")
    team_map = pa_df.groupby("batter")["bat_team"].last().to_dict()
    print(f"  -> {len(team_map)} batter-team mappings extracted")
    return team_map


# ═══════════════════════════════════════════════════════════════════
# ROLLING METRIC COMPUTATION (supports cross-season)
# ═══════════════════════════════════════════════════════════════════

def compute_rolling_metrics(pitch_df):
    """
    Compute rolling wOBA/xwOBA for every batter with MIN_PA+ plate appearances.

    If pitch_df contains a ``data_season`` column (cross-season concatenated
    data), the output includes per-window ``trend_seasons`` arrays and
    per-batter ``new_season_pa`` / ``cross_season`` flags.
    """
    print("  Computing rolling metrics...")

    pa_df = pitch_df[pitch_df["events"].notna()].copy()
    pa_df = pa_df.sort_values(["game_date", "at_bat_number"]).reset_index(drop=True)

    has_season_col = "data_season" in pa_df.columns

    pa_df["woba_value"] = pd.to_numeric(pa_df["woba_value"], errors="coerce").fillna(0)
    pa_df["woba_denom"] = pd.to_numeric(pa_df["woba_denom"], errors="coerce").fillna(0)
    pa_df["estimated_woba_using_speedangle"] = pd.to_numeric(
        pa_df["estimated_woba_using_speedangle"], errors="coerce"
    )
    pa_df["xwoba_value"] = pa_df["estimated_woba_using_speedangle"].fillna(
        pa_df["woba_value"]
    )

    results = {}
    grouped = pa_df.groupby("batter")
    total_batters = len(grouped)
    processed = 0

    for batter_id, batter_pa in grouped:
        batter_pa = batter_pa.sort_values(["game_date", "at_bat_number"]).reset_index(drop=True)
        total_pa = len(batter_pa)

        if total_pa < MIN_PA:
            continue

        # Record last PA date
        last_date = batter_pa["game_date"].iloc[-1]
        if hasattr(last_date, "strftime"):
            batter_result_date = last_date.strftime("%Y-%m-%d")
        else:
            batter_result_date = str(last_date)[:10]

        batter_result = {"total_pa": total_pa, "last_pa_date": batter_result_date, "windows": {}}

        # Season tracking
        if has_season_col:
            season_arr = batter_pa["data_season"].values
            unique_seasons = sorted(set(int(s) for s in season_arr))
            if len(unique_seasons) > 1:
                new_season = max(unique_seasons)
                batter_result["new_season_pa"] = int((season_arr == new_season).sum())
                batter_result["cross_season"] = True
            else:
                batter_result["new_season_pa"] = total_pa
                batter_result["cross_season"] = False

        for window in ROLLING_WINDOWS:
            if total_pa < window:
                continue

            woba_num = batter_pa["woba_value"].rolling(window=window, min_periods=window).sum()
            woba_den = batter_pa["woba_denom"].rolling(window=window, min_periods=window).sum()
            xwoba_num = batter_pa["xwoba_value"].rolling(window=window, min_periods=window).sum()

            rolling_woba = woba_num / woba_den
            rolling_xwoba = xwoba_num / woba_den
            rolling_diff = rolling_woba - rolling_xwoba

            latest_woba = rolling_woba.iloc[-1] if not pd.isna(rolling_woba.iloc[-1]) else None
            latest_xwoba = rolling_xwoba.iloc[-1] if not pd.isna(rolling_xwoba.iloc[-1]) else None
            latest_diff = rolling_diff.iloc[-1] if not pd.isna(rolling_diff.iloc[-1]) else None

            valid_woba = rolling_woba.dropna()
            valid_xwoba = rolling_xwoba.dropna()

            # Keep the last `window` valid rolling values so the chart's
            # data-point count matches its PA label (50 PA → 50 points, etc.)
            if len(valid_woba) > window:
                valid_woba = valid_woba.iloc[-window:]
                valid_xwoba = valid_xwoba.iloc[-window:]

            # Store all kept points at full resolution
            if len(valid_woba) > 0:
                trend_woba = [round(float(v), 3) for v in valid_woba.tolist()]
                trend_xwoba = [round(float(v), 3) for v in valid_xwoba.tolist()]
            else:
                trend_woba = []
                trend_xwoba = []

            valid_idx = valid_woba.index.values if len(valid_woba) > 0 else []

            win_result = {
                "rolling_woba": round(float(latest_woba), 3) if latest_woba is not None else None,
                "rolling_xwoba": round(float(latest_xwoba), 3) if latest_xwoba is not None else None,
                "diff_rolling_OBA": round(float(latest_diff), 3) if latest_diff is not None else None,
                "trend_woba": trend_woba,
                "trend_xwoba": trend_xwoba,
                "trend_diff": [round(float(trend_woba[i] - trend_xwoba[i]), 3) for i in range(len(trend_woba))],
            }

            # Cross-season: record which season each trend point belongs to
            if has_season_col and len(valid_idx) > 0:
                win_result["trend_seasons"] = [int(season_arr[pi]) for pi in valid_idx]

            batter_result["windows"][str(window)] = win_result

        if batter_result["windows"]:
            results[int(batter_id)] = batter_result

        processed += 1
        if processed % 50 == 0:
            print(f"  -> Processed {processed}/{total_batters} batters...")

    print(f"  -> {len(results)} batters with {MIN_PA}+ PA computed")
    return results


# ═══════════════════════════════════════════════════════════════════
# PLAYER BUILDER (shared by both modes)
# ═══════════════════════════════════════════════════════════════════

NAME_COL = "last_name, first_name"


def build_player(row, ev_df, rolling_data, team_map):
    """Build a single player dict from expected-stats row + auxiliary data."""
    player_id = int(row.get("player_id", 0))
    pa = int(row.get("pa", 0))

    raw_name = str(row.get(NAME_COL, "")).strip()
    if "," in raw_name:
        parts = raw_name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    else:
        name = raw_name

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

    if ev_df is not None:
        ev_row = ev_df[ev_df["player_id"] == player_id]
        if len(ev_row) > 0:
            ev_row = ev_row.iloc[0]
            player["exit_velocity"] = safe_round(ev_row.get("avg_hit_speed"), 1)
            player["launch_angle"] = safe_round(ev_row.get("avg_hit_angle"), 1)
            player["hard_hit_pct"] = safe_round(ev_row.get("ev95percent"), 1)
            player["barrel_pct"] = safe_round(ev_row.get("brl_percent"), 1)
            player["max_exit_velocity"] = safe_round(ev_row.get("max_hit_speed"), 1)

    if player_id in rolling_data:
        rd = rolling_data[player_id]
        player["rolling"] = rd["windows"]
        player["total_pa_events"] = rd["total_pa"]
        if "last_pa_date" in rd:
            player["last_pa_date"] = rd["last_pa_date"]

        for w in [100, 50, 250]:
            wkey = str(w)
            if wkey in rd["windows"]:
                wd = rd["windows"][wkey]
                if wd.get("diff_rolling_OBA") is not None:
                    player["diff_rolling_OBA"] = wd["diff_rolling_OBA"]
                    break

    if "diff_rolling_OBA" not in player:
        season_diff = row.get("est_woba_minus_woba_diff")
        if pd.notna(season_diff):
            player["diff_rolling_OBA"] = safe_round(-float(season_diff), 3)
        else:
            player["diff_rolling_OBA"] = 0.0

    return player


# ═══════════════════════════════════════════════════════════════════
# OUTPUT BUILDERS
# ═══════════════════════════════════════════════════════════════════

def build_output(expected_df, ev_df, rolling_data, team_map, year):
    """Single-season output."""
    print("[BUILD] Building output (single-season mode)...")
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


def build_transition_output(prev_expected, prev_ev, prev_year,
                            curr_expected, curr_ev, curr_year,
                            rolling_data, team_map):
    """
    Cross-season transition output.
    Rolling metrics come from concatenated pitch data (cross-season).
    Season-level stats come from the most appropriate single season.
    """
    print("[BUILD] Building output (transition mode — cross-season rolling)...")

    # Lookup dicts for season-level stats
    curr_lookup = {}
    if curr_expected is not None and len(curr_expected) > 0:
        for _, row in curr_expected.iterrows():
            curr_lookup[int(row["player_id"])] = row

    prev_lookup = {}
    for _, row in prev_expected.iterrows():
        prev_lookup[int(row["player_id"])] = row

    players = []

    # Include every batter that has rolling data (MIN_PA+ across seasons)
    for pid, roll_info in rolling_data.items():
        # Pick season-level stats: prefer current season if they have MIN_PA there
        if pid in curr_lookup and int(curr_lookup[pid].get("pa", 0)) >= MIN_PA:
            row = curr_lookup[pid]
            ev_df = curr_ev
            data_season = curr_year
        elif pid in prev_lookup:
            row = prev_lookup[pid]
            ev_df = prev_ev
            data_season = prev_year
        elif pid in curr_lookup:
            # Has some current season data but not MIN_PA
            row = curr_lookup[pid]
            ev_df = curr_ev
            data_season = curr_year
        else:
            continue

        player = build_player(row, ev_df, rolling_data, team_map)
        player["data_season"] = data_season

        if roll_info.get("cross_season"):
            player["cross_season"] = True
            player["new_season_pa"] = roll_info.get("new_season_pa", 0)

        players.append(player)

    players.sort(key=lambda p: p.get("diff_rolling_OBA", 0))

    n_curr = sum(1 for p in players if p.get("data_season") == curr_year)

    print(f"  -> {n_curr} players with {curr_year} season-level stats")
    print(f"  -> {len(players) - n_curr} players with {prev_year} season-level stats")
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
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("MLB Underestimated Player Analyzer — Data Pipeline")
    print("=" * 60)

    mode, year = determine_season_mode()

    if mode == "transition":
        prev_year = year - 1
        print(f"\n>>> TRANSITION MODE: cross-season rolling {prev_year} → {year}\n")

        # ---- previous season ----
        print(f"\n{'='*50}")
        print(f"  Fetching {prev_year} season data (fallback)")
        print(f"{'='*50}")
        prev_expected = fetch_expected_stats(prev_year)
        try:
            prev_ev = fetch_ev_barrels(prev_year)
        except Exception as e:
            print(f"  [WARN] No EV data for {prev_year}: {e}")
            prev_ev = None
        try:
            prev_pitch = fetch_pitch_level(prev_year)
        except Exception as e:
            print(f"  [WARN] No pitch data for {prev_year}: {e}")
            prev_pitch = pd.DataFrame()

        # ---- current season ----
        print(f"\n{'='*50}")
        print(f"  Fetching {year} season data (current)")
        print(f"{'='*50}")
        try:
            curr_expected = fetch_expected_stats(year)
        except Exception as e:
            print(f"  [WARN] No expected stats for {year}: {e}")
            curr_expected = pd.DataFrame()
        try:
            curr_ev = fetch_ev_barrels(year)
        except Exception as e:
            print(f"  [WARN] No EV data for {year}: {e}")
            curr_ev = None
        try:
            curr_pitch = fetch_pitch_level(year)
        except Exception as e:
            print(f"  [WARN] No pitch data for {year}: {e}")
            curr_pitch = pd.DataFrame()

        # ---- concatenate pitch data with season markers ----
        print("\n  Concatenating pitch data across seasons...")
        frames = []
        if len(prev_pitch) > 0:
            p = prev_pitch.copy()
            p["data_season"] = prev_year
            frames.append(p)
        if len(curr_pitch) > 0:
            c = curr_pitch.copy()
            c["data_season"] = year
            frames.append(c)

        if frames:
            combined_pitch = pd.concat(frames, ignore_index=True)
            print(f"  -> {len(combined_pitch)} total pitches across both seasons")
        else:
            combined_pitch = pd.DataFrame()

        # ---- team mappings (prefer current season) ----
        team_map = {}
        if len(prev_pitch) > 0:
            team_map.update(extract_batter_teams(prev_pitch))
        if len(curr_pitch) > 0:
            team_map.update(extract_batter_teams(curr_pitch))

        # ---- cross-season rolling metrics ----
        rolling_data = {}
        if len(combined_pitch) > 0:
            print("\n  Computing cross-season rolling metrics...")
            rolling_data = compute_rolling_metrics(combined_pitch)

        # ---- build output ----
        output = build_transition_output(
            prev_expected, prev_ev, prev_year,
            curr_expected, curr_ev, year,
            rolling_data, team_map,
        )

    else:
        # ---- single-season mode ----
        print(f"\n>>> Using season: {year}\n")
        expected_df = fetch_expected_stats(year)
        try:
            ev_df = fetch_ev_barrels(year)
        except Exception as e:
            print(f"  [WARN] No EV data: {e}")
            ev_df = None

        rolling_data = {}
        team_map = {}
        try:
            pitch_df = fetch_pitch_level(year)
            team_map = extract_batter_teams(pitch_df)
            rolling_data = compute_rolling_metrics(pitch_df)
        except Exception as e:
            print(f"  [WARN] No pitch data: {e}")

        output = build_output(expected_df, ev_df, rolling_data, team_map, year)

    # ---- save ----
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[DONE] Saved {output['total_players']} players to {OUTPUT_FILE}")
    print(f"       Mode: {output['mode']}")
    print(f"       Season: {output['season']}")
    if output["mode"] == "transition":
        print(f"       Fallback: {output['fallback_season']}")
        print(f"       New-season stats: {output['current_season_players']}")
    print(f"       Generated: {output['generated_at']}")


if __name__ == "__main__":
    main()
