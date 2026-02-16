# MLB Underestimated Player Analyzer

Statcast-powered dashboard that identifies underestimated MLB players by comparing rolling wOBA against expected wOBA (xwOBA). Data is fetched from [Baseball Savant](https://baseballsavant.mlb.com) and refreshed daily via GitHub Actions.

**Live dashboard:** `https://lch99310.github.io/Daily_underestimated_MLBplayer_ranking`

## How it works

The system computes `diff_rolling_OBA` — the difference between a player's rolling wOBA and xwOBA over configurable plate-appearance windows (50/100/150 PA). Players with negative values are underperforming their expected output based on contact quality, meaning they are likely underestimated.

## Files

| File | Purpose |
|------|---------|
| `fetch_data.py` | Fetches real Statcast data from Baseball Savant, computes rolling metrics, outputs `player_data.json` |
| `build.py` | Embeds `player_data.json` into `mlb-analyzer.html` to produce a self-contained `index.html` |
| `mlb-analyzer.html` | Dashboard template (not deployed directly) |
| `index.html` | Built dashboard with embedded data (this is what GitHub Pages serves) |
| `player_data.json` | Raw data output from the pipeline |
| `.github/workflows/update-data.yml` | GitHub Actions workflow that runs daily |

## Setup

### 1. Create GitHub repo

```bash
# Create a new repo on GitHub, then:
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

### 2. Enable GitHub Pages

1. Go to your repo → **Settings** → **Pages**
2. Under "Source", select **Deploy from a branch**
3. Select **main** branch and **/ (root)** folder
4. Click **Save**

Your dashboard will be live at `https://<your-username>.github.io/<repo-name>/`

### 3. Daily auto-refresh

The GitHub Actions workflow (`.github/workflows/update-data.yml`) runs automatically:
- **Daily at 8:00 AM UTC** (3:00 AM ET, after Statcast updates)
- Can also be triggered manually from the Actions tab
- Fetches fresh data, rebuilds `index.html`, and commits the changes
- GitHub Pages auto-deploys on each commit

During the MLB offseason (Nov–Mar), the data won't change, but the workflow will still run harmlessly.

### 4. Run locally

```bash
pip install pybaseball pandas numpy
python fetch_data.py    # Fetch data (~5-10 min first run)
python build.py         # Build self-contained index.html
# Open index.html in a browser — no server needed
```

## Data sources

- **Pitch-level event data**: Baseball Savant Statcast (via [pybaseball](https://github.com/jldbc/pybaseball))
- **Season-level expected stats**: wOBA, xwOBA, xBA, xSLG
- **Exit velocity & barrels**: avg EV, hard hit %, barrel %
- **Rolling metrics**: Computed from per-PA `woba_value` and `estimated_woba_using_speedangle`
