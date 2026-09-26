# IPL Team Performance & Match Analytics

A data analytics project analyzing 17 seasons of Indian Premier League (IPL) match data (2008–2024) to uncover team performance trends, toss-related patterns, venue effects, and season-wise changes.

**This is a data analytics project — no machine learning, no APIs, no databases, no web frameworks.** Just Python, Pandas, NumPy, Matplotlib, and Seaborn.

## Objective

Analyze historical IPL match-level data to answer questions such as:
- Which teams have performed best over the tournament's history?
- Does winning the toss actually help you win the match?
- Which venues have hosted the most matches, and how do teams perform there?
- Are IPL matches more often won by defending a total or by chasing?

## Dataset

- **File:** `data/matches.csv`
- **Source:** [IPL Complete Dataset (2008–2024)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) by patrickb1912 on Kaggle (originally sourced from Cricsheet)
- **Size:** 1,095 matches × 20 columns
- **Key fields used:** `season`, `date`, `venue`, `team1`, `team2`, `toss_winner`, `toss_decision`, `winner`, `result`, `result_margin`, `player_of_match`

## Technologies Used

- Python 3
- Pandas — data loading, cleaning, aggregation
- NumPy — numeric operations
- Matplotlib — plotting
- Seaborn — statistical visualizations
- Jupyter Notebook — analysis environment

## Analysis Performed

1. **Data Loading & Inspection** — shape, dtypes, missing values, duplicate check
2. **Data Cleaning**
   - Standardized team names for franchises that were renamed (e.g. Delhi Daredevils → Delhi Capitals), while keeping genuinely distinct/defunct franchises (e.g. Deccan Chargers) separate
   - Converted `date` to datetime; used the full `season` label (not just a starting year) to correctly separate seasons like `2009` and `2009/10`
   - Handled missing values contextually (e.g. `city` filled with `'Unknown'`; `winner`/`result_margin` left as `NaN` for genuinely no-result/tied matches rather than guessed)
3. **Exploratory Data Analysis**
   - Overall team performance (matches played, wins, losses, win %)
   - Season-wise trends (matches per season, top-team performance over time)
   - Toss analysis (bat vs field decision, toss-to-match-win rate, outcome by decision)
   - Venue analysis (matches per venue, team performance at the most-used venue)
   - Win margin analysis (runs-based vs wickets-based wins, margin distributions)
4. **Visualizations** — 8 charts saved to `visualizations/`

## Key Findings

1. **Mumbai Indians** have the most all-time wins in IPL history: 144 wins from 261 matches played.
2. Among teams with 30+ matches played, **Gujarat Titans** have the highest win percentage (62.22%), ahead of Chennai Super Kings (58.23%) and Lucknow Super Giants (55.81%).
3. Fielding first is the far more common toss decision — 704 matches (64.3%) vs. 391 (35.7%) choosing to bat first.
4. Winning the toss is only a slight edge: the toss winner also wins the match just 50.83% of the time (554/1,090 decisive matches) — close to a coin flip.
5. Choosing to field first after winning the toss converts to a match win more often (53.86%) than choosing to bat first (45.38%) — a gap of about 8.5 percentage points.
6. **Eden Gardens** is the most-used venue in IPL history (77 matches), just ahead of Wankhede Stadium (73 matches).
7. Wickets-based wins (578) outnumber runs-based wins (498), consistent with the fielding-first advantage above.
8. Runs-based wins average a 30.1-run margin, while wickets-based wins average a 6.2-wicket margin — most successful chases finish comfortably rather than as nail-biters.

*(Full computed output, including the raw findings text, is available in `notebooks/IPL_Team_Performance_Analytics.ipynb` and `key_findings.txt`.)*

## Project Structure

```
ipl-team-performance-analytics/
│
├── data/
│   └── matches.csv                              # Raw IPL match dataset
│
├── notebooks/
│   └── IPL_Team_Performance_Analytics.ipynb     # Full analysis notebook
│
├── visualizations/
│   ├── 01_top_teams_by_wins.png
│   ├── 02_team_win_percentage.png
│   ├── 03_matches_by_season.png
│   ├── 04_toss_decision_distribution.png
│   ├── 05_toss_winner_vs_match_winner.png
│   ├── 06_bat_vs_field_first_outcomes.png
│   ├── 07_top_venues.png
│   └── 08_win_margin_distribution.png
│
├── key_findings.txt                             # Plain-text summary of findings
├── README.md
└── requirements.txt
```

## How to Run

1. Clone/download this project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook notebooks/IPL_Team_Performance_Analytics.ipynb
   ```
4. Run all cells top to bottom (`Cell → Run All`). Plots will be regenerated into `visualizations/` and printed inline.
