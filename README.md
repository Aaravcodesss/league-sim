<div align="center">

# 📊 league-sim

**Who's winning the league? Simulate the rest of the season 10,000 times and find out.**

A Monte Carlo football season simulator in Python. Give it team ratings and the fixtures (with results so far), and it plays out the rest of the season thousands of times. It then tells you each team's chances of winning the **title**, finishing in the **top 4**, or going **down**.

[![Tests](https://github.com/Aaravcodesss/league-sim/actions/workflows/tests.yml/badge.svg)](https://github.com/Aaravcodesss/league-sim/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-00e5ff?logo=python&logoColor=white&labelColor=05070d)
![numpy](https://img.shields.io/badge/numpy-vectorised-39ff88?labelColor=05070d)

</div>

```text
  Projected final table — 10,000 simulated seasons  (0.90s)

   #  Team                      Pts   Title                       Top 4    Down
  ─────────────────────────────────────────────────────────────────────────────
   1  Regex Rovers             71.3   27.7%  █████                74.2%     –
   2  Stack Overflow Rovers    69.6   23.5%  ████                 68.7%   <0.1%
   3  Code FC                  69.6   21.9%  ████                 68.0%   <0.1%
   4  Async Athletic           66.2   11.3%  ██                   52.5%   <0.1%
   5  Lambda Albion            62.3    5.5%  █                    34.4%   <0.1%
  ...
  18  Bugs FC                  40.1   <0.1%                        0.1%   36.7%
  19  Segfault Town            38.6     –                         <0.1%   44.6%
  20  Spaghetti Code Villa     24.5     –                           –     95.1%
```

<sub>The sample league is made up (20 teams with coding-pun names, 14 rounds already played). Swap in a real league's CSVs to use it for real.</sub>

## Quick start

```bash
git clone https://github.com/Aaravcodesss/league-sim.git
cd league-sim
pip install -e .            # only needs numpy

league-sim --teams data/teams.csv --fixtures data/fixtures.csv
```

More things you can do:

```bash
league-sim --teams data/teams.csv --odds "Code FC" "Bugs FC"     # odds for a single match
league-sim --teams data/teams.csv                                # simulate a whole new season from scratch
league-sim --teams data/teams.csv --fixtures data/fixtures.csv \
           -n 50000 --seed 7 --json projection.json             # more sims, reproducible, save as JSON
league-sim --help
```

| Option | What it does |
|:--|:--|
| `-n, --sims` | Number of seasons to simulate (default 10,000) |
| `--seed` | Fix the random seed so you get the same results every run |
| `--static` | Keep ratings fixed instead of updating Elo after each simulated match |
| `--top`, `--relegated` | Size of the top zone and the relegation zone (default 4 and 3) |
| `--json FILE` | Write the projection to a JSON file |
| `--odds HOME AWAY` | Home win, draw and away win probabilities for one fixture |

## How it works

### 1. Ratings → expected goals
Each team has an **Elo rating**. For a match, the rating gap (plus a home-advantage bonus) sets each side's **expected goals**:

```
λ_home = 1.35 · e^( 0.0018 · (Elo_home + 40 − Elo_away))
λ_away = 1.20 · e^(−0.0018 · (Elo_home + 40 − Elo_away))
```

### 2. Expected goals → a scoreline
Goals are sampled from a **Poisson distribution** with those means. This naturally gives realistic scores like 1-0, 1-1 and 2-1. The constants are tuned so two equal teams produce **45% home wins, 26% draws and 29% away wins**, which is close to long-run averages in Europe's top leagues.

### 3. Ratings move during the season
After every simulated match, both teams' Elo ratings are updated (K = 20, with a bonus for bigger winning margins). So a team that goes on a run *in one simulated season* stays in form for the rest of it. This gives a more realistic spread of outcomes than fixed ratings. Turn it off with `--static`.

### 4. All seasons at once
Instead of looping through 10,000 seasons one by one, each fixture is sampled **for every simulated season at once** using numpy arrays of shape `(sims, teams)`. Ten thousand seasons of a 380-game league take about **one second**.

### 5. Final tables
Teams are ranked by points, then goal difference, then goals scored, then drawing lots. That gives a finishing position for every team in every simulated season, and the title, top-4 and relegation odds are just averages over those positions.

## Tests

```bash
python -m unittest discover -s tests -v
```

There are 17 tests. Highlights:
- The fast numpy simulation **matches the exact expected points** worked out by hand from the Poisson model, to within 0.05 points.
- Results that have already been played are always respected.
- Every simulated table is a valid ranking, and every position probability matrix sums to 1 across rows and columns.
- The same seed always gives the same result.

## Use your own league

`teams.csv`
```csv
team,elo
Code FC,1720
Bugs FC,1440
```

`fixtures.csv`: leave the goals blank for matches that haven't been played yet.
```csv
home,away,home_goals,away_goals
Code FC,Bugs FC,3,1
Bugs FC,Code FC,,
```

The ratings in `teams.csv` are treated as **pre-season** ratings. Results already played update them before the rest of the season is simulated.

## Project layout

```
league_sim/
  model.py    # Elo → expected goals → Poisson scorelines, Elo updates, exact match odds
  season.py   # CSV loading, fixture generation, vectorised season simulation, tables
  cli.py      # command-line interface
tests/        # unittest suite (runs in CI on Python 3.10, 3.12, 3.13)
data/         # sample league
```

## Ideas for next

- [ ] Fetch real fixtures and results from a public API
- [ ] Fit Elo ratings automatically from past results
- [ ] Position probability heatmap (matplotlib)
- [ ] Separate attack and defence strengths (Dixon–Coles model)

## License

[MIT](LICENSE) © Aarav
