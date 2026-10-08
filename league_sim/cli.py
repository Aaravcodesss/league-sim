"""Command-line interface.

    python -m league_sim --teams data/teams.csv --fixtures data/fixtures.csv -n 10000
    python -m league_sim --teams data/teams.csv --odds "Code FC" "Bugs FC"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .model import outcome_probabilities
from .season import League, current_table, simulate

BAR = "█"


class Style:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _c(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def green(self, s): return self._c("92", s)
    def cyan(self, s): return self._c("96", s)
    def red(self, s): return self._c("91", s)
    def dim(self, s): return self._c("2", s)
    def bold(self, s): return self._c("1", s)


def pct(p: float) -> str:
    if p == 0:
        return "   –  "
    if p < 0.001:
        return " <0.1%"
    return f"{p * 100:5.1f}%"


def bar(p: float, width: int = 18) -> str:
    n = round(p * width)
    return BAR * n + " " * (width - n)


def print_projection(rows, n_sims, top, relegated, st: Style, elapsed: float):
    name_w = max(len(r["team"]) for r in rows) + 2
    print()
    print(st.bold(f"  Projected final table — {n_sims:,} simulated seasons") + st.dim(f"  ({elapsed:.2f}s)"))
    print()
    head = f"  {'#':>2}  {'Team':<{name_w}}{'Pts':>6}  {'Title':>6}  {'':18}  {f'Top {top}':>6}  {'Down':>6}"
    print(st.dim(head))
    print(st.dim("  " + "─" * (len(head) - 2)))
    total = len(rows)
    for i, r in enumerate(rows, 1):
        line_name = f"{r['team']:<{name_w}}"
        if i <= top:
            line_name = st.cyan(line_name)
        elif i > total - relegated:
            line_name = st.red(line_name)
        print(
            f"  {i:>2}  {line_name}{r['avg_points']:6.1f}  {pct(r['title'])}  "
            f"{st.green(bar(r['title']))}  {pct(r['top'])}  {st.red(pct(r['relegation']))}"
        )
    print()


def print_current(table, st: Style):
    if not any(r["p"] for r in table):
        return
    name_w = max(len(r["team"]) for r in table) + 2
    print()
    print(st.bold("  Current table"))
    print(st.dim(f"  {'#':>2}  {'Team':<{name_w}} {'P':>3} {'W':>3} {'D':>3} {'L':>3} {'GD':>4} {'Pts':>4}"))
    for i, r in enumerate(table, 1):
        gd = r["gf"] - r["ga"]
        print(f"  {i:>2}  {r['team']:<{name_w}} {r['p']:>3} {r['w']:>3} {r['d']:>3} {r['l']:>3} {gd:>+4} {r['pts']:>4}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="league-sim", description="Monte Carlo football season simulator (Elo + Poisson).")
    p.add_argument("--teams", required=True, help="CSV with columns: team, elo")
    p.add_argument("--fixtures", help="CSV with columns: home, away, home_goals, away_goals (blank = not played yet). "
                                      "If omitted, a full double round-robin is generated.")
    p.add_argument("-n", "--sims", type=int, default=10_000, help="number of seasons to simulate (default 10000)")
    p.add_argument("--seed", type=int, help="random seed for reproducible results")
    p.add_argument("--static", action="store_true", help="keep ratings fixed (no in-season Elo updates)")
    p.add_argument("--top", type=int, default=4, help="size of the 'top' zone (default 4)")
    p.add_argument("--relegated", type=int, default=3, help="number of relegation places (default 3)")
    p.add_argument("--json", metavar="FILE", help="also write the projection to a JSON file")
    p.add_argument("--odds", nargs=2, metavar=("HOME", "AWAY"), help="just print win/draw/loss odds for one match")
    p.add_argument("--no-color", action="store_true", help="disable coloured output")
    args = p.parse_args(argv)

    st = Style(sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ)

    try:
        league = League.from_csv(args.teams, args.fixtures)
    except (OSError, ValueError, KeyError) as e:
        print(f"league-sim: {e}", file=sys.stderr)
        return 2

    if args.odds:
        home, away = args.odds
        missing = [t for t in (home, away) if t not in league.teams]
        if missing:
            print(f"league-sim: unknown team(s): {', '.join(missing)}", file=sys.stderr)
            return 2
        w, d, l = outcome_probabilities(league.teams[home], league.teams[away])
        print(f"\n  {st.cyan(home)} vs {st.red(away)}")
        print(f"  Home win {pct(w)}  {st.green(bar(w))}")
        print(f"  Draw     {pct(d)}  {st.dim(bar(d))}")
        print(f"  Away win {pct(l)}  {st.red(bar(l))}\n")
        return 0

    if args.sims < 1:
        print("league-sim: --sims must be at least 1", file=sys.stderr)
        return 2

    played = sum(f.played for f in league.fixtures)
    print(st.dim(f"\n  {len(league.teams)} teams · {len(league.fixtures)} fixtures · {played} played · "
                 f"{len(league.fixtures) - played} to simulate"))
    print_current(current_table(league), st)

    t0 = time.perf_counter()
    result = simulate(league, n_sims=args.sims, seed=args.seed, dynamic=not args.static)
    elapsed = time.perf_counter() - t0
    rows = result.summary(top=args.top, relegated=args.relegated)
    print_projection(rows, result.n_sims, args.top, args.relegated, st, elapsed)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"sims": result.n_sims, "seed": args.seed, "teams": rows}, fh, indent=2)
        print(st.dim(f"  wrote {args.json}\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
