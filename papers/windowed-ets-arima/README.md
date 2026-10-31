# Training windows for automatic ARIMA and ETS on long series (paper draft)

First draft, 2026-10-27. `main.tex` (compiles with pdflatex + bibtex, also on
Overleaf), `main.pdf` the current build. Open items are marked `\todo{}` in red.

| file | |
|---|---|
| `main.tex` | the paper |
| `refs.bib` | bibliography, every entry verified (see `refs_verification.md`) |
| `check_refs.py` | Crossref lookup and BibTeX fetch by DOI; `refs_crossref.bib` is its raw output |
| `evaluate_schemes.py` | scores every combination scheme from the forecast store |
| `make_tables.py` | builds `tables/*.tex` from `results/aeon-8585c7bf/` (do not edit the tables by hand) |

Rebuild everything from stored forecasts (minutes):

```bash
source paper/env.sh
$PY scripts/window_store.py build --lssh99 $RESULTS/lssh99-window-fitstats-8585c7bf \
    --m4 $RESULTS/m4-window-fitstats-8585c7bf --out $RESULTS/window-store-8585c7bf
$PY papers/windowed-ets-arima/evaluate_schemes.py
$PY papers/windowed-ets-arima/make_tables.py
cd papers/windowed-ets-arima && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The window runs themselves: `scripts/lssh99_window.py`, `scripts/m4_window.py`
(commands in `scratch/run_window_fitstats.sh`; about 1.5 hours on 32 cores).
Experiment notes and plan: `notes/lssh99-window-experiment.md`; earlier write-up:
`reports/windowed-arima-ets.md`.
