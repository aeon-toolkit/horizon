# DrCIF (Δy) and deseasonalising (2026-10-28/29)

Does M4-style deseasonalising help or hurt DrCIF (Δy)? Three versions of the
same forecaster (DrCIF regressor, 50 trees, windows of 100 levels, target the
next change), aeon 8585c7bf:

- **with**: the paper's runs. M4 deseasonalising up front (`--deseasonalise`):
  90% ACF test at the seasonal period, then classical decomposition,
  multiplicative, or additive when the series is not strictly positive. The
  test only considers lags up to 10·log10 n, so periods 48, 52, 144 and 1440
  are never deseasonalised at these lengths.
- **without**: no seasonal treatment (`scripts/aaltd_noseason.py`,
  `scratch/run_m4_long_noseason.sh`).
- **4th representation** (prototype, `scripts/drcif4.py`): no deseasonalising
  up front; each window has a second channel holding the seasonally adjusted
  series (same decomposition, fitted on the whole training series, no
  seasonality test, whenever there are three seasons). DrCIF's three
  representations (series, differences, periodogram) use the raw channel and
  a fourth uses the adjusted one, so the forest chooses. `scratch/run_drcif4.sh`.

## LSSH-99 (99 series, 30 one-step origins, 3,000 fits per version)

Relative MAE (MAE over Naive's, geometric mean over the 98 series where it is
defined). Wilcoxon signed-rank on per-series MAE.
`scripts/lssh99_drcif_deseason_compare.py` →
`results/aeon-8585c7bf/lssh99-drcif-deseason/`.

| Series | n | with | without | 4th rep. | 4th rep. vs with (better/worse, p) |
|---|---:|---:|---:|---:|---|
| All | 98 | 0.781 | **0.744** | 0.745 | 49/49, 0.80 |
| Sub-daily | 36 | 0.549 | **0.483** | 0.487 | 18/18, 0.85 |
| Daily or slower | 62 | 0.958 | 0.957 | **0.955** | 31/31, 0.41 |
| with: additive (zeros) | 9 | 0.712 | **0.487** | 0.499 | **9/0, 0.004** |
| with: multiplicative | 52 | 0.840 | 0.820 | **0.815** | 28/24, 0.83 |
| with: not deseasonalised | 37 | **0.720** | 0.721 | 0.725 | 21/16, 0.28 |

- Without vs with: 37 series better, 31 worse, p = 0.29 overall. The gain is
  the additive (zero-heavy) series: traffic hourly ×5, pedestrian ×3, sunspot;
  worse with deseasonalising on 8 of 9 (p = 0.055), traffic T1 and T4 more
  than twice as bad. Multiplicative series: level (one outlier, M4 hourly
  T171, is 4× worse with deseasonalising; without it 0.861 vs 0.864).
- Where the test did not fire, "with" and "without" should be identical; they
  differ by up to 2% (weather, solar), no direction: run-to-run numerical noise.
- 4th representation: matches "without" (no significant difference anywhere)
  and fixes the zero-heavy series (better than "with" on all 9). Slightly
  worse on periods 144 and 1440 (7 of 8 series), where it adjusts series the
  M4 test never would.
- Fit time on 10,000-point series: ~570 s per refit for the 4th
  representation, ~450 s for DrCIF (Δy) (different machine loads).

## Long M4 series (≥ 500 training points, with a seasonal period)

M4 has 5,070 series with ≥ 500 training points (yearly 12, quarterly 43,
monthly 1,403, weekly 252, daily 2,946, hourly 414). Yearly, weekly and daily
have period 1 in M4, where deseasonalising does nothing, so the comparison uses
the 1,860 quarterly, monthly and hourly series (the test fired on 1,857).
Standard M4 protocol, OWA against the official Naive2 on this subset.
`scripts/m4_long.py` (subset wrapper), `scripts/m4_long_deseason_compare.py` →
`results/aeon-8585c7bf/m4-long-deseason/`.

| | n | with | without | 4th rep. |
|---|---:|---:|---:|---:|
| Quarterly | 43 | **1.249** | 1.399 | 1.344 |
| Monthly | 1,403 | 1.218 | **1.181** | 1.194 |
| Hourly | 414 | **0.643** | 0.685 | 0.696 |
| All | 1,860 | 1.007 | **1.004** | 1.015 |

- Without vs with: monthly better without (MASE p = 0.006, 746 vs 655
  series); hourly the measures disagree: MASE better without (1.13 vs 1.26,
  235 vs 178 series, p = 0.006) but sMAPE much worse (16.5 vs 14.0), driven by
  the block H125–H162 (sMAPE 44.4 with, 61.5 without). Quarterly: too few
  series to tell.
- 4th representation: like "without" (nothing significant against it);
  slightly worse than "with" on MASE overall (903 vs 957 series, p = 0.025).
  It does not recover the gain of deseasonalising on H125–H162 (sMAPE 64.5).
- On M4 (all positive) all deseasonalising is multiplicative, and it is about
  neutral overall, as on LSSH-99's multiplicative series.

## Conclusions

1. The harm from deseasonalising DrCIF on LSSH-99 is the **additive**
   treatment of series with zeros. Multiplicative deseasonalising is roughly
   neutral on both LSSH-99 and long M4.
2. The 4th (adjusted) representation behaves like not deseasonalising: it
   removes the harm on zero-heavy series but does not keep the benefit on the
   M4 hourly series where deseasonalising matters, costs ~25% more fit time,
   and gains nothing reliable. Not worth adding as it stands.
3. Untested, cheap from existing runs (no refits): deseasonalise
   multiplicatively only and leave non-positive series raw ("with" on
   multiplicative series, "without" on the rest). Possible prototype
   variations: restrict the adjusted channel to series where the M4 test
   fires; more intervals on the adjusted channel.
4. The paper's results (fixed) are unchanged: DrCIF (Δy) is first on LSSH-99
   in all three versions.
