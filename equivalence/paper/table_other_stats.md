| aeon | reference | sim. ratio n=100 | sim. ratio n=500 | M4 ratio | M4 TOST | M4 freq. aeon/ref better | speed-up n=100 |
|---|---|---|---|---|---|---|---|
| Theta | statsforecast Theta | 0.999+ | 0.998+ | 1.000 | equivalent | 4/1 | 1.9 |
| Theta | M4 Theta | - | - | 1.000 | equivalent | 3/1 | - |
| DOTM | statsforecast DOTM | 1.002 | 1.001 | 0.990 | equivalent | 3/2 | 3.0 |
| AutoCES | statsforecast AutoCES | 1.000 | 1.000 | 1.000 | equivalent | 0/0 | 1.2 |
| AutoCES | smooth AutoCES | 1.001- | 1.004- | 1.000 | equivalent | 1/4 | 31.4 |

Ratios are geometric means of per-series MASE ratios aeon / reference (< 1: aeon more accurate). Simulated: 12 processes x 100 series, + or - where a Wilcoxon signed-rank test (Holm) finds aeon or the reference more accurate; every simulated ratio is equivalent within 5% (TOST). M4: all 100,000 series; TOST verdict overall, and the number of the six frequencies where the signed-rank test (Holm) favours aeon / the reference. Speed-up: reference time / aeon time at length 100, geometric mean of medians over the 12 processes (100 series each), single-threaded.
