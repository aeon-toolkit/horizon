| frequency | aeon AutoETS | R ets | ratio (90% CI) | TOST | aeon AutoARIMA | R auto.arima | ratio (90% CI) | TOST |
|---|---|---|---|---|---|---|---|---|
| yearly | 0.905 | 0.903 | 0.988 (0.984-0.993) | equivalent | 0.918 | 0.892 | 1.029 (1.025-1.033) | equivalent |
| quarterly | 0.892 | 0.890 | 1.010 (1.006-1.014) | equivalent | 0.894 | 0.898 | 1.012 (1.008-1.016) | equivalent |
| monthly | 0.929 | 0.914 | 1.038 (1.035-1.041) | equivalent | 0.918 | 0.903 | 1.047 (1.044-1.050) | inconclusive |
| weekly | 0.938 | 0.931 | 1.015 (1.001-1.029) | equivalent | 0.902 | 0.932 | 0.980 (0.954-1.006) | equivalent |
| daily | 1.001 | 0.996 | 1.003 (1.000-1.005) | equivalent | 1.012 | 1.044 | 0.979 (0.973-0.986) | equivalent |
| hourly | 0.788 | 0.852 | 0.912 (0.866-0.962) | inconclusive | 0.692 | 0.577 | 1.614 (1.494-1.743) | aeon worse |
| all | 0.914 | 0.908 | 1.018 (1.015-1.020) | equivalent | 0.915 | 0.903 | 1.033 (1.031-1.035) | equivalent |

OWA relative to Naive2 on the M4 test sets; ratio is the geometric mean of per-series MASE ratios aeon / published, 90% interval; TOST margin ±5%. Non-seasonal aeon AutoARIMA is fitted to M4-deseasonalised series; aeon has no seasonal ARIMA, hence hourly.
