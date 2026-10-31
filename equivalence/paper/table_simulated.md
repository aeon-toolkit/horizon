
**auto mode**

| toolkit | AutoARIMA n=100 | AutoARIMA n=500 | AutoETS n=100 | AutoETS n=500 |
|---|---|---|---|---|
| statsforecast | 0.992 (0.983-1.001) | 0.995 (0.991-1.000) | 0.914 (0.889-0.940)* | 0.936 (0.918-0.955)* |
| sktime | 0.998 (0.984-1.013) | 0.994 (0.987-1.000) | 0.981 (0.966-0.997) | 0.999 (0.993-1.005) |
| skforecast | 0.996 (0.986-1.006) | 0.998 (0.992-1.003) | 1.009 (0.996-1.023) | 0.999 (0.994-1.004) |
| pmdarima | 0.999 (0.984-1.014) | 0.994 (0.988-1.001) | - | - |
| statsmodels | 0.991 (0.981-1.001) | 0.995 (0.989-1.001) | 0.981 (0.966-0.997) | 0.999 (0.993-1.005) |
| smooth | 0.987 (0.969-1.005) | 0.995 (0.988-1.002) | 0.995 (0.977-1.012) | 1.001 (0.995-1.007) |

**fixed mode**

| toolkit | AutoARIMA n=100 | AutoARIMA n=500 | AutoETS n=100 | AutoETS n=500 |
|---|---|---|---|---|
| statsforecast | 1.000 (0.999-1.002) | 1.000 (0.999-1.000) | 0.918 (0.896-0.940)* | 0.959 (0.944-0.974) |
| sktime | 1.000 (0.999-1.002) | 1.000 (1.000-1.000) | 0.947 (0.930-0.965)* | 1.000 (0.996-1.004) |
| skforecast | 1.000 (0.999-1.002) | 1.000 (1.000-1.000) | 1.003 (0.992-1.014) | 0.999 (0.995-1.003) |
| pmdarima | 1.000 (0.999-1.002) | 1.000 (1.000-1.000) | - | - |
| statsmodels | 1.000 (0.999-1.002) | 1.000 (1.000-1.000) | 0.947 (0.930-0.965)* | 1.000 (0.996-1.004) |
| smooth | 1.000 (0.999-1.001) | 1.000 (1.000-1.000) | 0.988 (0.975-1.000)* | 1.000 (0.997-1.003) |

Geometric mean of per-series MASE ratios aeon / toolkit over 600 series (6 processes x 100), 90% interval; < 1 means aeon is more accurate. * Wilcoxon signed-rank p < 0.05 after Holm correction (all such cases have aeon more accurate). TOST with a ±5% margin: 38 of 44 equivalent, 2 aeon better, 4 inconclusive (interval below 1, extending past the margin), 0 aeon worse.
