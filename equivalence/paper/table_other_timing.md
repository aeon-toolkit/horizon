| aeon | reference | aeon ms n=100 | ref. ms n=100 | speed-up n=100 | aeon ms n=1000 | ref. ms n=1000 | speed-up n=1000 | aeon ms n=10000 | ref. ms n=10000 | speed-up n=10000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Theta | statsforecast Theta | 0.31 | 0.59 | 1.9 | 0.63 | 13.50 | 21.3 | 3.88 | 1866.95 | 481.3 |
| DOTM | statsforecast DOTM | 0.50 | 1.49 | 3.0 | 4.95 | 38.44 | 7.8 | 81.11 | 5597.54 | 69.0 |
| AutoCES | statsforecast AutoCES | 0.63 | 0.72 | 1.2 | 1.47 | 3.38 | 2.3 | 12.39 | 37.12 | 3.0 |
| AutoCES | smooth AutoCES | 0.63 | 19.75 | 31.4 | 1.47 | 83.33 | 56.8 | 12.22 | 800.63 | 65.5 |

Milliseconds to fit and forecast one series, single-threaded: geometric mean over the 12 horizon processes of the median over 100 series at length 100, 20 series at length 1000, 20 series at length 10000; speed-up is reference time / aeon time.
