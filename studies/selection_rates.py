"""How often AutoARIMA selects the true order of simulated ARIMA processes.

Works with any aeon. Prints, per process, the fraction of 100 series (length
500) for which ``AutoARIMA(max_p=3, max_d=2, max_q=2)`` selects the generating
order, and the most frequent selections.

    python studies/selection_rates.py <label>
"""

import sys
from collections import Counter

import aeon
from aeon.forecasting.stats import AutoARIMA

from horizon.simulate import ALL_PROCESSES, TRUE_MODELS

label = sys.argv[1] if len(sys.argv) > 1 else ""
print(label, "aeon", aeon.__version__, "from", aeon.__file__)
for name in ["arma_11", "ar_2", "ma_2", "white_noise", "arima_110"]:
    counts = Counter()
    for r in range(100):
        y = ALL_PROCESSES[name].sample(500, 1000 + r)
        f = AutoARIMA(max_p=3, max_d=2, max_q=2).fit(y)
        counts[f"ARIMA({f.p_},{f.d_},{f.q_})"] += 1
    rate = counts[TRUE_MODELS[name]] / 100
    print(f"  {name:12s} true-model rate {rate:.2f}  top: {dict(counts.most_common(3))}")
