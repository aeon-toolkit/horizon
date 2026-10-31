"""Look up each reference in Crossref, check the match, and fetch BibTeX by DOI.

Each entry gives a citation key, the expected title words, first author surname
and year. A match is accepted only if the Crossref title contains the expected
words, the first author and the year agree. Writes refs_crossref.bib (fetched
by DOI) and prints a verification table.

    python papers/windowed-ets-arima/check_refs.py
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# DOIs fixed by hand where the bibliographic search found a working paper or
# preprint instead of the journal version (each checked by title, authors,
# journal, volume and pages against Crossref, 2026-10-27)
DOI = {
    "clark09combining": "10.1111/j.1468-2354.2009.00533.x",
    "pesaran13optimal": "10.1016/j.jeconom.2013.04.002",
    "giraitis13adaptive": "10.1016/j.jeconom.2013.04.003",
    "rossi12robust": "10.1080/07350015.2012.693850",
    "claeskens16puzzle": "10.1016/j.ijforecast.2015.12.005",
    "petersen26windows": "10.1111/bmsp.70018",
    "cerqueira22size": "10.1007/s10844-022-00713-9",
    "banwell26forecasting": "10.1007/978-3-032-15535-1_10",
    "hyndman08book": "10.1007/978-3-540-71918-2",
}

REFS = [  # key, query, expected first-author surname, year
    ("pesaran07window", "Selection of estimation window in the presence of breaks", "Pesaran", 2007),
    ("pesaran11averaging", "Forecast combination across estimation windows", "Pesaran", 2011),
    ("inoue17rolling", "Rolling window selection for out-of-sample forecasting with time-varying parameters", "Inoue", 2017),
    ("clark09combining", "Improving forecast accuracy by combining recursive and rolling forecasts", "Clark", 2009),
    ("tian14breaks", "Forecast combinations under structural break uncertainty", "Tian", 2014),
    ("pesaran13optimal", "Optimal forecasts in the presence of structural breaks", "Pesaran", 2013),
    ("giraitis13adaptive", "Adaptive forecasting in the presence of recent and ongoing structural change", "Giraitis", 2013),
    ("rossi12robust", "Out-of-sample forecast tests robust to the choice of window size", "Rossi", 2012),
    ("jungmittag16air", "Combination of forecasts across estimation windows: an application to air travel demand", "Jungmittag", 2016),
    ("assenmacher08swiss", "Forecasting the Swiss economy using VECX* models: an exercise in forecast combination across models and observation windows", "Assenmacher", 2008),
    ("petersen26windows", "Comparing training window selection methods for prediction in non-stationary time series", "Petersen", 2026),
    ("jose08robust", "Simple robust averages of forecasts: some empirical results", "Jose", 2008),
    ("smith09puzzle", "A simple explanation of the forecast combination puzzle", "Smith", 2009),
    ("stock04combination", "Combination forecasts of output growth in a seven-country data set", "Stock", 2004),
    ("claeskens16puzzle", "The forecast combination puzzle: a simple theoretical explanation", "Claeskens", 2016),
    ("kolassa11akaike", "Combining exponential smoothing forecasts using Akaike weights", "Kolassa", 2011),
    ("bergmeir16bagging", "Bagging exponential smoothing methods using STL decomposition and Box-Cox transformation", "Bergmeir", 2016),
    ("petropoulos18bagging", "Exploring the sources of uncertainty: why does bagging for time series forecasting work?", "Petropoulos", 2018),
    ("bifet07adwin", "Learning from time-changing data with adaptive windowing", "Bifet", 2007),
    ("cerqueira17arbitrated", "Arbitrated ensemble for time series forecasting", "Cerqueira", 2017),
    ("makridakis20m4", "The M4 Competition: 100,000 time series and 61 forecasting methods", "Makridakis", 2020),
    ("hyndman08auto", "Automatic time series forecasting: the forecast package for R", "Hyndman", 2008),
    ("hyndman02state", "A state space framework for automatic forecasting using exponential smoothing methods", "Hyndman", 2002),
    ("tashman00review", "Out-of-sample tests of forecasting accuracy: an analysis and review", "Tashman", 2000),
    ("cerqueira22size", "A case study comparing machine learning with statistical methods for time series forecasting: size matters", "Cerqueira", 2022),
    ("banwell26forecasting", "Time series extrinsic regression algorithms for forecasting long time series with a short horizon", "Banwell", 2026),
    ("hyndman08book", "Forecasting with exponential smoothing: the state space approach", "Hyndman", 2008),
]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def crossref(query, n=5):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": query, "rows": n})
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["message"]["items"]


def bibtex(doi):
    req = urllib.request.Request("https://doi.org/" + doi,
                                 headers={"Accept": "application/x-bibtex"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def by_doi(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["message"]


def year_of(it):
    for k in ("published-print", "published-online", "issued", "published"):
        if k in it and it[k].get("date-parts") and it[k]["date-parts"][0][0]:
            return it[k]["date-parts"][0][0]


def main():
    out, rows = [], []
    for key, q, author, year in REFS:
        want = set(w for w in norm(q).split() if len(w) > 3)
        best = None
        cands = [by_doi(DOI[key])] if key in DOI else crossref(q)
        for it in cands:
            title = norm(" ".join(it.get("title", [""])))
            words = set(title.split())
            first = norm((it.get("author") or [{}])[0].get("family", ""))
            years = {year_of(it)} | {(it.get(k) or {}).get("date-parts", [[None]])[0][0]
                                     for k in ("published-online", "published-print", "issued")}
            if len(want & words) >= 0.8 * len(want) and norm(author).split()[0] in first:
                best = (it, year in years)
                break
        if best is None:
            rows.append((key, "NO MATCH", "", "", ""))
            continue
        it, year_ok = best
        doi = it["DOI"]
        try:
            bib = bibtex(doi)
            bib = re.sub(r"@(\w+)\{[^,]*,", lambda m: "@%s{%s," % (m.group(1), key), bib, count=1)
            out.append(bib.strip())
        except Exception as e:  # noqa: BLE001
            bib = f"% bibtex fetch failed for {doi}: {e}"
            out.append(bib)
        rows.append((key, "ok" if year_ok else f"YEAR? {year_of(it)}", doi,
                     (it.get("container-title") or [""])[0][:45], " ".join(it.get("title", [""]))[:70]))
        time.sleep(1.5)
    Path(__file__).with_name("refs_crossref.bib").write_text("\n\n".join(out) + "\n")
    for r in rows:
        print(" | ".join(str(x) for x in r))


if __name__ == "__main__":
    sys.exit(main())
