# Reference verification (2026-10-27)

Every entry in `refs.bib` was checked against Crossref: title words, first
author and year for the search matches, or by DOI lookup (title, authors,
journal, volume, pages) where the search found a working paper or preprint
first. BibTeX for the 26 Crossref entries was fetched from doi.org by DOI
(`check_refs.py`, output `refs_crossref.bib`), then cleaned for LaTeX.

| key | DOI | how checked |
|---|---|---|
| pesaran07window | 10.1016/j.jeconom.2006.03.010 | search match |
| pesaran11averaging | 10.1198/jbes.2010.09018 | search match |
| inoue17rolling | 10.1016/j.jeconom.2016.03.006 | search match |
| clark09combining | 10.1111/j.1468-2354.2009.00533.x | DOI (search found the 2008 St Louis Fed WP) |
| tian14breaks | 10.1016/j.ijforecast.2013.06.003 | search match |
| pesaran13optimal | 10.1016/j.jeconom.2013.04.002 | DOI (search found the SSRN WP) |
| giraitis13adaptive | 10.1016/j.jeconom.2013.04.003 | DOI (search found the SSRN WP) |
| rossi12robust | 10.1080/07350015.2012.693850 | DOI (search found the SSRN WP) |
| jungmittag16air | 10.1002/for.2400 | search match |
| assenmacher08swiss | 10.1177/0027950108089681 | search match |
| petersen26windows | 10.1111/bmsp.70018 | DOI (search found the PsyArXiv preprint); full text read |
| jose08robust | 10.1016/j.ijforecast.2007.06.001 | search match |
| smith09puzzle | 10.1111/j.1468-0084.2008.00541.x | search match |
| stock04combination | 10.1002/for.928 | search match |
| claeskens16puzzle | 10.1016/j.ijforecast.2015.12.005 | DOI (search found the SSRN WP) |
| kolassa11akaike | 10.1016/j.ijforecast.2010.04.006 | search match |
| bergmeir16bagging | 10.1016/j.ijforecast.2015.07.002 | search match |
| petropoulos18bagging | 10.1016/j.ejor.2018.01.045 | search match |
| bifet07adwin | 10.1137/1.9781611972771.42 | search match |
| cerqueira17arbitrated | 10.1007/978-3-319-71246-8_29 | search match |
| makridakis20m4 | 10.1016/j.ijforecast.2019.04.014 | search match |
| hyndman08auto | 10.18637/jss.v027.i03 | search match |
| hyndman02state | 10.1016/s0169-2070(01)00110-8 | search match |
| tashman00review | 10.1016/s0169-2070(00)00065-0 | search match |
| cerqueira22size | 10.1007/s10844-022-00713-9 | DOI; journal version (2022) of arXiv 1909.13316 (2019) |
| banwell26forecasting | 10.1007/978-3-032-15535-1_10 | DOI (entry from the aeon paper's forecasting.bib) |
| hyndman08book | 10.1007/978-3-540-71918-2 | DOI lookup by hand (Crossref title lacks the subtitle) |
| hyndman21fpp3 | - | otexts.com/fpp3, section 13.7 title checked on the page |
| godahewa21monash | - | from the aeon paper's bib; NeurIPS 2021 Datasets and Benchmarks (no DOI) |
| aeon24jmlr | - | from the aeon paper's bib; JMLR 25(289) (no DOI) |

Not verified beyond metadata: the content claims made about each paper come
from abstracts and secondary summaries, except Petersen et al. (full text)
and FPP3 section 13.7 (page read).
