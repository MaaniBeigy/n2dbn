#!/usr/bin/env python3
"""
Download the PDFs of every paper cited in benchmarks.bib.

Run this on your own machine, on the TU/e VPN. It cannot run in the Cowork
sandbox, which has no outbound network access to publisher domains.

    cd "C:\\Users\\20240700\\OneDrive - TU Eindhoven\\Phd_Maani\\Projects\\n2dbn\\paper"
    python fetch_pdfs.py

Behavior
    Writes into ./pdfs/<bibkey>.pdf
    Skips anything already downloaded, so it is safe to re-run.
    Tries several candidate URLs per paper, in order, and stops at the first
    one that returns a real PDF.
    Waits between requests. Publishers throttle or block IP ranges that look
    like scrapers, and a suspended university subscription is a bad trade for
    a few saved minutes. Do not lower DELAY.
    Writes fetch_report.md listing what came down and what did not.

Options
    --open-only   only the sources that are free to anyone (arXiv, JMLR, PMLR,
                  NeurIPS, PLOS, BMC, MDPI, PMC). No publisher requests at all.
    --delay N     seconds between requests, default 3.
    --list        print the manifest and exit, download nothing.
"""

import argparse
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("This script needs `requests`. Install it with: pip install requests")


# ---------------------------------------------------------------------------
# Manifest
#
# Each entry is (bibkey, tier, [candidate urls in priority order]).
#
# tier "open"  the source is free to anyone, no subscription involved
# tier "pub"   publisher-hosted, needs the VPN or an institutional login
#
# Helpers below expand the short forms:
#   arxiv:XXXX.XXXXX          -> https://arxiv.org/pdf/XXXX.XXXXX
#   jmlr:vol/stem             -> https://www.jmlr.org/papers/volumeVOL/STEM/STEM.pdf
#   pmlr:vNNN/stem            -> https://proceedings.mlr.press/vNNN/stem/stem.pdf
#   neurips:YEAR/hash         -> proceedings.neurips.cc paper file url
#   springer:DOI              -> https://link.springer.com/content/pdf/DOI.pdf
#   plos:DOI                  -> PLOS printable pdf
#   doi:DOI                   -> https://doi.org/DOI  (last resort, via VPN)
# ---------------------------------------------------------------------------

MANIFEST = [
    # --- non-stationary and time-varying DBNs, classical -------------------
    ("robinson2008nsdbn",            "open", ["neurips:2008/f5f8590cd58a54e94377e6ae2eded4d9"]),
    ("robinson2010nsdbn",            "open", ["jmlr:11/robinson10a"]),
    ("jia2010flexiblelag",           "open", ["springer:10.1186/1471-2105-11-S6-S27"]),
    ("grzegorczyk2008allocation",    "pub",  ["doi:10.1093/bioinformatics/btn367"]),
    ("lebre2010artiva",              "open", ["springer:10.1186/1752-0509-4-130"]),
    ("grzegorczyk2009nscontinuous",  "open", ["neurips:2009/0ff39bbbf981ac0151d340c9aa40e63e",
                                              "http://eprints.gla.ac.uk/69618/1/69618.pdf"]),
    ("grzegorczyk2011nhdbncontinuous", "pub", ["springer:10.1007/s10994-010-5230-7",
                                              "doi:10.1007/s10994-010-5230-7"]),
    ("grzegorczyk2011dynprog",       "pub",  ["doi:10.1093/bioinformatics/btq711"]),
    ("grzegorczyk2012globalcoupling", "open", ["pmlr:v22/grzegorzyk12"]),   # PMLR misspells the name
    ("grzegorczyk2013hierarchical",  "pub",  ["springer:10.1007/s10994-012-5326-3",
                                              "doi:10.1007/s10994-012-5326-3"]),
    ("dondelinger2013edison",        "pub",  ["springer:10.1007/s10994-012-5311-x",
                                              "doi:10.1007/s10994-012-5311-x"]),
    ("kamalabad2020edgewise",        "open", ["pmc:PMC7703764", "doi:10.1093/bioinformatics/btz690"]),
    ("zhang2023improvedmcmc",        "open", ["springer:10.1186/s12859-023-05381-2"]),
    ("talih2005structural",          "pub",  ["doi:10.1111/j.1467-9868.2005.00504.x"]),
    ("xuan2007changingdep",          "open", ["https://www.cs.ubc.ca/~murphyk/Papers/xuanICML07.pdf",
                                              "doi:10.1145/1273496.1273629"]),
    ("fearnhead2006changepoint",     "pub",  ["springer:10.1007/s11222-006-8450-8",
                                              "doi:10.1007/s11222-006-8450-8"]),
    ("song2009keller",               "open", ["pmc:PMC2687946", "doi:10.1093/bioinformatics/btp192"]),
    ("song2009tvdbn",                "open", ["neurips:2009/a67f096809415ca1c9f112d96d27689b"]),
    ("ahmed2009tesla",               "open", ["https://www.cs.cmu.edu/~epxing/papers/2009/ahmed_xing_pnas09.pdf",
                                              "doi:10.1073/pnas.0901910106"]),
    ("kolar2010estimating",          "open", ["arxiv:0812.5087",
                                              "https://www.cs.cmu.edu/~epxing/papers/2010/kolar_song_xing_aoas10.pdf"]),

    # --- modern non-stationary and temporal causal discovery ---------------
    ("huang2020cdnod",               "open", ["jmlr:21/19-232", "arxiv:1903.01672"]),
    ("saggioro2020rpcmci",           "open", ["https://elib.dlr.de/139171/1/Saggioro_Chaos2020.pdf",
                                              "arxiv:2007.00267"]),
    ("mameche2025spacetime",         "open", ["arxiv:2501.10235",
                                              "https://eda.rg.cispa.io/pubs/2025/spacetime-mameche,cornanguer,ninad,vreeken.pdf"]),
    ("rahmani2025castor",            "open", ["arxiv:2311.01412", "pmlr:v258/rahmani25a"]),
    ("rahmani2025fantom",            "open", ["arxiv:2506.17065"]),
    ("gong2023rhino",                "open", ["arxiv:2210.14706"]),
    ("cheng2023cuts",                "open", ["arxiv:2302.07458"]),
    ("cheng2024cutsplus",            "open", ["arxiv:2305.05890",
                                              "https://ojs.aaai.org/index.php/AAAI/article/view/29034"]),
    ("lowe2022amortized",            "open", ["pmlr:v177/lowe22a", "arxiv:2006.10833"]),
    ("gao2025changepoint",           "open", ["pmlr:v258/gao25g", "arxiv:2407.07290"]),
    ("balsellsrodas2024markovswitching", "open", ["arxiv:2406.17698"]),
    ("hulsman2026flowmsm",           "open", ["arxiv:2606.02231"]),
    ("yi2026causalcompass",          "open", ["arxiv:2602.07915"]),

    # --- stationary DBN and time-series baselines --------------------------
    ("pamfil2020dynotears",          "open", ["pmlr:v108/pamfil20a", "arxiv:2002.00498"]),
    ("runge2019pcmci",               "open", ["https://centaur.reading.ac.uk/89762/1/sciadv.aau4996.pdf",
                                              "doi:10.1126/sciadv.aau4996"]),
    ("runge2020pcmciplus",           "open", ["pmlr:v124/runge20a", "arxiv:2003.03685"]),
    ("gerhardus2020lpcmci",          "open", ["neurips:2020/94e70705efae423efda1088614128d0b"]),
    ("tank2022neuralgc",             "open", ["https://par.nsf.gov/servlets/purl/10490277",
                                              "arxiv:1802.05842"]),
    ("nauta2019tcdf",                "open", ["https://www.mdpi.com/2504-4990/1/1/19/pdf",
                                              "doi:10.3390/make1010019"]),
    ("hyvarinen2010varlingam",       "open", ["jmlr:11/hyvarinen10a"]),
    ("malinsky2018tsfci",            "open", ["pmlr:v92/malinsky18a"]),
    ("yaman2025dbnscores",           "open", ["plos:10.1371/journal.pone.0336250"]),
    ("trabelsi2013benchmarking",     "pub",  ["doi:10.1109/ICMSAO.2013.6552549"]),
    ("trabelsi2013dmmhc",            "pub",  ["springer:10.1007/978-3-642-41398-8_34",
                                              "doi:10.1007/978-3-642-41398-8_34"]),
    ("ouyang2023divideconquer",      "open", ["arxiv:2312.01739"]),

    # --- benchmark platforms and dataset suites ----------------------------
    ("ferdous2025timegraph",         "open", ["arxiv:2506.01361", "doi:10.1145/3711896.3737439"]),
    ("gamella2025causalchambers",    "open", ["arxiv:2404.11341", "doi:10.1038/s42256-024-00964-x"]),
    ("stein2025causalrivers",        "open", ["arxiv:2503.17452",
                                              "https://openreview.net/pdf?id=wmV4cIbgl6"]),
    ("cheng2024causaltime",          "open", ["arxiv:2310.01753"]),
    ("herdeanu2025causaldynamics",   "open", ["arxiv:2505.16620"]),
    ("menegozzo2022cipcad",          "open", ["arxiv:2208.01529"]),
    ("rieth2017tep",                 "open", ["doi:10.7910/DVN/6C3JR1"]),
    ("downs1993tep",                 "pub",  ["doi:10.1016/0098-1354(93)80018-I"]),
    ("smith2011netsim",              "pub",  ["doi:10.1016/j.neuroimage.2010.08.063"]),
    ("runge2019causeme",             "open", ["https://www.nature.com/articles/s41467-019-10105-3.pdf",
                                              "doi:10.1038/s41467-019-10105-3"]),
    ("chevalley2025causalbench",     "open", ["https://www.nature.com/articles/s42003-025-07764-y.pdf",
                                              "doi:10.1038/s42003-025-07764-y"]),

    # --- systems biology benchmarks and simulators -------------------------
    ("prill2010dream3",              "open", ["plos:10.1371/journal.pone.0009202"]),
    ("marbach2010revealing",         "open", ["pmc:PMC2851957", "doi:10.1073/pnas.0913357107"]),
    ("marbach2009generating",        "pub",  ["doi:10.1089/cmb.2008.09TT"]),
    ("marbach2012dream5",            "open", ["https://dream.broadinstitute.org/papers/Marbach2012.pdf",
                                              "doi:10.1038/nmeth.2016"]),
    ("schaffter2011gnw",             "open", ["pmc:PMC3150036", "doi:10.1093/bioinformatics/btr373"]),
    ("vandenbulcke2006syntren",      "open", ["springer:10.1186/1471-2105-7-43"]),
    ("cantone2009irma",              "pub",  ["doi:10.1016/j.cell.2009.01.055"]),
    ("pratapa2020beeline",           "open", ["pmc:PMC7098036", "doi:10.1038/s41592-019-0690-6"]),
    ("dibaeinia2020sergio",          "open", ["pmc:PMC7530147",
                                              "https://www.biorxiv.org/content/10.1101/716811v1.full.pdf"]),
    ("hill2016hpndream",             "open", ["pmc:PMC4854847", "doi:10.1038/nmeth.3773"]),

    # --- real datasets used in the DBN literature --------------------------
    ("arbeitman2002drosophila",      "pub",  ["doi:10.1126/science.1072152"]),
    ("edwards2006arabidopsis",       "open", ["pmc:PMC1383640", "doi:10.1105/tpc.105.038315"]),
    ("mockler2007diurnal",           "pub",  ["doi:10.1101/sqb.2007.72.006"]),
    ("spellman1998yeast",            "open", ["pmc:PMC25624", "doi:10.1091/mbc.9.12.3273"]),
    ("rangel2004tcell",              "open", ["https://cse.buffalo.edu/faculty/mbeal/papers/bioinfbeal04.pdf",
                                              "doi:10.1093/bioinformatics/bth093"]),
    ("sachs2005flowcytometry",       "pub",  ["doi:10.1126/science.1105809"]),
    ("zhao2006mdl",                  "pub",  ["doi:10.1093/bioinformatics/btl364"]),
    ("guo2007rewiring",              "open", ["http://www.cs.cmu.edu/~epxing/papers/ICML2007_net.pdf",
                                              "doi:10.1145/1273496.1273537"]),
    ("teixeira2023yeastract",        "open", ["pmc:PMC9825609", "doi:10.1093/nar/gkac1041"]),

    # --- software, toolkits and repositories -------------------------------
    ("scutari2010bnlearn",           "open", ["https://www.jstatsoft.org/index.php/jss/article/view/v035i03/v35i03.pdf",
                                              "doi:10.18637/jss.v035.i03"]),
    ("leonelli2025bnrep",            "open", ["arxiv:2409.19158", "doi:10.1016/j.neucom.2025.129502"]),
    ("rios2025benchpress",           "open", ["arxiv:2107.03863",
                                              "https://www.jstatsoft.org/index.php/jss/article/view/v114i12/v114i12.pdf"]),
    ("franzin2017bnstruct",          "open", ["doi:10.1093/bioinformatics/btw807"]),
    ("ankan2024pgmpy",               "open", ["jmlr:25/23-0487"]),
    ("zheng2024causallearn",         "open", ["jmlr:25/23-0970", "arxiv:2307.16405"]),
    ("ramsey2023pytetrad",           "open", ["arxiv:2301.08644"]),
    ("zhang2021gcastle",             "open", ["arxiv:2111.15155"]),
    ("kalainathan2020cdt",           "open", ["jmlr:21/19-187", "arxiv:1903.02278"]),
    ("murphy2001bnt",                "open", ["https://www.cs.ubc.ca/~murphyk/Papers/bnt.pdf"]),

    # --- evaluation metrics and known pitfalls -----------------------------
    ("tsamardinos2006mmhc",          "open", ["https://pages.mtu.edu/~lebrown/supplements/mmhc_paper/paper_online.pdf",
                                              "springer:10.1007/s10994-006-6889-7"]),
    ("peters2015sid",                "open", ["arxiv:1306.1043",
                                              "https://people.math.ethz.ch/~peterbu/Files/Manuscripts/SID-final.pdf"]),
    ("davis2006prroc",               "open", ["https://ftp.cs.wisc.edu/machine-learning/shavlik-group/davis.icml06.pdf",
                                              "doi:10.1145/1143844.1143874"]),
    ("schwarz1978bic",               "open", ["https://projecteuclid.org/journals/annals-of-statistics/volume-6/issue-2/Estimating-the-Dimension-of-a-Model/10.1214/aos/1176344136.pdf",
                                              "doi:10.1214/aos/1176344136"]),
    ("heckerman1995bde",             "open", ["arxiv:1302.6815", "springer:10.1023/A:1022623210503"]),
    ("reisach2021varsortability",    "open", ["neurips:2021/e987eff4a7c7b7e580d659feb6f60c1a",
                                              "arxiv:2102.13647"]),
    ("reisach2023r2sortability",     "open", ["neurips:2023/027e86facfe7c1ea52ca1fca7bc1402b",
                                              "arxiv:2303.18211"]),
    ("runge2018pitfalls",            "open", ["arxiv:1710.05609", "doi:10.1063/1.5025050"]),
    ("constantinou2021noisy",        "open", ["arxiv:2005.09020", "doi:10.1016/j.ijar.2021.01.001"]),
    ("constantinou2019bsf",          "open", ["arxiv:1905.12666"]),
    ("gentzel2019interventional",    "open", ["neurips:2019/a87c11b9100c608b7f8e98cfa316ff7b",
                                              "arxiv:1910.05387"]),
    ("faller2024selfcompatibility",  "open", ["pmlr:v238/faller24a", "arxiv:2307.09552"]),
    ("dejongh2009distances",         "open", ["https://sites.pitt.edu/~druzdzel/ftp/iis09.pdf"]),
    ("cheng2022evaluation",          "open", ["arxiv:2202.02896", "doi:10.1109/TAI.2022.3150264"]),
    ("zheng2018notears",             "open", ["neurips:2018/e347c51419ffb23ca3fd5050202f9c3d",
                                              "arxiv:1803.01422"]),

    # --- surveys and reviews ------------------------------------------------
    ("shiguihara2021dbnsurvey",      "open", ["https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9514854",
                                              "doi:10.1109/ACCESS.2021.3105520"]),
    ("kungurtsev2024dbnfoundations", "open", ["arxiv:2406.17585"]),
    ("mihajlovic2001dbnstate",       "open", ["https://research.utwente.nl/files/27679465/0000006a.pdf"]),
    ("kitson2023bnsurvey",           "open", ["springer:10.1007/s10462-022-10351-w", "arxiv:2109.11415"]),
    ("scanagatta2019bnsurvey",       "open", ["https://faculty.ist.psu.edu/vhonavar/Courses/causality/bn-rev19.pdf",
                                              "springer:10.1007/s13748-019-00194-y"]),
    ("vowels2022dags",               "open", ["arxiv:2103.02582",
                                              "https://personalpages.surrey.ac.uk/r.bowden/publications/2022/Vowels_ACM2022pp.pdf"]),
    ("assaad2022tssurvey",           "open", ["https://www.jair.org/index.php/jair/article/download/13428/26913",
                                              "doi:10.1613/jair.1.13428"]),
    ("hasan2023cdsurvey",            "open", ["arxiv:2303.15027", "https://openreview.net/pdf?id=YdMrdhGx9y"]),
    ("gong2025temporalsurvey",       "open", ["arxiv:2303.10112", "doi:10.1145/3705297"]),
    ("nogueira2022cdtools",          "pub",  ["doi:10.1002/widm.1449"]),
    ("shojaie2022granger",           "open", ["arxiv:2105.02675",
                                              "doi:10.1146/annurev-statistics-040120-010930"]),
    ("runge2023causalinference",     "pub",  ["doi:10.1038/s43017-023-00431-y"]),

    # --- applied-domain datasets -------------------------------------------
    ("cook2013casas",                "open", ["pmc:PMC4130117", "doi:10.1109/MC.2012.328"]),
    ("chavarriaga2013opportunity",   "pub",  ["doi:10.1016/j.patrec.2012.12.014"]),
    ("wang2014studentlife",          "open", ["https://studentlife.cs.dartmouth.edu/studentlife.pdf",
                                              "doi:10.1145/2632048.2632054"]),
    ("yfantidou2022lifesnaps",       "open", ["https://www.nature.com/articles/s41597-022-01764-x.pdf",
                                              "doi:10.1038/s41597-022-01764-x"]),
    ("klasnja2019heartsteps",        "open", ["pmc:PMC6546936", "doi:10.1093/abm/kay067"]),
    ("faltys2021hirid",              "pub",  ["doi:10.1038/s41591-020-0789-4"]),
    ("johnson2023mimiciv",           "open", ["https://www.nature.com/articles/s41597-022-01899-x.pdf",
                                              "doi:10.1038/s41597-022-01899-x"]),
    ("reyna2020sepsischallenge",     "open", ["pmc:PMC6964870", "doi:10.1097/CCM.0000000000004145"]),
    ("xu2022globem",                 "open", ["https://papers.nips.cc/paper_files/paper/2022/file/9c7e8a0821dfcb58a9a83cbd37cc8131-Paper-Datasets_and_Benchmarks.pdf"]),
    ("bachlin2010daphnet",           "pub",  ["doi:10.1109/TITB.2009.2036165"]),
    ("candanedo2016occupancy",       "pub",  ["doi:10.1016/j.enbuild.2015.11.071"]),
    ("guo2019astgcn",                "open", ["https://ojs.aaai.org/index.php/AAAI/article/view/3881/3759",
                                              "doi:10.1609/aaai.v33i01.3301922"]),
    ("li2018dcrnn",                  "open", ["arxiv:1707.01926"]),
    ("makridakis2022m5",             "pub",  ["doi:10.1016/j.ijforecast.2021.11.013"]),
    ("athanasopoulos2009tourism",    "open", ["https://robjhyndman.com/papers/hierarchical.pdf",
                                              "doi:10.1016/j.ijforecast.2008.07.004"]),
    ("wickramasuriya2019mint",       "open", ["https://robjhyndman.com/papers/mint.pdf",
                                              "doi:10.1080/01621459.2018.1448825"]),
    ("miller2020bdg2",               "open", ["https://www.nature.com/articles/s41597-020-00712-x.pdf",
                                              "doi:10.1038/s41597-020-00712-x"]),
    ("benzeev2017crosscheck",        "open", ["pmc:PMC5605409", "doi:10.1037/prj0000243"]),
    ("du2016rmtpp",                  "open", ["https://www.cs.jhu.edu/~mdredze/teaching/cs650_2017/readings/du2016recurrent.pdf",
                                              "doi:10.1145/2939672.2939875"]),

    # --- foundations --------------------------------------------------------
    ("murphy2002dbnthesis",          "open", ["https://www.cs.ubc.ca/~murphyk/Thesis/thesis.pdf"]),
    ("friedman1998dbnstructure",     "open", ["https://arxiv.org/pdf/1301.7374"]),
    ("chickering2002ges",            "open", ["jmlr:3/chickering02b"]),
    ("granger1969causality",         "pub",  ["doi:10.2307/1912791"]),
    ("chowliu1968",                  "pub",  ["doi:10.1109/TIT.1968.1054142"]),
    ("dean1989persistence",          "pub",  ["doi:10.1111/j.1467-8640.1989.tb00324.x"]),
]


# ---------------------------------------------------------------------------
# URL expansion
# ---------------------------------------------------------------------------

def expand(short):
    """Return a list of candidate URLs for one short form, best first."""
    if short.startswith("arxiv:"):
        i = short[6:]
        return [f"https://arxiv.org/pdf/{i}", f"https://arxiv.org/pdf/{i}v1"]
    if short.startswith("jmlr:"):
        vol, stem = short[5:].split("/", 1)
        return [f"https://www.jmlr.org/papers/volume{vol}/{stem}/{stem}.pdf"]
    if short.startswith("pmlr:"):
        vol, stem = short[5:].split("/", 1)
        # PMLR moved recent volumes onto the mlresearch GitHub assets path.
        # Older volumes are still served from proceedings.mlr.press.
        return [f"https://proceedings.mlr.press/{vol}/{stem}/{stem}.pdf",
                f"https://raw.githubusercontent.com/mlresearch/{vol}/main/assets/{stem}/{stem}.pdf",
                f"http://proceedings.mlr.press/{vol}/{stem}/{stem}.pdf"]
    if short.startswith("neurips:"):
        year, h = short[8:].split("/", 1)
        return [f"https://proceedings.neurips.cc/paper_files/paper/{year}/file/{h}-Paper.pdf",
                f"https://proceedings.neurips.cc/paper/{year}/file/{h}-Paper.pdf",
                f"https://proceedings.neurips.cc/paper_files/paper/{year}/file/{h}-Paper-Conference.pdf"]
    if short.startswith("springer:"):
        return [f"https://link.springer.com/content/pdf/{short[9:]}.pdf"]
    if short.startswith("plos:"):
        doi = short[5:]
        journal = "plosone" if "pone" in doi else "plosbiology"
        return [f"https://journals.plos.org/{journal}/article/file?id={doi}&type=printable"]
    if short.startswith("pmc:"):
        return [f"https://pmc.ncbi.nlm.nih.gov/articles/{short[4:]}/pdf/",
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{short[4:]}/pdf/"]
    if short.startswith("doi:"):
        return [f"https://doi.org/{short[4:]}"]
    return [short]


HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# A NeurIPS alternate path, since older years use a different directory layout
NEURIPS_ALT = re.compile(r"/paper_files/paper/(\d{4})/file/")


def looks_like_pdf(resp):
    if resp.status_code != 200:
        return False
    ctype = resp.headers.get("Content-Type", "").lower()
    head = resp.content[:5]
    return head == b"%PDF-" or ("pdf" in ctype and len(resp.content) > 20000)


def try_url(session, url, timeout=45):
    try:
        r = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if looks_like_pdf(r):
        return r.content, None
    return None, f"HTTP {r.status_code}, {r.headers.get('Content-Type', '?')}, {len(r.content)} bytes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-only", action="store_true",
                    help="skip publisher-hosted sources entirely")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="seconds between requests, default 3")
    ap.add_argument("--outdir", default="pdfs")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    args = ap.parse_args()

    if args.list:
        for key, tier, cands in MANIFEST:
            print(f"{tier:5s} {key:34s} {expand(cands[0])[0]}")
        print(f"\n{len(MANIFEST)} papers, "
              f"{sum(1 for _, t, _ in MANIFEST if t == 'open')} open access, "
              f"{sum(1 for _, t, _ in MANIFEST if t == 'pub')} publisher-hosted")
        return

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    session = requests.Session()

    got, skipped, failed = [], [], []
    todo = [m for m in MANIFEST if not (args.open_only and m[1] == "pub")]
    print(f"{len(todo)} papers to try, writing into {out.resolve()}\n")

    for n, (key, tier, cands) in enumerate(todo, 1):
        dest = out / f"{key}.pdf"
        if dest.exists() and dest.stat().st_size > 10000:
            skipped.append(key)
            print(f"[{n:3d}/{len(todo)}] {key:34s} already there")
            continue

        reasons = []
        urls = [u for cand in cands for u in expand(cand)
                if not (args.open_only and cand.startswith("doi:"))]
        for url in urls:
            blob, why = try_url(session, url)
            time.sleep(args.delay)
            if blob:
                dest.write_bytes(blob)
                got.append((key, url, len(blob)))
                print(f"[{n:3d}/{len(todo)}] {key:34s} OK  {len(blob)//1024:5d} KB  {url}")
                break
            reasons.append(f"{url}: {why}")
        else:
            failed.append((key, tier, reasons))
            print(f"[{n:3d}/{len(todo)}] {key:34s} FAILED")

    # ---------------------------------------------------------------- report
    lines = ["# PDF fetch report", ""]
    lines += [f"Downloaded {len(got)}, already present {len(skipped)}, "
              f"failed {len(failed)}, out of {len(todo)} tried.", ""]
    if got:
        lines += ["## Downloaded", ""]
        lines += [f"- `{k}.pdf` ({b//1024} KB) from {u}" for k, u, b in got] + [""]
    if failed:
        lines += ["## Failed", "",
                  "Open each of these by DOI in a browser on the VPN and save it "
                  "into this folder as `<bibkey>.pdf`.", ""]
        for k, tier, reasons in failed:
            lines.append(f"- **{k}** ({tier})")
            lines += [f"    - {r}" for r in reasons]
        lines.append("")
    Path("fetch_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDownloaded {len(got)}, already present {len(skipped)}, failed {len(failed)}.")
    print("Details in fetch_report.md")


if __name__ == "__main__":
    main()
