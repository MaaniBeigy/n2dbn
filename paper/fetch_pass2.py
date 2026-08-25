#!/usr/bin/env python3
"""
Second pass. Retries the papers fetch_pdfs.py could not get, using mirrors
that do not sit behind a captcha or a bot filter.

What changed against pass 1
    Europe PMC replaces pmc.ncbi.nlm.nih.gov, which serves a reCAPTCHA to
    anything that is not an interactive browser.
    BioMed Central articles go through the journal's own /counter/pdf/ path
    rather than link.springer.com, which rate-limits.
    Several author-hosted and repository copies replace publisher landing pages.
    Longer default delay, because pass 1 tripped Springer's rate limit.

    python3 fetch_pass2.py --delay 4
"""

import argparse, time
from pathlib import Path
import requests

# (bibkey, [urls in priority order])
RETRY = [
    # --- Europe PMC, the OA full-text mirror without the captcha -----------
    ("jia2010flexiblelag",      ["epmc:PMC3026374",
                                 "bmc:bmcbioinformatics/10.1186/1471-2105-11-S6-S27"]),
    ("lebre2010artiva",         ["epmc:PMC2955603",
                                 "bmc:bmcsystbiol/10.1186/1752-0509-4-130"]),
    ("kamalabad2020edgewise",   ["epmc:PMC7703764"]),
    ("song2009keller",          ["epmc:PMC2687946"]),
    ("zhang2023improvedmcmc",   ["bmc:bmcbioinformatics/10.1186/s12859-023-05381-2",
                                 "epmc:PMC10360340"]),
    ("vandenbulcke2006syntren", ["bmc:bmcbioinformatics/10.1186/1471-2105-7-43",
                                 "epmc:PMC1435933"]),
    ("marbach2010revealing",    ["epmc:PMC2851957"]),
    ("schaffter2011gnw",        ["epmc:PMC3150036"]),
    ("pratapa2020beeline",      ["epmc:PMC7098036"]),
    ("hill2016hpndream",        ["epmc:PMC4854847"]),
    ("edwards2006arabidopsis",  ["epmc:PMC1383640"]),
    ("spellman1998yeast",       ["epmc:PMC25624"]),
    ("teixeira2023yeastract",   ["epmc:PMC9825609"]),
    ("franzin2017bnstruct",     ["epmc:PMC5408797"]),
    ("cook2013casas",           ["epmc:PMC4130117"]),
    ("klasnja2019heartsteps",   ["epmc:PMC6546936"]),
    ("reyna2020sepsischallenge",["epmc:PMC6964870"]),
    ("benzeev2017crosscheck",   ["epmc:PMC5605409"]),
    ("runge2019pcmci",          ["epmc:PMC6884412",
                                 "https://www.science.org/doi/pdf/10.1126/sciadv.aau4996"]),

    # --- repository and author-hosted copies -------------------------------
    ("assaad2022tssurvey",      ["https://hal.science/hal-03765355/document",
                                 "https://www.jair.org/index.php/jair/article/download/13428/26918",
                                 "https://www.jair.org/index.php/jair/article/download/13428/26917"]),
    ("mihajlovic2001dbnstate",  ["https://ris.utwente.nl/ws/portalfiles/portal/27679465/0000006a.pdf",
                                 "https://research.utwente.nl/files/27679465/0000006a.pdf"]),
    ("wang2014studentlife",     ["https://studentlife.cs.dartmouth.edu/StudentLife.pdf",
                                 "https://web.cs.dartmouth.edu/~campbell/papers/studentlife.pdf"]),
    ("athanasopoulos2009tourism",["https://robjhyndman.com/papers/hierarchical.pdf",
                                 "https://robjhyndman.com/papers/Hierarchical.pdf",
                                 "https://robjhyndman.com/papers/hts.pdf"]),
    ("du2016rmtpp",             ["https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf",
                                 "https://dunan.github.io/RMTPP.pdf"]),
    ("nauta2019tcdf",           ["https://www.mdpi.com/2504-4990/1/1/19/pdf?version=1550135619",
                                 "https://research.utwente.nl/files/135872756/make_01_00019.pdf",
                                 "https://www.mdpi.com/2504-4990/1/1/19/pdf"]),
    ("shiguihara2021dbnsurvey", ["https://ieeexplore.ieee.org/iel7/6287639/9312710/09514854.pdf",
                                 "https://doi.org/10.1109/ACCESS.2021.3105520"]),

    # --- Springer journals, retried slowly ---------------------------------
    ("grzegorczyk2011nhdbncontinuous", ["springer:10.1007/s10994-010-5230-7"]),
    ("grzegorczyk2013hierarchical",    ["springer:10.1007/s10994-012-5326-3"]),
    ("dondelinger2013edison",          ["springer:10.1007/s10994-012-5311-x"]),
    ("fearnhead2006changepoint",       ["springer:10.1007/s11222-006-8450-8"]),
    ("trabelsi2013dmmhc",              ["springer:10.1007/978-3-642-41398-8_34"]),
    ("kitson2023bnsurvey_alt",         ["springer:10.1007/s10462-022-10351-w"]),

    # --- Oxford Bioinformatics, free after embargo -------------------------
    ("grzegorczyk2008allocation", ["oup:bioinformatics/10.1093/bioinformatics/btn367"]),
    ("grzegorczyk2011dynprog",    ["oup:bioinformatics/10.1093/bioinformatics/btq711"]),
    ("zhao2006mdl",               ["oup:bioinformatics/10.1093/bioinformatics/btl364"]),
]


def expand(s):
    if s.startswith("epmc:"):
        a = s[5:]
        return [f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={a}&blobtype=pdf",
                f"https://europepmc.org/articles/{a}?pdf=render"]
    if s.startswith("bmc:"):
        journal, doi = s[4:].split("/", 1)
        return [f"https://{journal}.biomedcentral.com/counter/pdf/{doi}.pdf"]
    if s.startswith("springer:"):
        return [f"https://link.springer.com/content/pdf/{s[9:]}.pdf"]
    if s.startswith("oup:"):
        j, doi = s[4:].split("/", 1)
        return [f"https://academic.oup.com/{j}/article-pdf/doi/{doi}",
                f"https://doi.org/{doi}"]
    return [s]


H = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
     "Accept": "application/pdf,*/*;q=0.8",
     "Accept-Language": "en-US,en;q=0.9"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--outdir", default="pdfs")
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(exist_ok=True)
    s = requests.Session()
    got, failed = [], []

    for i, (key, cands) in enumerate(RETRY, 1):
        name = key.replace("_alt", "")
        dest = out / f"{name}.pdf"
        if dest.exists() and dest.stat().st_size > 10000:
            print(f"[{i:2d}/{len(RETRY)}] {name:32s} already there"); continue
        why = []
        for url in [u for c in cands for u in expand(c)]:
            try:
                r = s.get(url, headers=H, timeout=60, allow_redirects=True)
                ok = r.status_code == 200 and (r.content[:5] == b"%PDF-" or
                     ("pdf" in r.headers.get("Content-Type", "").lower() and len(r.content) > 20000))
            except Exception as e:
                ok = False; r = None; why.append(f"{url}: {type(e).__name__}")
            time.sleep(a.delay)
            if ok:
                dest.write_bytes(r.content); got.append(name)
                print(f"[{i:2d}/{len(RETRY)}] {name:32s} OK  {len(r.content)//1024:5d} KB  {url}")
                break
            if r is not None:
                why.append(f"{url}: HTTP {r.status_code} {r.headers.get('Content-Type','?')}")
        else:
            failed.append((name, why)); print(f"[{i:2d}/{len(RETRY)}] {name:32s} FAILED")

    Path("fetch_report_pass2.md").write_text(
        "# Pass 2 report\n\nRecovered {}, still failing {}.\n\n## Still failing\n\n".format(
            len(got), len(failed))
        + "\n".join(f"- **{k}**\n" + "\n".join(f"    - {r}" for r in w) for k, w in failed),
        encoding="utf-8")
    print(f"\nRecovered {len(got)}, still failing {len(failed)}. See fetch_report_pass2.md")


if __name__ == "__main__":
    main()
