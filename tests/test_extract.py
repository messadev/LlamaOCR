"""Fast sanity net for extract.py, meant to run against whatever scans currently sit in the
project root. Real scans are gitignored (never committed as business data), so this doesn't use
fixed fixtures — it discovers *.pdf files at runtime and checks structural invariants that should
hold regardless of scan content: no footer/signature text leaking into an item row, no negative
sums, no fabricated "Итого" row. This won't catch every OCR misread (raw character accuracy on a
noisy scan has a hard ceiling), but it catches the systematic parsing bugs regressions would
reintroduce, in seconds instead of eyeballing CSVs by hand.

Usage: place one or more delivery-note PDFs in the project root, then run
    .venv/Scripts/python.exe -m pytest tests/test_extract.py -v
or plain:
    .venv/Scripts/python.exe tests/test_extract.py
"""
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import extract as ex

FOOTER_WORDS = ["Всего отпущено", "Уполномоченное лицо", "Главный бухгалтер", "Итого"]


def discover_pdfs():
    return sorted(glob.glob(os.path.join(_ROOT, "*.pdf")))


def check_document(path):
    """Returns a list of failure strings (empty if the document passes)."""
    failures = []
    rows = ex.extract_document(path)
    for row in rows:
        name = row["Наименование, характеристика"]
        nomen = row["Номенклатурный номер"]
        for marker in FOOTER_WORDS:
            if marker.lower() in name.lower() or marker.lower() in nomen.lower():
                failures.append(
                    f"row {row['№']}: footer/total text leaked into item fields "
                    f"(matched {marker!r}): name={name!r} nomen={nomen!r}"
                )
        sum_val = row["Сумма без НДС"]
        if isinstance(sum_val, (int, float)) and sum_val < 0:
            failures.append(f"row {row['№']}: negative Сумма без НДС ({sum_val})")
    return failures


def test_no_footer_bleed_or_negative_sums():
    pdfs = discover_pdfs()
    if not pdfs:
        import pytest
        pytest.skip("no *.pdf files in project root to test against")

    all_failures = {}
    for path in pdfs:
        failures = check_document(path)
        if failures:
            all_failures[os.path.basename(path)] = failures

    assert not all_failures, "\n" + "\n".join(
        f"{name}:\n  " + "\n  ".join(fails) for name, fails in all_failures.items()
    )


if __name__ == "__main__":
    pdfs = discover_pdfs()
    if not pdfs:
        print("No *.pdf files found in project root — nothing to check.")
        sys.exit(0)

    exit_code = 0
    for path in pdfs:
        failures = check_document(path)
        status = "FAIL" if failures else "ok"
        print(f"[{status}] {os.path.basename(path)}")
        for f in failures:
            print(f"    {f}")
            exit_code = 1
    sys.exit(exit_code)
