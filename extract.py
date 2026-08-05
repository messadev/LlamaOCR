import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pytesseract import Output

_DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_DEFAULT_TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_TESSERACT_CMD

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESSDATA_DIR = os.path.join(_HERE, "tessdata")
_OCR_LANG = "rus+eng"
_TESSDATA_ARG = f"--tessdata-dir {_TESSDATA_DIR}" if os.path.isdir(_TESSDATA_DIR) else ""
_OCR_CONFIG = f"{_TESSDATA_ARG} --psm 6"

# "Единица измерения" is always a short Cyrillic abbreviation (шт, кг, л, компл...), never
# mixed with Latin text. Requesting "rus+eng" for it (like the rest of the item table) invites
# exactly the kind of Latin-lookalike misread this column suffers from ("шт" -> "WT"/"LUT").
# OCR'ing it separately with Cyrillic only removes that ambiguity at the source instead of
# only patching known misreads after the fact. (A character whitelist would narrow this
# further, but passing non-ASCII whitelist values through to the tesseract subprocess isn't
# reliable on every Windows console codepage, so this relies on the language restriction alone.)
_UNIT_COLUMN_LANG = "rus"
_ROW_NUMBER_CONFIG = f"{_TESSDATA_ARG} --psm 7"

PDF_RENDER_DPI = 600
MIN_OCR_WIDTH = 5000  # upscale images narrower than this before OCR
MIN_WORD_CONFIDENCE = 30  # drop low-confidence OCR "words" (table gridline noise etc.)
ROW_Y_TOLERANCE = 0.6  # fraction of median word height used to cluster words into rows

TARGET_COLUMNS = [
    "№", "Наименование, характеристика", "Номенклатурный номер",
    "Единица измерения", "Количество", "Цена за единицу без НДС",
    "Сумма без НДС", "Наименование поставщика", "ИИН/БИН",
    "Номер документа", "Дата документа",
]

# Anchors used to locate the item table's header row. Deliberately excludes "Номер" here:
# it's unambiguous once we're scoped to the header row (see ITEM_COLUMN_ANCHORS below), but
# it also appears in the unrelated "Номер документа" box near the top of the page, which
# would make find_band() lock onto the wrong row if included in this search list.
ITEM_HEADER_ANCHORS = ["Наименование", "Номенклатур", "Единица", "Количество", "Цена", "Сумма"]

# Item-table column anchors, in left-to-right order, for use once the header row is already
# located. "Номер" anchors the leading "№" column explicitly (otherwise its unbounded left
# edge can swallow item-name text that wraps further left than the "Наименование" anchor).
# "Сумма" matches twice (Сумма с НДС, then Сумма НДС) so it's handled separately by x-order.
ITEM_COLUMN_ANCHORS = ["Номер", "Наименование", "Номенклатур", "Единица", "Количество", "Цена"]

# Document header-block label anchors, in left-to-right column order.
BLOCK_LABEL_ANCHORS = ["отправитель", "получатель", "Ответственный", "Транспортная", "накладная"]

IIN_BIN_LABEL_RE = re.compile(r"ИИН|БИН", re.IGNORECASE)
IIN_BIN_NUMBER_RE = re.compile(r"\d{9,14}")

# "Итого" (the table total) plus phrases from the footer/signature block below it ("Всего
# отпущено...", "Уполномоченное лицо Поставщика...", "Главный бухгалтер..."). Used to (1) bound
# the item table's bottom, and (2) reject any row dominated by this text from being misparsed as
# a line item — including "Итого" itself, since whole-page vs. cropped-and-resized re-OCR passes
# can read it differently and one pass finding it cleanly doesn't guarantee the other did too.
FOOTER_MARKERS_RE = re.compile(
    r"Итого|Всего отпущено|Уполномоченное лицо|Главный бухгалтер|Отпуск.{0,10}разрешил|расшифровка",
    re.IGNORECASE,
)

# Tesseract's rus+eng model occasionally reads the short, all-caps unit abbreviation "ШТ"
# (pieces) as the similarly-shaped Latin "WT" instead of committing to the Cyrillic script.
# The unit column only ever holds a handful of short abbreviations, so known misreads are
# corrected here rather than trying to tune OCR confidence.
UNIT_OCR_FIXES = {"wt": "шт"}


def normalize_unit(text: str) -> str:
    if not text:
        return text
    lowered = text.strip().lower()
    return UNIT_OCR_FIXES.get(lowered, lowered)


def clean_nomen(text: str) -> str:
    """A long item name or its trailing unit suffix (e.g. "...18л") occasionally overflows,
    physically on the page, past the Наименование|Номенклатурный boundary line, and gets
    OCR'd into this column's cell alongside the actual code. A nomenclature number is always
    a run of 5+ digits (see nomen_has_code in parse_item_rows), so extracting just that run
    discards whatever bled in around it."""
    match = re.search(r"\d{5,}", text)
    return match.group(0) if match else text


def pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> List[Image.Image]:
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return images


def load_images(path: str) -> List[Image.Image]:
    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            return pdf_to_images(f.read())
    return [Image.open(path).convert("RGB")]


def _osd_rotate(image: Image.Image) -> int:
    try:
        osd = pytesseract.image_to_osd(image, config=_TESSDATA_ARG, output_type=Output.DICT)
    except pytesseract.TesseractError:
        return 0
    return osd.get("rotate", 0) or 0


def orient_for_ocr(image: Image.Image):
    """Pick whichever page rotation actually lets the item-table header be located, instead of
    trusting Tesseract's OSD blindly. OSD's own orientation_conf on this document type has been
    observed running single digits to low 20s even when its rotate guess happens to be right, so
    it's too unreliable a signal to apply unconditionally — a wrong guess here would silently
    degrade every downstream step (header-anchor matching, column bounds, row clustering).
    Tries OSD's guess first (cheap, usually correct), then falls back to the other three
    rotations if the header can't be found. Returns (image, words, rows) for whichever rotation
    worked, or the OSD guess with whatever it found if none locate a header."""
    osd_rotate = _osd_rotate(image)
    candidates = [osd_rotate] + [r for r in (0, 90, 180, 270) if r != osd_rotate]
    fallback = None
    for rotate in candidates:
        candidate_image = image.rotate(-rotate, expand=True) if rotate else image
        words = ocr_words(candidate_image)
        if words.empty:
            if fallback is None:
                fallback = (candidate_image, words, [])
            continue
        rows = cluster_rows(words)
        header_band, _bounds = locate_item_table(rows)
        if header_band is not None:
            return candidate_image, words, rows
        if fallback is None:
            fallback = (candidate_image, words, rows)
    return fallback if fallback is not None else (image, ocr_words(image), [])


def preprocess(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    if image.width < MIN_OCR_WIDTH:
        scale = MIN_OCR_WIDTH / image.width
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(image)


def ocr_words(image: Image.Image, lang: str = _OCR_LANG, config: str = _OCR_CONFIG) -> pd.DataFrame:
    # Output.DATAFRAME routes Tesseract's raw TSV through pandas.read_csv, which infers one
    # dtype per column from all rows at once — a crop containing only digit tokens (exactly what
    # the per-cell numeric refinement passes below deliberately isolate) gets its entire "text"
    # column silently coerced to float64, turning "00406" into 406 or "45"/"910.00" into
    # "45.0"/"910.0" that then concatenate into garbage numbers downstream. Output.DICT parses
    # each field independently instead, so "text" keeps its original string form.
    raw = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)
    data = pd.DataFrame(raw)
    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    data["conf"] = pd.to_numeric(data["conf"], errors="coerce")
    data = data[data["conf"] >= MIN_WORD_CONFIDENCE]
    data["center_y"] = data["top"] + data["height"] / 2
    data["right"] = data["left"] + data["width"]
    return data


def cluster_rows(data: pd.DataFrame) -> List[pd.DataFrame]:
    if data.empty:
        return []
    tolerance = max(data["height"].median() * ROW_Y_TOLERANCE, 5)
    ordered = data.sort_values("center_y")
    rows: List[List[int]] = []
    current: List[int] = []
    current_y: Optional[float] = None
    for idx, word in ordered.iterrows():
        if current_y is None or abs(word["center_y"] - current_y) <= tolerance:
            current.append(idx)
            ys = ordered.loc[current, "center_y"]
            current_y = ys.mean()
        else:
            rows.append(current)
            current = [idx]
            current_y = word["center_y"]
    if current:
        rows.append(current)
    return [data.loc[row_idx].sort_values("left") for row_idx in rows]


def row_text(row: pd.DataFrame) -> str:
    return " ".join(row["text"].tolist())


def find_anchor_word(rows: List[pd.DataFrame], pattern: str, start: int = 0):
    regex = re.compile(pattern, re.IGNORECASE)
    for row in rows[start:]:
        for _, word in row.iterrows():
            if regex.search(word["text"]):
                return word
    return None


def find_band(rows: List[pd.DataFrame], patterns: List[str], start: int = 0, max_gap: int = 2):
    """Return (start_row_idx, end_row_idx_exclusive) spanning the first contiguous cluster of
    rows containing any anchor. Only the first cluster is kept (not the global min/max across
    all matches) because a header keyword like "Количество" can also appear, case-insensitively,
    as an ordinary word much later in the document (e.g. in a signature-line sentence) — including
    that stray match would stretch the band across unrelated content in between."""
    matched = []
    for i, row in enumerate(rows[start:], start=start):
        text = row_text(row)
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            matched.append(i)
    if not matched:
        return None
    cluster = [matched[0]]
    for idx in matched[1:]:
        if idx - cluster[-1] <= max_gap:
            cluster.append(idx)
        else:
            break
    return cluster[0], cluster[-1] + 1


def interpolate_missing(positions: List[Optional[float]]) -> Optional[List[float]]:
    known = [(i, p) for i, p in enumerate(positions) if p is not None]
    if len(known) < 2:
        return None
    result = list(positions)
    for i in range(len(positions)):
        if result[i] is not None:
            continue
        left = max((k for k in known if k[0] < i), default=None)
        right = min((k for k in known if k[0] > i), default=None)
        if left and right:
            frac = (i - left[0]) / (right[0] - left[0])
            result[i] = left[1] + frac * (right[1] - left[1])
        elif right:
            others = [k for k in known if k[0] > i]
            spacing = (others[1][1] - others[0][1]) / (others[1][0] - others[0][0]) if len(others) >= 2 else 0
            result[i] = right[1] - spacing * (right[0] - i)
        else:
            others = [k for k in known if k[0] < i]
            spacing = (others[-1][1] - others[-2][1]) / (others[-1][0] - others[-2][0]) if len(others) >= 2 else 0
            result[i] = left[1] + spacing * (i - left[0])
    return result


def column_bounds_from_anchors(rows: List[pd.DataFrame], band, anchors: List[str]) -> Optional[List[float]]:
    """Locate each anchor's x-position within the header band. Column order is fixed for this
    document layout, so a keyword that fails to OCR cleanly (garbled text) has its column boundary
    interpolated from the anchors on either side rather than aborting the whole row."""
    band_rows = rows[band[0]:band[1]]
    positions: List[Optional[float]] = []
    for anchor in anchors:
        word = find_anchor_word(band_rows, re.escape(anchor))
        positions.append(word["left"] if word is not None else None)
    return interpolate_missing(positions)


def item_column_bounds(rows: List[pd.DataFrame], band) -> Optional[List[float]]:
    """8 item-table columns: Номер, Наименование, Номенклатур, Единица, Количество, Цена,
    Сумма-с-НДС, Сумма-НДС. "Сумма" appears twice so it's matched by x-order instead of by
    name, and any anchor that fails to OCR cleanly is interpolated from its neighbors (fixed
    column order)."""
    band_rows = rows[band[0]:band[1]]
    positions: List[Optional[float]] = []
    for anchor in ITEM_COLUMN_ANCHORS:
        word = find_anchor_word(band_rows, re.escape(anchor))
        positions.append(word["left"] if word is not None else None)

    sum_hits = sorted({
        word["left"]
        for row in band_rows
        for _, word in row.iterrows()
        if re.search(r"Сумма", word["text"], re.IGNORECASE)
    })
    positions.append(sum_hits[0] if len(sum_hits) >= 1 else None)
    positions.append(sum_hits[1] if len(sum_hits) >= 2 else None)
    return interpolate_missing(positions)


def slice_by_columns(row: pd.DataFrame, bounds: List[float]) -> List[str]:
    cells = []
    for i in range(len(bounds)):
        x_start = bounds[i]
        x_end = bounds[i + 1] if i + 1 < len(bounds) else float("inf")
        mask = (row["left"] >= x_start - 5) & (row["left"] < x_end - 5)
        cells.append(" ".join(row.loc[mask, "text"].tolist()).strip())
    # everything left of the first bound belongs to column 0
    lead_mask = row["left"] < bounds[0] - 5
    lead = " ".join(row.loc[lead_mask, "text"].tolist()).strip()
    return [lead] + cells


def parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"[\d][\d\s]*(?:[.,]\d+)?", text)
    if not match:
        return None
    cleaned = match.group(0).replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def last_number(text: str) -> Optional[float]:
    numbers = re.findall(r"\d[\d\s]*(?:[.,]\d+)?", text)
    if not numbers:
        return None
    return parse_number(numbers[-1])


def extract_header_block(rows: List[pd.DataFrame]) -> Dict[str, str]:
    result = {"sender": "", "receiver": "", "responsible": "", "transport": "", "doc_ref": ""}
    label_band = find_band(rows, [re.escape(a) for a in BLOCK_LABEL_ANCHORS])
    if label_band is None:
        return result
    bounds = column_bounds_from_anchors(rows, label_band, BLOCK_LABEL_ANCHORS)
    if bounds is None:
        return result

    item_header_band = find_band(rows, [re.escape(a) for a in ITEM_HEADER_ANCHORS], start=label_band[1])
    value_end = item_header_band[0] if item_header_band else len(rows)

    value_rows = rows[label_band[1]:value_end]
    keys = ["sender", "receiver", "responsible", "transport", "doc_ref"]
    for key in keys:
        result[key] = ""
    for row in value_rows:
        cells = slice_by_columns(row, bounds)
        for key, cell in zip(keys, cells):
            if cell:
                result[key] = (result[key] + " " + cell).strip()
    return result


def locate_item_table(rows: List[pd.DataFrame]):
    """Find the item table header and its column x-bounds. Returns (header_band, bounds) or
    (None, None) if the header row can't be located."""
    header_band = find_band(rows, [re.escape(a) for a in ITEM_HEADER_ANCHORS])
    if header_band is None:
        return None, None
    bounds = item_column_bounds(rows, header_band)
    if bounds is None:
        return None, None
    return header_band, bounds


def collect_name_overflow(cells: List[str]) -> str:
    """A long item description can overflow past the "Наименование" column's own width into
    the neighboring "№" and "Номенклатурный номер" columns. Pull that overflow text back in,
    skipping only text that looks like real data for its own column (a bare 1-3 digit item
    number, or an actual 5+ digit nomenclature code)."""
    parts = []
    left = cells[1].strip()
    if left and not re.fullmatch(r"\d{1,3}", left):
        parts.append(left)
    center = cells[2].strip()
    if center:
        parts.append(center)
    right = cells[3].strip()
    if right and not re.search(r"\d{5,}", right):
        parts.append(right)
    return " ".join(parts).strip()


def find_column_divider(
    image: Image.Image, x_start: int, x_end: int, y_start: int, y_end: int, frac: float = 0.6
) -> Optional[int]:
    """Find the x-position of a ruled vertical table border within [x_start, x_end): the first
    column whose pixels are almost entirely dark across [y_start, y_end). Needed because a
    header cell merged across sub-columns (e.g. "Количество" spanning "подлежит отпуску" and
    "отпущено") has its own OCR'd text anchored somewhere inside that merged span rather than
    at the sub-column's true left edge — the printed rule line is the only reliable boundary."""
    if x_end <= x_start or y_end <= y_start:
        return None
    band = image.crop((x_start, y_start, x_end, y_end))
    arr = np.asarray(band, dtype=np.int32)
    darkness = (255 - arr).sum(axis=0)
    threshold = 255 * (y_end - y_start) * frac
    hits = np.where(darkness >= threshold)[0]
    if len(hits) == 0:
        return None
    return x_start + int(hits[0])


def refine_row_unit_qty(image: Image.Image, row: pd.DataFrame, bounds: List[float]):
    """"Единица измерения" and "Количество" hold small text squeezed against ruled table
    lines, and OCR'ing them as part of the whole page or the whole item-table crop often
    misses them entirely — leaving that x-range empty and letting the next column's value
    (Цена) bleed into what should be Количество. Re-OCRing each one in isolation, at its own
    true column boundary (via find_column_divider — the header-anchor bounds aren't reliable
    for this, see there), recovers both far more reliably. `row` must be in the same pixel
    coordinate space as `image` (i.e. whichever image `row` was originally OCR'd from)."""
    top = int(row["top"].min())
    bottom = int((row["top"] + row["height"]).max())

    unit_search_start = max(0, int(bounds[3]) - 5)
    qty_end = max(unit_search_start + 1, int(bounds[5]) - 5)
    divider = find_column_divider(image, unit_search_start + 30, qty_end, top - 3, bottom + 3)
    if divider is None:
        divider = max(unit_search_start + 1, int(bounds[4]) - 5)

    # The "Единица" header anchor has the same merged-header imprecision as "Количество" does
    # (see find_column_divider) one column over, so back up past it too; a margin proportional
    # to image width holds up across scans rendered at different effective resolutions.
    left_margin = max(20, int(image.width * 0.009))
    unit_start = max(0, unit_search_start - left_margin)

    unit = None
    if divider > unit_start:
        y_start, y_end = max(0, top - 8), min(image.height, bottom + 8)
        strip = image.crop((unit_start, y_start, divider, y_end))
        strip = strip.resize((strip.width * 3, strip.height * 3), Image.LANCZOS)
        strip = ImageOps.autocontrast(strip)
        words = ocr_words(strip, lang=_UNIT_COLUMN_LANG, config=_OCR_CONFIG)
        if not words.empty:
            # A stray digit sometimes leaks in from the neighboring nomenclature/quantity
            # columns; a unit abbreviation is never purely numeric, so only letter-bearing
            # tokens are genuine readings of this cell.
            letter_tokens = [t for t in words["text"] if re.search(r"[а-яёА-ЯЁ]", str(t))]
            if letter_tokens:
                unit = normalize_unit(" ".join(letter_tokens))

    qty = None
    if qty_end > divider:
        y_start, y_end = max(0, top - 3), min(image.height, bottom + 3)
        strip = image.crop((divider, y_start, qty_end, y_end))
        words = ocr_words(strip, lang="eng", config=_ROW_NUMBER_CONFIG)
        if not words.empty:
            numbers = []
            for text in words.sort_values("left")["text"]:
                numbers.extend(re.findall(r"\d[\d.,]*", str(text)))
            qty = parse_number(numbers[0]) if numbers else None

    return unit, qty


def refine_row_sum_with_vat(image: Image.Image, row: pd.DataFrame, bounds: List[float]) -> Optional[float]:
    """"Сумма с НДС" is squeezed against ruled lines the same way Количество is, and on some
    scans goes entirely undetected in the whole-page/crop OCR pass, while "Цена за единицу"
    (identical to it whenever quantity is 1) and "Сумма НДС" on either side usually are
    detected — so callers should treat a value recovered here as the authoritative "Сумма с
    НДС" reading and re-derive "Сумма НДС" placement accordingly rather than trusting
    positional cell-slicing for either. Re-OCRs the column in isolation at its own
    gridline-detected boundaries (see find_column_divider)."""
    top = int(row["top"].min())
    bottom = int((row["top"] + row["height"]).max())
    y_start, y_end = top - 3, bottom + 3

    price_start = max(0, int(bounds[5]) - 5)
    search_end = min(image.width, int(bounds[7]) + int(bounds[7] - bounds[6]))
    div1 = find_column_divider(image, price_start + 30, search_end, y_start, y_end)
    if div1 is None:
        return None
    div2 = find_column_divider(image, div1 + 30, search_end, y_start, y_end)
    if div2 is None or div2 <= div1:
        return None

    strip = image.crop((div1, max(0, y_start - 8), div2, min(image.height, y_end + 8)))
    strip = strip.resize((strip.width * 3, strip.height * 3), Image.LANCZOS)
    strip = ImageOps.autocontrast(strip)
    words = ocr_words(strip, lang="eng", config=_OCR_CONFIG)
    if words.empty:
        return None
    # This crop is a single number and nothing else, so any token that isn't purely digits/
    # punctuation is noise (observed: a faint second "ghost" line of garbage text OCR'd inside
    # the same strip, interleaving by x-position with the real digits and splitting the number
    # mid-run, e.g. "15" + "AN" + "200,00" instead of "15" + "200,00"). Dropping non-numeric
    # tokens before concatenating keeps the real digit run intact.
    numeric_tokens = words[words["text"].str.fullmatch(r"[\d.,]+")]
    if numeric_tokens.empty:
        return None
    combined = "".join(str(t) for t in numeric_tokens.sort_values("left")["text"])
    return parse_number(combined)


def parse_item_rows(
    rows: List[pd.DataFrame], bounds: List[float], image: Optional[Image.Image] = None
) -> List[Dict[str, str]]:
    """Parse item rows out of `rows` (searched from the start) using pre-computed column
    x-bounds, stopping at the "Итого" total row if present. When `image` is given (the same
    image `rows` was OCR'd from), unit and quantity get a second, per-row refinement pass via
    refine_row_unit_qty — see its docstring for why that column needs special handling."""
    total_row = None
    for i, row in enumerate(rows):
        if FOOTER_MARKERS_RE.search(row_text(row)):
            total_row = i
            break
    end = total_row if total_row is not None else len(rows)

    items = []
    running_no = 0
    pending_name_lines: List[str] = []
    for i, row in enumerate(rows[:end]):
        if FOOTER_MARKERS_RE.search(row_text(row)):
            # Defense in depth: the table-bottom boundary above should already exclude this row,
            # but if it slipped through (e.g. total_row was found later via "Итого" while a
            # footer line got interleaved earlier by row-clustering), never let it be scored as
            # a possible item row.
            continue
        cells = slice_by_columns(row, bounds)
        # cells: [lead(unused), no, name, nomen, unit, qty, price, sum_vat, vat_amount]
        nomen_has_code = bool(re.search(r"\d{5,}", cells[3]))
        qty = last_number(cells[5])
        price = parse_number(cells[6])
        sum_with_vat = parse_number(cells[7])
        vat_amount = parse_number(cells[8]) if len(cells) > 8 else None
        numeric_signals = sum(x is not None for x in (qty, price, sum_with_vat, vat_amount))
        if not nomen_has_code and numeric_signals < 2:
            # not a real item row: skips the "1 2 3 4 5 6 7 8 9" column-index legend row and
            # blank/stray OCR noise, but a long item description often wraps onto its own line
            # (with no nomenclature code or numbers of its own) directly above the row that
            # carries the numeric data — stash any such text and attach it to the next real item.
            # The first couple of rows are excluded from this: when this table is parsed from
            # a padded crop (see extract_items_via_crop), the padding above the header can pull
            # in its own tail end as leading rows, and that header-label text isn't part of
            # any item's description.
            is_header_leak = i < 2 or re.search(r"Наименование", cells[2], re.IGNORECASE)
            if not is_header_leak:
                overflow = collect_name_overflow(cells)
                if overflow:
                    pending_name_lines.append(overflow)
            continue

        running_no += 1
        unit = normalize_unit(cells[4])
        if image is not None:
            refined_unit, refined_qty = refine_row_unit_qty(image, row, bounds)
            if refined_unit and FOOTER_MARKERS_RE.search(refined_unit):
                # The unit-column refinement re-OCRs at 3x zoom with autocontrast, and can read
                # "Итого"/footer text cleanly even when the whole-page pass read it as unrecog-
                # nizable noise (the case the boundary checks above rely on) — so this is the
                # last chance to catch a total/footer row that slipped through as a fake item.
                # A real item's "Единица измерения" is never one of these markers.
                running_no -= 1
                continue
            if refined_unit:
                unit = refined_unit
            if refined_qty is not None:
                qty = refined_qty

            refined_sum_with_vat = refine_row_sum_with_vat(image, row, bounds)
            if refined_sum_with_vat is not None:
                expected = price * qty if (price is not None and qty) else None
                if (
                    expected is not None
                    and sum_with_vat is not None
                    and abs(sum_with_vat - expected) <= abs(refined_sum_with_vat - expected)
                ):
                    # The refined reading's gridline-divider detection can lock onto the wrong
                    # pair of columns (e.g. "Цена" instead of "Сумма с НДС") and return a
                    # plausible-looking but wrong number. Количество × Цена is the table's own
                    # arithmetic ground truth when both are available, so use it to prefer
                    # whichever candidate — refined or the original positional reading — is
                    # actually consistent, instead of always trusting the refined one blindly.
                    refined_sum_with_vat = sum_with_vat
                if vat_amount is not None and vat_amount > refined_sum_with_vat:
                    # A VAT amount can never exceed the tax-inclusive total it's drawn from, so
                    # cells[8]'s positional reading was wrong (it likely picked up stray digits
                    # from beyond the table's right edge, which this column's slice is unbounded
                    # against — see slice_by_columns). Treat it as unread rather than let a
                    # bogus, too-large value flip sum_without_vat negative below.
                    vat_amount = None
                if vat_amount is None and sum_with_vat is not None:
                    # cells[8] (vat_amount's normal slot) was empty, which — now that a genuine
                    # "Сумма с НДС" reading exists — means what cells[7] actually held was
                    # "Сумма НДС" that fell short of reaching its own anchor bound (bounds[7]),
                    # not "Сумма с НДС" at all.
                    vat_amount = sum_with_vat
                sum_with_vat = refined_sum_with_vat
        sum_without_vat = None
        if sum_with_vat is not None and vat_amount is not None:
            sum_without_vat = sum_with_vat - vat_amount
        price_without_vat = None
        if sum_without_vat is not None and qty:
            price_without_vat = sum_without_vat / qty
        name = " ".join(pending_name_lines + [collect_name_overflow(cells)]).strip()
        pending_name_lines = []
        items.append({
            "no": str(running_no),
            "name": name,
            "nomen": clean_nomen(cells[3]),
            "unit": unit,
            "qty": qty,
            "price_without_vat": price_without_vat,
            "sum_without_vat": sum_without_vat,
        })
    return items


def merge_item_fields(primary: Dict[str, str], secondary: Dict[str, str]) -> Dict[str, str]:
    """Fill in any blank field of `primary` with `secondary`'s value for that field."""
    merged = dict(primary)
    for key, value in secondary.items():
        current = merged.get(key)
        is_blank = current is None or (isinstance(current, str) and not current.strip())
        if is_blank and value not in (None, ""):
            merged[key] = value
    return merged


def extract_items_via_crop(image: Image.Image, rows: List[pd.DataFrame]) -> List[Dict[str, str]]:
    """Whole-page OCR reliably locates the item table header and its column x-bounds, but
    Tesseract's layout analysis on the full page (competing with signatures, stamps, and other
    tables elsewhere on the page) can miss item-row text it would otherwise read cleanly.
    Cropping just the item-table band and re-OCRing it in isolation recovers that text
    reliably, so that's tried first; the whole-page rows are a fallback if the crop fails."""
    header_band, bounds = locate_item_table(rows)
    if header_band is None:
        return []

    header_bottom = int(max((r["top"] + r["height"]).max() for r in rows[header_band[0]:header_band[1]]))
    total_bottom = None
    for row in rows[header_band[1]:]:
        if FOOTER_MARKERS_RE.search(row_text(row)):
            total_bottom = int((row["top"] + row["height"]).max())
            break
    if total_bottom is None:
        total_bottom = min(header_bottom + int(image.height * 0.15), image.height)

    # Generous padding above/below matters: a crop trimmed tight to the text starves
    # Tesseract's layout analysis of the margin it needs to segment lines correctly, and
    # produces mostly garbage even though the text itself is perfectly legible.
    pad_top = int(image.height * 0.03)
    pad_bottom = int(image.height * 0.04)
    crop_box = (
        0,
        max(0, header_bottom - pad_top),
        image.width,
        min(image.height, total_bottom + pad_bottom),
    )
    crop = image.crop(crop_box)
    plain_items = []
    plain_words = ocr_words(crop)
    if not plain_words.empty:
        plain_items = parse_item_rows(cluster_rows(plain_words), bounds, image=crop)

    # Small digits sitting right against table gridlines (e.g. the qty sub-columns) can be
    # missed entirely on the plain crop — sharpening exaggerates the digit strokes enough to
    # separate them from the adjacent border lines during Tesseract's layout analysis. But the
    # same sharpening sometimes degrades longer text runs (item descriptions) that the plain
    # crop read fine, so both passes are run and merged rather than picking one over the other.
    sharpened = crop.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=2))
    sharpened = ImageOps.autocontrast(sharpened)
    sharp_items = []
    sharp_words = ocr_words(sharpened)
    if not sharp_words.empty:
        sharp_items = parse_item_rows(cluster_rows(sharp_words), bounds, image=sharpened)

    if plain_items and sharp_items and len(plain_items) == len(sharp_items):
        items = [merge_item_fields(p, s) for p, s in zip(plain_items, sharp_items)]
    elif plain_items:
        items = plain_items
    elif sharp_items:
        items = sharp_items
    else:
        items = []

    # Tesseract's layout analysis on an isolated crop can, on some scans, drop entire rows of
    # small text it reads fine as part of the whole page (its "uniform block" assumption
    # breaks down without the rest of the page for context) — a silent failure mode distinct
    # from the per-cell misreads the two passes above are built to correct. A lower item count
    # than the whole-page reading is the signal that happened, so fall back to whichever found
    # more complete rows rather than trusting the crop unconditionally.
    whole_page_items = parse_item_rows(rows[header_band[1]:], bounds, image=image)
    if len(whole_page_items) > len(items):
        items = whole_page_items

    return items


def extract_iin_bin(rows: List[pd.DataFrame]) -> str:
    """The ИИН/БИН value can OCR into its own row either just before or just after the
    "ИИН/БИН" label row (small scan misalignment splits them apart in the y-clustering), so
    search a small window of nearby rows by row-index proximity rather than raw text distance
    (which is skewed by how much unrelated text sits in between in the flattened page text)."""
    for i, row in enumerate(rows):
        if not IIN_BIN_LABEL_RE.search(row_text(row)):
            continue
        lo, hi = max(0, i - 2), min(len(rows), i + 3)
        best, best_dist = "", None
        for j in range(lo, hi):
            for m in IIN_BIN_NUMBER_RE.finditer(row_text(rows[j])):
                dist = abs(j - i)
                if best_dist is None or dist < best_dist:
                    best, best_dist = m.group(0), dist
        if best:
            return best
    return ""


def extract_doc_meta(rows: List[pd.DataFrame]) -> (str, str):
    """The "Номер документа" / "Дата составления" box near the top of the page (separate
    from the "Товарно-транспортная накладная (номер, дата)" cell, which is normally blank
    for this internal-issuance document type). Just 2 columns and no leading column before
    them, unlike the other boxes, so this splits on a single midpoint rather than reusing
    slice_by_columns (whose "everything left of the first anchor" bucket doesn't apply here)."""
    band = find_band(rows, [r"документа", r"составлени"])
    if band is None:
        return "", ""
    band_rows = rows[band[0]:band[1]]
    doc_word = find_anchor_word(band_rows, r"документа")
    date_word = find_anchor_word(band_rows, r"составлени")
    if doc_word is None or date_word is None:
        return "", ""
    midpoint = (doc_word["left"] + date_word["left"]) / 2

    doc_number, doc_date = "", ""
    for row in rows[band[1]:band[1] + 2]:
        left_text = " ".join(row.loc[row["left"] < midpoint, "text"].tolist())
        right_text = " ".join(row.loc[row["left"] >= midpoint, "text"].tolist())
        if not doc_number and re.search(r"\d{4,}", left_text):
            doc_number = left_text
        if not doc_date:
            date_match = re.search(r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}", right_text)
            if date_match:
                doc_date = date_match.group(0)
    return doc_number.strip(), doc_date.strip()


def extract_document(path: str) -> List[Dict[str, str]]:
    images = load_images(path)
    doc_rows: List[Dict[str, str]] = []
    header_block = None
    doc_number, doc_date = "", ""
    iin_bin = ""

    for image in images:
        base_image = preprocess(image)
        proc_image, words, rows = orient_for_ocr(base_image)
        if words.empty:
            continue

        if header_block is None:
            hb = extract_header_block(rows)
            if hb.get("sender"):
                header_block = hb

        if not doc_number and not doc_date:
            doc_number, doc_date = extract_doc_meta(rows)

        if not iin_bin:
            iin_bin = extract_iin_bin(rows)

        items = extract_items_via_crop(proc_image, rows)
        if items:
            doc_rows.extend(items)

    supplier = header_block.get("sender", "") if header_block else ""

    output_rows = []
    for item in doc_rows:
        output_rows.append({
            "№": item["no"],
            "Наименование, характеристика": item["name"],
            "Номенклатурный номер": item["nomen"],
            "Единица измерения": item["unit"],
            "Количество": item["qty"] if item["qty"] is not None else "",
            "Цена за единицу без НДС": (
                round(item["price_without_vat"], 2) if item["price_without_vat"] is not None else ""
            ),
            "Сумма без НДС": (
                round(item["sum_without_vat"], 2) if item["sum_without_vat"] is not None else ""
            ),
            "Наименование поставщика": supplier,
            "ИИН/БИН": iin_bin,
            "Номер документа": doc_number,
            "Дата документа": doc_date,
        })
    return output_rows


def collect_input_paths(inputs: List[str]) -> List[str]:
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.pdf"):
                paths.extend(sorted(glob.glob(os.path.join(item, ext))))
        else:
            paths.append(item)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract line-item data from scanned накладная documents into a review CSV."
    )
    parser.add_argument("inputs", nargs="+", help="Image/PDF file(s) or a folder containing them")
    parser.add_argument("-o", "--output", default="extracted_purchases.csv", help="Output CSV path")
    args = parser.parse_args()

    paths = collect_input_paths(args.inputs)
    if not paths:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for path in paths:
        print(f"Processing {path} ...")
        try:
            rows = extract_document(path)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        if not rows:
            print("  No item rows found.")
            continue
        print(f"  Found {len(rows)} item row(s).")
        for row in rows:
            missing = [k for k in ("ИИН/БИН", "Номер документа", "Дата документа") if not row[k]]
            if missing:
                print(f"    NOTE: row {row['№']} is missing: {', '.join(missing)} (fill in manually)")
        all_rows.extend(rows)

    if not all_rows:
        print("No data extracted from any input file.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows, columns=TARGET_COLUMNS)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(df)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
