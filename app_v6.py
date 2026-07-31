import os
from typing import List

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
from pytesseract import Output

# On Windows, pytesseract needs the Tesseract-OCR binary; point at the default
# install location if it isn't already on PATH.
_DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_DEFAULT_TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_TESSERACT_CMD

PDF_RENDER_DPI = 300
COLUMN_GAP_FACTOR = 2.5  # word gap > (median char width * factor) starts a new column


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


def image_to_rows(image: Image.Image) -> List[List[str]]:
    """OCR one page image and reconstruct it as table rows (list of column strings)."""
    data = pytesseract.image_to_data(image, output_type=Output.DATAFRAME)
    data = data.dropna(subset=["text"])
    data = data[data["text"].astype(str).str.strip() != ""]
    if data.empty:
        return []

    char_widths = data["width"] / data["text"].astype(str).str.len().clip(lower=1)
    gap_threshold = (char_widths.median() or 10) * COLUMN_GAP_FACTOR

    rows = []
    for _, line_words in data.groupby(["block_num", "par_num", "line_num"], sort=False):
        line_words = line_words.sort_values("left")
        columns, current_words, prev_right = [], [], None
        for _, word in line_words.iterrows():
            left = word["left"]
            if prev_right is not None and (left - prev_right) > gap_threshold:
                columns.append(" ".join(current_words))
                current_words = []
            current_words.append(str(word["text"]))
            prev_right = left + word["width"]
        if current_words:
            columns.append(" ".join(current_words))
        rows.append(columns)
    return rows


def rows_to_dataframe(rows: List[List[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    max_cols = max(len(r) for r in rows)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]
    columns = [f"Column {i + 1}" for i in range(max_cols)]
    return pd.DataFrame(padded, columns=columns)


# Streamlit Application
st.title("Scanned PDF to Table (Local OCR)")
st.markdown(
    "Extracts text from scanned PDFs with Tesseract OCR and reconstructs rows/columns "
    "from word positions. Runs on CPU only \u2014 no GPU or local vision model required."
)

with st.sidebar:
    st.markdown("#### Upload Scanned PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more scanned PDF files", type=["pdf"], accept_multiple_files=True
    )
    dpi = st.number_input(
        "Render DPI", min_value=100, max_value=600, value=PDF_RENDER_DPI, step=50,
        help="Higher DPI improves OCR accuracy on small text but is slower.",
    )

if uploaded_files:
    all_frames = []
    for uploaded_file in uploaded_files:
        st.markdown(f"#### {uploaded_file.name}")
        pages = pdf_to_images(uploaded_file.read(), dpi=dpi)

        progress_bar = st.progress(0)
        status_box = st.empty()
        file_rows = []
        for i, page_image in enumerate(pages, start=1):
            status_box.markdown(f"**Processing page {i}/{len(pages)}...**")
            file_rows.extend(image_to_rows(page_image))
            progress_bar.progress(i / len(pages))
        status_box.markdown("**Done.**")

        df = rows_to_dataframe(file_rows)
        if not df.empty:
            df.insert(0, "Source File", uploaded_file.name)
        all_frames.append(df)
        st.dataframe(df, width="stretch")

    combined_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    st.markdown("#### Combined Results")
    st.dataframe(combined_df, width="stretch")

    st.download_button(
        label="Download Combined CSV",
        data=combined_df.to_csv(index=False).encode("utf-8"),
        file_name="extracted_data.csv",
        mime="text/csv",
    )
else:
    st.markdown("Output will be displayed here after uploading scanned PDF(s).")
