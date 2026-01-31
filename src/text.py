import fitz
from typing import List, Dict, Any


def extract_pdf_text_by_page(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []
    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages
