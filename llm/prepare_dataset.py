from typing import Dict
import requests
import io
import logging
import time
import json
from pathlib import Path
from collections import defaultdict

import pdfplumber
from pdfplumber.page import Page

import config


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def get_all_cropped_text(url: str) -> None:
    cropped_pages: Dict[int, str] = defaultdict()
    response = requests.get(url)
    content = io.BytesIO(response.content)
    pdf = pdfplumber.open(content)
    for page in pdf.pages:
        if page.page_number >= config.start_page and page.page_number <= config.end_page:
            timestamp = time.time()
            cropped_page = extract_cropped_page(
                page=page,
                header_height=config.header_height,
                footer_height=config.footer_height
            )
            cropped_pages[page.page_number] = cropped_page.extract_text()
            LOGGER.info(f'It took {time.time() - timestamp} seconds to extract page {page.page_number}.')
    LOGGER.info(f"{len(cropped_pages)} pages were extracted from the pdf.")
    to_json(cropped_pages=cropped_pages, path=config.extraction_path)


def extract_cropped_page(page: Page, header_height: int, footer_height: int):
    bbox = (0, header_height, page.width, footer_height) # Top-left corner, bottom-right corner
    cropped_page = page.crop(bbox=bbox)
    return cropped_page


def to_json(cropped_pages: Dict[int, str], path: Path):
    with open(path, 'w') as f:
        json.dump(cropped_pages, f, indent=4)


if __name__ == "__main__":
    cropped_pages = get_all_cropped_text(url=config.url)
    for page, text in cropped_pages.items():
        print(text)
        break