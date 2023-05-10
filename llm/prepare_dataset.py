from typing import Dict, Generator, List
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


def extract_pages_as_batches_from_url(url: str, batch_size: int) -> Generator[List[Page], None, None]:
    LOGGER.info(f'Start extracting pages from {url}')
    response = requests.get(url)
    content = io.BytesIO(response.content)
    pages = pdfplumber.open(content).pages
    LOGGER.info(f'Finished extracting pages from {url}')
    return batch_loader(pages, batch_size=batch_size)


def batch_loader(pages: List, batch_size: int) -> Generator[List[Page], None, None]:
    for i in range(0, len(pages), batch_size):
        yield pages[i:i + batch_size]


def extract(url: str, batch_size: int) -> None:
    """Extract text over selected pages

    Args:
        url (str): url of the pdf
        batch_size (int): number of pages to extract at once to avoid memory issues
    """
    batches = extract_pages_as_batches_from_url(url=url, batch_size=batch_size)
    for batch in batches:
        # We append text to the existing file with "a" mode (append)
        with open(config.extraction_path, 'a') as f:
            for page in batch:
                if page.page_number >= config.start_page and page.page_number <= config.end_page:
                    timestamp = time.time()
                    cropped_page = extract_cropped_page(
                        page=page,
                        header_height=config.header_height,
                        footer_height=config.footer_height
                    )
                    dict_page = {page.page_number: cropped_page}
                    json.dump(dict_page, f)
                    f.write('\n')
                    LOGGER.info(f'It took {time.time() - timestamp} seconds to extract page {page.page_number}.')


def extract_cropped_page(page: Page, header_height: int, footer_height: int) -> str:
    bbox = (0, header_height, page.width, footer_height) # Top-left corner, bottom-right corner
    cropped_page = page.crop(bbox=bbox)
    return cropped_page.extract_text()


if __name__ == "__main__":
    extract(url=config.url, batch_size=config.batch_size)