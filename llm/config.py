import os
from pathlib import Path

REPO_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

# Text extraction
url = 'https://gosafir.com/mag/wp-content/uploads/2019/12/Tolkien-J.-The-lord-of-the-rings-HarperCollins-ebooks-2010.pdf'
header_height = 60  # Main text distance from the top of the page: to remove header
footer_height = 540 # Remove footer
start_page = 45
end_page = 1055
extraction_path = REPO_DIR / "llm/data/extracted_text.json"
