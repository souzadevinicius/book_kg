from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
from pathlib import Path
import csv

_blacklist = ['[document]', 'noscript', 'header', 'html', 'meta', 'head', 'input', 'script']
def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    text = soup.find_all(text=True)
    for t in text:
        if t.parent.name not in _blacklist:
            output += '{} '.format(t)
    return output.strip()


def epub_to_csv(epub_file: Path, csv_file: Path):
    book = epub.read_epub(epub_file)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            chapters.append(item.get_content())

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Chapter Index', 'Chapter Number', 'Chapter Content'])  # Write header
        
        for i, chapter in enumerate(chapters, 1):
            text = chap2text(chapter)
            text_lines = text.split('\n')
            
            if len(text_lines) >= 3:  # Ensure we have at least 3 lines
                chapter_name = ' '.join(text_lines[:1]).strip()  # First two lines for chapter name
                chapter_content = '\n'.join(text_lines[1:]).strip()  # Rest for content
                
                # Only write non-empty chapters
                if chapter_name and chapter_content:
                    writer.writerow([i, chapter_name, chapter_content])