import pandas as pd
import json
from pathlib import Path
from book_utils import epub_to_csv
from repositories.words import WordsNeo4JRepo
from datetime import datetime

def import_book(co_ocurrences: pd.DataFrame, book_title: str, book_pages: int = None, book_author: str = None, book_year: str = None):
    word_repo = WordsNeo4JRepo()
    word_repo.init_collection()
    # linenumber = 1
    # chunksize=10
    # try:
    #     chunks = pd.read_json(co_ocurrence_file, lines=True, chunksize=chunksize)
    #     for chunk in chunks:
    word_repo.add_words(co_ocurrences.to_dict('records'), book_title=book_title, batch_size=1000)
            # linenumber += chunksize
    # except Exception as e:
    #     print(e, linenumber)

    
