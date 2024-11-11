from book_utils import epub_to_csv
from book_kg import book_analysis
from nlp_utils import count_cooccurrence, count_cooccurrence2, text_analysis
import pandas as pd
from ingest import importer
import spacy

def test_book_analysis():
    # epub_to_csv(csv_file="./resources/historia_europa.csv", epub_file="./resources/historia_europa.epub")
    # cooccurences_df = pd.read_csv("./cooccurences.csv")

    nlp = spacy.load('pt_core_news_sm')
    book = pd.read_csv('./resources/historia_europa.csv')
    book = book.iloc[2:3]
    print(book)
    analysed_text_df = book_analysis(book=book, excluded_words=["c", "el", "in", "i", "or", "di"], model=nlp)
    print(analysed_text_df)
    # cooccurences_df = pd.concat(analysed_text_df["cooccurrences"].values)
    # cooccurences_df.to_csv("cooccurences.csv", index = None)
    # cooccurences_df[['first', 'second']] = cooccurences_df['key'].str.split('_', expand=True)
    # importer.import_book(cooccurences_df, book_title="História da Europa")
