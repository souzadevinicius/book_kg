import pandas as pd
from nlp_utils import text_analysis
import spacy

def _flatten_occurences(ner_result):
    flattened = {}
    if isinstance(ner_result, dict):
        for entity_type, count in ner_result['entity_counts'].items():
            flattened[f'Entity_{entity_type}'] = count
        for entity_type, tokens in ner_result['token_counts'].items():
            for token, count in tokens.items():
                flattened[f'{entity_type}_{token}'] = count
    return flattened

def _parse_chapter(row: pd.Series, model, excluded_words = []):
    occurrences, cooccurences = text_analysis(row["Chapter Number"], row["Chapter Content"], model, excluded_words)
    occurrences_flatened = _flatten_occurences(occurrences)
    occurences_summary_df = pd.DataFrame([occurrences_flatened]).fillna(0)
    occurrences_overall_summary = occurences_summary_df.sum().sort_values(ascending=False)
    occurrences_overall_summary = pd.DataFrame(occurrences_overall_summary)
    occurrences_overall_summary["chapter"] = row["Chapter Number"]
    occurrences_overall_summary.reset_index(inplace=True)
    occurrences_overall_summary.rename(columns={0: "count", "index": "key"}, inplace=True)
    occurrences_overall_summary = occurrences_overall_summary[["chapter", "key", "count"]]
    if cooccurences:
        result_df = pd.DataFrame.from_dict(cooccurences, orient='index')
        result_df.reset_index(inplace=True)
        result_df["chapter"] = row["Chapter Number"]
        result_df.rename(columns={"index": "key"}, inplace=True)
        result_df = result_df[["chapter", "key", "total_importance", "global","paragraph","sentence"]]
    else:
        result_df = None

    return pd.Series({
        "occurrences": occurrences_overall_summary,
        "cooccurrences": result_df
    })

def book_analysis(book: pd.DataFrame, model: spacy.Language, excluded_words=[]) -> pd.DataFrame:
    return book.apply(lambda row: _parse_chapter(row, model, excluded_words=excluded_words), axis=1)


def _get_unique_elements(series):
    return list(set([item for sublist in series.str.split('_') for item in sublist]))

def occurences_graph(ocurrences: pd.DataFrame):
    nodes = _get_unique_elements(ocurrences['key'])
    nodes_list = [{"data": {"id": node, "label": node}} for node in nodes]

    edges_list = []
    for _, row in ocurrences.iterrows():
        source, target = row['key'].split('_')
        edges_list.append({"data": {"source": source, "target": target}})

    return {
        "nodes": nodes_list,
        "edges": edges_list
    }
