from itertools import combinations
import re
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import concurrent.futures
from time import gmtime, strftime

stop_words = set(stopwords.words('portuguese'))
currency_pattern = re.compile(r"^[1-9][0-9,]*(\.[0-9]{1,2})?$")
letter_point_pattern = re.compile(r'^(?!^[a-zA-Z]+\.$).*$')

def preprocess_token(token):
    return token.lemma_.strip().lower()

def is_token_allowed(token, excluded_words = []):
    return bool(
        token
        and str(token).strip()
        and str(token).strip().lower() != "r$"
        and str(token).strip().lower() != "us$"
        and str(token).strip().lower() != "\n"
        and str(token).strip().lower() != "\n "
        and str(token).strip().lower() != "$"
        and str(token).strip().lower() != "+"
        and not currency_pattern.match(str(token).strip())
        and letter_point_pattern.match(str(token).strip())
        and not token.is_stop
        and not str(token).strip() in excluded_words
        and not token.is_punct
        and token not in stop_words
    )

def ner_filter(doc, excluded_words = []):
    tokens = []
    for token in doc:
        if token.ent_type_ != "" and is_token_allowed(token, excluded_words):
            tokens.append(token)
    return tokens

def ner_count(doc, excluded_words = [], filter = True):
    ners = []
    tokens = ner_filter(doc, excluded_words) if filter else doc
    for token in tokens:
        ners.append({"token": str(token), "entity_type":token.ent_type_})

    entity_counts = Counter()
    token_counts = defaultdict(Counter)
    for n in ners:
        entity_type = n['entity_type']
        token = n['token']

        # Increment the count for the entity_type
        entity_counts[entity_type] += 1

        # Increment the count for the token within its entity_type
        token_counts[entity_type][token] += 1

    # Convert defaultdict to regular dict for better printing
    token_counts = dict(token_counts)

    return {
        'entity_counts': dict(entity_counts),
        'token_counts': token_counts
    }

def count_cooccurrence(doc, counted_occurences=None, excluded_words = [], filter=True):
    if counted_occurences is None or "token_counts" not in counted_occurences:
        counted_occurences = ner_count(doc, excluded_words, filter)
    counted_occurences = counted_occurences["token_counts"]
    counters = counted_occurences.values()
    merged_counter = Counter()
    for counter in counters:
        merged_counter.update(counter)

    threshold = 1
    filtered_counter = {key: value for key, value in merged_counter.items() if value > threshold}

    if "PER" not in counted_occurences:
        return None
    per_keys = counted_occurences["PER"].keys()
    cmb = list(combinations(filtered_counter.keys(), 2))
    combos = [tup for tup in cmb if tup[0] in per_keys or tup[1] in per_keys]
    print(f"combos length: {len(combos)}")
    paragraphs = str(doc).split("\n \n")
    sentences = list(doc.sents)

    def calculate_cooccurrence(word_pair):
        word1, word2 = word_pair
        summation_key = f"{word1}_{word2}"
        global_summation = (filtered_counter[word1] + filtered_counter[word2]) / 2
        sk = {"global": global_summation, "paragraph": 0, "sentence": 0, "total_importance": 0}

        for paragraph in paragraphs:
            if word1 in paragraph and word2 in paragraph:
                sk["paragraph"] += 1

        for sentence in sentences:
            if word1 in sentence.text and word2 in sentence.text:
                sk["sentence"] += 1

        # sk["total_importance"] = sk["global"] + (sk["paragraph"] * 2) + (sk["sentence"] * 3)
        sk["total_importance"] = (sk["paragraph"]) + (sk["sentence"] * 2)
        return summation_key, sk

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(calculate_cooccurrence, combos))

    summation = dict(results)
    return dict(sorted(summation.items(), key=lambda item: item[1]['total_importance'], reverse=True))

def text_analysis(chapter, doc, model, excluded_words = [], filter = True):
    print(f"cap {chapter}", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    if model:
        doc = model(doc)
    counted_occurences = ner_count(doc, excluded_words, filter)
    counted_cooccurrences = count_cooccurrence(doc, counted_occurences, excluded_words, filter)
    return counted_occurences, counted_cooccurrences