from itertools import combinations, product, permutations
import re
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import concurrent.futures
from time import gmtime, strftime
from enum import Enum
import multiprocessing
from collections import Counter
from functools import partial
from itertools import chain

stop_words = set(stopwords.words('portuguese'))
currency_pattern = re.compile(r"^[1-9][0-9,]*(\.[0-9]{1,2})?$")
letter_point_pattern = re.compile(r'^(?!^[a-zA-Z]+\.$).*$')

class EntityType(Enum):
    LOC = "LOC"
    MISC = "MISC"
    ORG = "ORG"
    PER = "PER"

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
        and not str(token).strip().lower() in excluded_words
        and not token.is_punct
        and token not in stop_words
    )

def sentence_filter(doc, excluded_words = []):
    new_sents = []
    for sent in doc.sents:
        parsed_sent = []
        for token in sent:
            if token.ent_type_ != "" and is_token_allowed(token, excluded_words):
                parsed_sent.append(str(token))
        if len(parsed_sent) > 0:
            new_sents.append(parsed_sent)
    return new_sents




def ner_filter(doc, entity_types = [EntityType.PER, EntityType.LOC], excluded_words = []):
    tokens = []
    entity_types_names = [e.name for e in entity_types]
    for token in doc:
        if token.ent_type_ in entity_types_names and is_token_allowed(token, excluded_words):
            tokens.append(token)
    return tokens

def ner_parse(doc, excluded_words, filter, structure_key = None, structure_index = None):
    ners = []
    tokens = ner_filter(doc, excluded_words=excluded_words) if filter else doc
    for token in tokens:
        obj = {"token": str(token), "entity_type":token.ent_type_}
        if structure_key and structure_index:
            obj[structure_key] = structure_index
        ners.append(obj)
    return ners


def ner_count(doc, excluded_words = [], filter = True):
    ners = ner_parse(doc, excluded_words, filter)

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

def generate_unique_token_entity_combinations(objects, level, index):
    object_combinations = list(combinations(objects, 2))
    
    unique_combinations = []
    for obj1, obj2 in object_combinations:
        if (obj1["token"] == obj2["token"]) and (obj1["entity_type"] == obj2["entity_type"]):
            continue
        combination = (
            obj1["token"], obj1["entity_type"],
            obj2["token"], obj2["entity_type"],
            level, index
        )
        unique_combinations.append(combination)
    
    return unique_combinations


def process_paragraph(model, excluded_words, pi, p):
    doc = model(p)
    paragraph_ners = ner_parse(doc, excluded_words=excluded_words, filter=True)
    paragraph_combinations = generate_unique_token_entity_combinations(paragraph_ners, "paragraph", pi)
    
    sentence_combinations = []
    for si, s in enumerate(doc.sents):
        sentence_ners = ner_parse(s, excluded_words=excluded_words, filter=True)
        sentence_combinations.extend(generate_unique_token_entity_combinations(sentence_ners, "sentence", si))
    
    return paragraph_combinations + sentence_combinations

def count_cooccurrence2(text, model, entity_types=[EntityType.PER, EntityType.LOC], excluded_words=[], threshold=1):
    text = text.lower()
    paragraphs = text.split("\n \n")
    
    # Create a partial function with fixed arguments
    process_func = partial(process_paragraph, model, excluded_words)
    
    # Use multiprocessing to process paragraphs in parallel
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_func, enumerate(paragraphs))
    
    # Flatten the results and count occurrences
    combination_counter = Counter(chain.from_iterable(results))
    
    # Convert the counter to a list of dictionaries
    res = [
        {
            "first_token": c[0],
            "first_entity": c[1],
            "second_token": c[2],
            "second_entity": c[3],
            "level": c[4],
            "index": c[5],
            "count": count,
            "importance": count if c[4] == 'paragraph' else (count) * 2,
        }
        for c, count in combination_counter.items()
    ]
    
    return res





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

def text_analysis(chapter, text, model, excluded_words = [], filter = True):
    print(f"cap {chapter}", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    counted_occurences = count_cooccurrence2(text=text, model=model, excluded_words=excluded_words)
    # counted_occurences = ner_count(text, model, excluded_words, filter)
    # counted_cooccurrences = count_cooccurrence(doc, counted_occurences, excluded_words, filter)
    # counted_cooccurrences = count_cooccurrence2(doc, text, filter=filter, excluded_words=excluded_words)
    return counted_occurences
    return counted_occurences, counted_cooccurrences