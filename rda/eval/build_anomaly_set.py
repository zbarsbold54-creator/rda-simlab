"""
Build a synthetic anomaly-injection evaluation set for testing salience-based
recall in the nanochat RDA experiment.

This script ONLY generates the evaluation dataset. It does not build a scoring
harness, it does not touch the model, and it does not touch the training
pipeline.

Source data
-----------
The FineWeb-EDU validation shard already present on disk, i.e. the LAST parquet
file of the fineweb-edu data dir. Per `nanochat.dataset.parquets_iter_batched`
the last shard is always the validation shard, which is exactly the split
`scripts/base_eval.py` uses for the baseline BPB number. Nothing is downloaded.

Anomalies
---------
Exactly three deterministic anomaly types, assigned by strict round-robin:

  0. factual_contradiction - double (or 0 -> 1) the first numeric literal of the
     first qualifying sentence occurring before the injection boundary.
  1. topic_break          - insert a VERBATIM sentence from the lexically least
     similar other source document (lowest Jaccard over stopword-filtered terms).
  2. token_noise          - replace 3-5 whitespace-delimited words of one
     sentence with the same number of rare BPE token strings, where "rare"
     comes from an empirical token-frequency table built over the 500
     UNMODIFIED source documents. BPE merge ranks are explicitly NOT used as
     frequency information.

Everything is seeded with 42, so a rerun on the same repository / data / tokenizer
state produces byte-for-byte identical output.

Usage
-----
    python eval/build_anomaly_set.py
"""

import os
import re
import sys
import math
import json
import random
from collections import Counter

# -----------------------------------------------------------------------------#
# The one and only seed.
# -----------------------------------------------------------------------------#
SEED = 42

# The spec pins the source to the FineWeb-EDU validation shard, so the dataset
# selection is fixed here rather than inherited from the ambient environment.
os.environ["NANOCHAT_DATASET"] = "fineweb-edu"

# Make `python eval/build_anomaly_set.py` work from the repo root without any
# installation step by putting the repo root on sys.path.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUTPUT_PATH = os.path.join(REPO_ROOT, "eval", "eval_anomaly_set.jsonl")
TARGET_DOC_COUNT = 500

ANOMALY_TYPES = ["factual_contradiction", "topic_break", "token_noise"]

# A numeric literal, exactly as specified.
NUMERIC_LITERAL_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

# A sentence boundary: terminal punctuation plus any closing quotes/brackets.
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")

# A small, fixed list of standard English stopwords (topic_break similarity only).
STOPWORDS = frozenset("""
a an the and or but if while of to in on at by for with from into over under
is are was were be been being am do does did doing have has had having
it its this that these those there here as not no nor so than then too very
can will just should would could may might must shall
he she they we you i him her them us me my your his their our mine yours
ours theirs who whom whose which what when where why how all any both each
few more most other some such only own same s t don now
""".split())

# topic_break donor sentence shape.
DONOR_MIN_WORDS = 8
DONOR_MAX_WORDS = 40

# token_noise replacement shape.
NOISE_MIN_REPLACEMENTS = 3
NOISE_MAX_REPLACEMENTS = 5


# -----------------------------------------------------------------------------#
# Sentence segmentation
# -----------------------------------------------------------------------------#
def split_sentences(text):
    """Deterministically split `text` into sentence spans.

    Each returned span is exactly one sentence's content: it starts at the
    first non-whitespace character and ends at that sentence's terminator
    (including any closing quote or bracket). Inter-sentence whitespace belongs
    to no span, which is what makes a span directly usable as the reported
    `anomaly_char_start` / `anomaly_char_end` range and as a zero-based
    `anomaly_sentence_index` into the final document.
    """
    spans = []
    prev = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end()
        start = _skip_whitespace(text, prev, end)
        if end > start:
            spans.append((start, end))
        prev = end
    start = _skip_whitespace(text, prev, len(text))
    if len(text) > start:
        spans.append((start, len(text)))
    return spans


def _skip_whitespace(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    return start


def sentences_text(text, sentences):
    return [text[start:end] for start, end in sentences]


def valid_boundaries(n_sentences):
    """Valid insertion boundaries for the 40%-60% rule, honouring the
    "not the first or final sentence boundary" constraint.

    Returns None when the document has no valid boundary at all.
    """
    lower = math.ceil(0.40 * n_sentences)
    upper = math.floor(0.60 * n_sentences)
    low = max(lower, 1)                # never the first boundary
    high = min(upper, n_sentences - 1)  # never the final boundary
    if low > high:
        return None
    return list(range(low, high + 1))


def insert_at(text, sentences, boundary, anomaly_sentence):
    """Insert `anomaly_sentence` immediately BEFORE sentence `boundary`.

    Returns (new_text, char_start, char_end) where char_start is inclusive and
    char_end exclusive, both referring to the FINAL anomalous text.

    A span begins at its sentence's first real character, and the source corpus
    is not guaranteed to put whitespace between two sentences (a citation, a
    scraped page, a regex boundary that fired mid-abbreviation). We therefore
    bracket the anomaly with one space on each side so that it always becomes a
    standalone sentence at index `boundary`, which is what the zero-based
    `anomaly_sentence_index` requires. The two spaces are separators only:
    [char_start, char_end) delimits exactly the inserted sentence, and dropping
    that range plus the leading separator restores the control text byte for
    byte.
    """
    start = sentences[boundary][0]
    char_start = start + 1
    char_end = char_start + len(anomaly_sentence)
    new_text = text[:start] + " " + anomaly_sentence + " " + text[start:]

    # The anomaly must be a standalone sentence at index `boundary`, and the
    # reported offsets must delimit exactly that sentence.
    spans = split_sentences(new_text)
    assert spans[boundary] == (char_start, char_end), (spans[boundary], char_start, char_end)
    return new_text, char_start, char_end


# -----------------------------------------------------------------------------#
# Anomaly 1: factual_contradiction
# -----------------------------------------------------------------------------#
def doubled_literal(literal):
    """Deterministically contradict a numeric literal.

    abs(value) > 0 -> multiply by exactly 2;  value == 0 -> replace with 1.
    """
    value = float(literal)
    if value == 0:
        return "1"
    doubled = value * 2
    if "." in literal:
        decimals = len(literal.split(".", 1)[1])
        return f"{doubled:.{decimals}f}"
    return str(int(doubled))


def find_contradiction(text, sentences, boundary):
    """Find the first sentence BEFORE `boundary` holding a numeric literal.

    Returns a dict describing the candidate, or None when there is no
    qualifying candidate (in which case the document is not valid for this
    evaluation set).
    """
    for start, end in sentences[:boundary]:
        source = text[start:end]
        match = NUMERIC_LITERAL_RE.search(source)
        if match is None:
            continue
        original = match.group(0)
        replacement = doubled_literal(original)
        mutated = source[:match.start()] + replacement + source[match.end():]
        return {
            "source_start": start,
            "source_end": end,
            "source_sentence": source,
            "original_literal": original,
            "new_literal": replacement,
            "mutated_sentence": mutated,
        }
    return None


# -----------------------------------------------------------------------------#
# Anomaly 2: topic_break
# -----------------------------------------------------------------------------#
def lexical_terms(text):
    """Lowercase, whitespace-tokenize, drop stopwords, return a term set."""
    return {term for term in text.lower().split() if term not in STOPWORDS}


def jaccard(a, b):
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def pick_topic_break_sentence(target_terms, donors):
    """Return (donor_doc_id, donor_source_text, donor_sentence).

    `donors` holds every valid document in original source-document order. The
    donor is the lexically LEAST similar document, ties broken by original
    source order.
    """
    best = None  # (similarity, original_source_index, entry)
    for entry in donors:
        key = (jaccard(target_terms, entry["terms"]), entry["original_index"])
        if best is None or key < best[0]:
            best = (key, entry)
    if best is None:
        return None
    entry = best[1]

    # First sentence with 8..40 whitespace-delimited words and no newlines.
    for sentence in sentences_text(entry["text"], entry["sentences"]):
        if "\n" in sentence or "\r" in sentence:
            continue
        word_count = len(sentence.split())
        if DONOR_MIN_WORDS <= word_count <= DONOR_MAX_WORDS:
            return entry["doc_id"], entry["text"], sentence

    # Fallback: first non-empty sentence.
    for sentence in sentences_text(entry["text"], entry["sentences"]):
        if sentence.strip():
            return entry["doc_id"], entry["text"], sentence
    return None


# -----------------------------------------------------------------------------#
# Anomaly 3: token_noise
# -----------------------------------------------------------------------------#
def build_rare_token_pool(texts, tokenizer):
    """Empirical token-frequency table over the UNMODIFIED source documents.

    BPE merge ranks are explicitly NOT treated as frequency information. We count
    real occurrences of every non-special token, then order token types by
    (frequency ascending, token id ascending) and take the bottom 5% as the
    rare pool.
    """
    special_ids = set()
    for name in tokenizer.get_special_tokens():
        try:
            special_ids.add(tokenizer.encode_special(name))
        except Exception:
            pass
    special_ids.add(tokenizer.get_bos_token_id())

    counts = Counter()
    for ids in tokenizer.encode(list(texts)):
        for token_id in ids:
            if token_id in special_ids:
                continue
            decoded = tokenizer.id_to_token(token_id)
            # Exclude tokens that decode to nothing / to only whitespace.
            if not decoded or decoded.strip() == "":
                continue
            counts[token_id] += 1

    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    pool_size = max(1, int(len(ordered) * 0.05))
    return [token_id for token_id, _ in ordered[:pool_size]], counts


def usable_as_sentence_end(token_string):
    """A replacement token may not swallow the sentence it lands in.

    The spec excludes tokens that decode to only whitespace, and forbids
    special tokens. A token that itself contains a newline would also split the
    anomaly into two sentences, which would break the zero-based
    `anomaly_sentence_index` and the character offsets. Both are internal
    whitespace and excluded for the same reason whitespace-only tokens are.
    """
    if not token_string or token_string.strip() == "":
        return False
    if "\n" in token_string or "\r" in token_string:
        return False
    return True


def make_token_noise_sentence(plain_sentences, rare_strings, rng):
    """Replace 3-5 whitespace-delimited words with rare BPE token strings.

    Returns (mutated_sentence, replacements) or None when no eligible sentence
    with at least 3 replaceable words exists.

    The sentence's final word is never a replacement candidate. Whitespace
    splitting keeps the trailing period attached to that word, so replacing it
    would strip the sentence terminator and make the injected anomaly merge with
    the following sentence, which breaks the zero-based `anomaly_sentence_index`
    and the character-offset semantics. Everything else about the sentence is
    preserved, so exactly 3-5 words are still replaced.
    """
    eligible = []
    for sentence in plain_sentences:
        # Splitting on whitespace runs keeps every separator exactly where it
        # was, so only the chosen words change.
        chunks = re.split(r"(\s+)", sentence)
        words = [i for i, chunk in enumerate(chunks) if chunk.strip()]
        # Drop the final word so the terminator always survives.
        replaceable = words[:-1]
        if len(replaceable) >= NOISE_MIN_REPLACEMENTS:
            eligible.append((chunks, replaceable))
    if not eligible:
        return None

    chunks, words = eligible[rng.randrange(len(eligible))]
    n_replace = rng.randint(NOISE_MIN_REPLACEMENTS, NOISE_MAX_REPLACEMENTS)
    n_replace = min(n_replace, len(words), len(rare_strings))
    positions = sorted(rng.sample(range(len(words)), n_replace))
    replacements = rng.sample(list(rare_strings), n_replace)

    chosen = {words[position]: replacements[i] for i, position in enumerate(positions)}
    rebuilt = [chosen.get(i, chunk) for i, chunk in enumerate(chunks)]
    mutated = "".join(rebuilt)

    # The replacement tokens themselves must not swallow the following text, so
    # the anomaly keeps whatever terminator the original sentence ended with.
    if not mutated.rstrip().endswith((".", "!", "?", '"', "'", ")", "]", "}")):
        mutated = mutated.rstrip() + "."
    return mutated, replacements


# -----------------------------------------------------------------------------#
# Source loading
# -----------------------------------------------------------------------------#
def load_val_documents():
    """Read the FineWeb-EDU validation shard, in source order.

    The last parquet file is always the validation shard, which is the exact
    split `scripts/base_eval.py` uses for the baseline BPB number.
    """
    from nanochat.dataset import parquets_iter_batched
    texts = []
    for batch in parquets_iter_batched("val"):
        texts.extend(batch)
    return texts


# -----------------------------------------------------------------------------#
# Generation
# -----------------------------------------------------------------------------#
def screen_documents(all_texts, tokenizer, rng_boundary):
    """Deterministically sample and screen candidate source documents.

    The boundary RNG is advanced for every screened candidate in shuffled order
    (valid or not), so the boundary stream is fully deterministic regardless of
    how many candidates end up valid.
    """
    order = list(range(len(all_texts)))
    random.Random(SEED).shuffle(order)

    screened = []
    for original_index in order:
        text = all_texts[original_index]
        # (1) usable text
        if not isinstance(text, str) or not text.strip():
            continue
        # (2)/(3) deterministic sentence split, at least 3 sentences
        sentences = split_sentences(text)
        if len(sentences) < 3:
            continue
        # (4) at least one valid 40%-60% insertion boundary
        boundaries = valid_boundaries(len(sentences))
        if boundaries is None:
            continue
        boundary = boundaries[rng_boundary.randrange(len(boundaries))]
        # (5) at least one deterministic factual_contradiction candidate
        contradiction = find_contradiction(text, sentences, boundary)
        if contradiction is None:
            continue
        # (7) tokenizable by the nanochat tokenizer
        if len(tokenizer.encode(text)) == 0:
            continue
        screened.append({
            "original_index": original_index,
            "text": text,
            "sentences": sentences,
            "boundary": boundary,
            "contradiction": contradiction,
        })
        if len(screened) >= TARGET_DOC_COUNT:
            break
    return screened


def build_rows():
    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()

    print("Reading the FineWeb-EDU validation shard ...")
    all_texts = load_val_documents()
    print(f"  documents in the validation shard : {len(all_texts)}")

    rng_boundary = random.Random(SEED)
    rng_noise = random.Random(SEED)

    screened = screen_documents(all_texts, tokenizer, rng_boundary)
    num_docs = len(screened)
    print(f"  valid source documents            : {num_docs}")
    if num_docs == 0:
        raise RuntimeError("no valid source documents found in the validation shard")

    # Rare-token pool, built from the UNMODIFIED selected documents.
    rare_ids, token_counts = build_rare_token_pool([d["text"] for d in screened], tokenizer)
    rare_strings = [tokenizer.id_to_token(token_id) for token_id in rare_ids]
    # The bottom-5% ordering is already fixed by (frequency, id); we only drop
    # members that cannot be spliced in without splitting a sentence in half.
    rare_strings = [s for s in rare_strings if usable_as_sentence_end(s)]
    rare_string_set = set(rare_strings)
    print(f"  token types observed              : {len(token_counts)}")
    print(f"  rare-token pool (bottom 5%)       : {len(rare_strings)}")

    # Lexical terms for topic_break donor selection. The donor search relies on
    # `doc_id`, so stamp it on before the search can run.
    for doc_id, entry in enumerate(screened):
        entry["doc_id"] = doc_id
        entry["terms"] = lexical_terms(entry["text"])

    rows = []
    records = []
    for doc_id, entry in enumerate(screened):
        text = entry["text"]
        sentences = entry["sentences"]
        boundary = entry["boundary"]
        anomaly_type = ANOMALY_TYPES[doc_id % len(ANOMALY_TYPES)]

        meta = {
            "donor_doc_id": None,
            "donor_source_text": None,
            "replacements": None,
        }

        if anomaly_type == "factual_contradiction":
            anomaly_sentence = entry["contradiction"]["mutated_sentence"]

        elif anomaly_type == "topic_break":
            picked = pick_topic_break_sentence(entry["terms"], screened)
            if picked is None:
                raise RuntimeError(f"doc {doc_id}: no topic_break donor sentence found")
            donor_doc_id, donor_source_text, anomaly_sentence = picked
            meta["donor_doc_id"] = donor_doc_id
            meta["donor_source_text"] = donor_source_text

        else:  # token_noise
            result = make_token_noise_sentence(
                sentences_text(text, sentences), rare_strings, rng_noise
            )
            if result is None:
                raise RuntimeError(f"doc {doc_id}: no token_noise-eligible sentence found")
            anomaly_sentence, replacements = result
            meta["replacements"] = replacements

        # An anomaly sentence may carry leading/trailing whitespace (a mutated
        # token_noise sentence can begin with a newline). Normalize it here so
        # the recorded sentence, the inserted text, and the reported character
        # offsets all refer to exactly the same substring.
        anomaly_sentence = anomaly_sentence.strip()

        new_text, char_start, char_end = insert_at(
            text, sentences, boundary, anomaly_sentence
        )

        rows.append({
            "doc_id": doc_id,
            "condition": "anomalous",
            "anomaly_type": anomaly_type,
            "text": new_text,
            "anomaly_sentence_index": boundary,  # zero-based in the FINAL document
            "anomaly_char_start": char_start,
            "anomaly_char_end": char_end,
        })
        rows.append({
            "doc_id": doc_id,
            "condition": "control",
            "anomaly_type": None,
            "text": text,
            "anomaly_sentence_index": None,
            "anomaly_char_start": None,
            "anomaly_char_end": None,
        })
        records.append({
            "doc_id": doc_id,
            "anomaly_type": anomaly_type,
            "boundary": boundary,
            "anomaly_sentence": anomaly_sentence,
            "source_text": text,
            "contradiction": entry["contradiction"],
            "meta": meta,
        })

    return rows, records, rare_string_set


def serialize(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def write_output(payload):
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def load_output():
    """Reopen the written file from disk, exactly as an external consumer would."""
    rows = []
    with open(OUTPUT_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                rows.append(json.loads(line))
    return rows


# -----------------------------------------------------------------------------#
# Self-validation
# -----------------------------------------------------------------------------#
class Validator:
    def __init__(self):
        self.passed = []

    def check(self, name, condition, detail=""):
        if not condition:
            raise AssertionError(f"VALIDATION FAILED: {name} {detail}".strip())
        self.passed.append(name)


def validate(records, rare_string_set, payload):
    """Reopen the written JSONL from disk and validate every requirement."""
    validator = Validator()
    rows = load_output()  # validate what was actually persisted
    num_docs = len(records)

    anomalous = [r for r in rows if r["condition"] == "anomalous"]
    controls = [r for r in rows if r["condition"] == "control"]

    # 1. correct total number of rows
    validator.check("1. total rows", len(rows) == 2 * num_docs,
                    f"got {len(rows)}, want {2 * num_docs}")

    # 2. exactly half anomalous and half control
    validator.check("2. half anomalous and half control",
                    len(anomalous) == len(controls) == num_docs,
                    f"anomalous={len(anomalous)} control={len(controls)}")

    # 3. anomaly counts
    counts = Counter(r["anomaly_type"] for r in anomalous)
    validator.check("3. no unexpected anomaly types",
                    set(counts) <= set(ANOMALY_TYPES), f"got {sorted(counts)}")
    if num_docs == TARGET_DOC_COUNT:
        validator.check("3. factual_contradiction == 167",
                        counts["factual_contradiction"] == 167,
                        f"got {counts['factual_contradiction']}")
        validator.check("3. topic_break == 167", counts["topic_break"] == 167,
                        f"got {counts['topic_break']}")
        validator.check("3. token_noise == 166", counts["token_noise"] == 166,
                        f"got {counts['token_noise']}")

    # exact field set, no more, no less
    fields = {"doc_id", "condition", "anomaly_type", "text",
              "anomaly_sentence_index", "anomaly_char_start", "anomaly_char_end"}
    validator.check("5. exact field set on every row",
                    all(set(r) == fields for r in rows))

    # ordering: doc0 anomalous, doc0 control, doc1 anomalous, ...
    expected_order = []
    for doc_id in range(num_docs):
        expected_order.append((doc_id, "anomalous"))
        expected_order.append((doc_id, "control"))
    validator.check("5. row ordering",
                    [(r["doc_id"], r["condition"]) for r in rows] == expected_order)

    by_doc_anomalous = {r["doc_id"]: r for r in anomalous}
    by_doc_control = {r["doc_id"]: r for r in controls}

    # 4. one anomalous + one control per doc_id
    pair_counts = Counter((r["doc_id"], r["condition"]) for r in rows)
    validator.check("4. one anomalous + one control per doc_id",
                    all(pair_counts[(d, "anomalous")] == 1
                        and pair_counts[(d, "control")] == 1
                        for d in range(num_docs)))

    for record in records:
        doc_id = record["doc_id"]
        arow = by_doc_anomalous[doc_id]
        crow = by_doc_control[doc_id]
        text = arow["text"]
        start = arow["anomaly_char_start"]
        end = arow["anomaly_char_end"]
        index = arow["anomaly_sentence_index"]

        # 5. control text identical to the original source text
        validator.check(f"5. doc {doc_id} control text == source",
                        crow["text"] == record["source_text"])
        # The control must be recoverable from the anomalous row by dropping the
        # inserted sentence and the two spaces that bracket it.
        validator.check(f"5. doc {doc_id} control recoverable from anomalous",
                        text[:start - 1] + text[end + 1:] == crow["text"])

        # 6. anomalous row has non-null sentence index and character offsets
        validator.check(f"6. doc {doc_id} anomaly metadata non-null",
                        index is not None and start is not None and end is not None)

        # 7. control row has null anomaly metadata
        validator.check(f"7. doc {doc_id} control metadata null",
                        crow["anomaly_type"] is None
                        and crow["anomaly_sentence_index"] is None
                        and crow["anomaly_char_start"] is None
                        and crow["anomaly_char_end"] is None)

        # 8. anomalous character range inside the final text
        validator.check(f"8. doc {doc_id} char range inside text",
                        isinstance(start, int) and isinstance(end, int)
                        and 0 <= start <= end <= len(text))

        # 9. char_end > char_start
        validator.check(f"9. doc {doc_id} char_end > char_start", end > start)

        # the char span really is the inserted anomaly sentence
        validator.check(f"9. doc {doc_id} char span is the anomaly sentence",
                        text[start:end] == record["anomaly_sentence"])

        # 10. sentence index valid
        final_sentences = sentences_text(text, split_sentences(text))
        validator.check(f"10. doc {doc_id} sentence index in range",
                        isinstance(index, int) and 0 <= index < len(final_sentences))
        validator.check(f"10. doc {doc_id} sentence index points at the anomaly",
                        final_sentences[index].strip()
                        == record["anomaly_sentence"].strip())
        # not the first or final sentence of the final document
        validator.check(f"10. doc {doc_id} anomaly not first or final",
                        0 < index < len(final_sentences) - 1)

        # 11. round-robin assignment
        expected_type = ANOMALY_TYPES[doc_id % len(ANOMALY_TYPES)]
        validator.check(f"11. doc {doc_id} round-robin assignment",
                        arow["anomaly_type"] == expected_type == record["anomaly_type"],
                        f"got {arow['anomaly_type']} want {expected_type}")

        # 12. the contradiction really changes a numeric literal from earlier text
        if record["anomaly_type"] == "factual_contradiction":
            validator.check(f"12. doc {doc_id} contradiction changes one literal",
                            _verify_contradiction(record, final_sentences, index))

        # 13. the topic-break sentence came from a different source document
        if record["anomaly_type"] == "topic_break":
            donor_doc_id = record["meta"]["donor_doc_id"]
            donor_source = record["meta"]["donor_source_text"]
            validator.check(f"13. doc {doc_id} topic_break donor is a different doc",
                            donor_doc_id is not None and donor_doc_id != doc_id,
                            f"donor={donor_doc_id}")
            validator.check(f"13. doc {doc_id} topic_break sentence verbatim from donor",
                            record["anomaly_sentence"] in donor_source)
            validator.check(f"13. doc {doc_id} topic_break sentence unmodified",
                            record["anomaly_sentence"] in text)
            # the donor is the least similar document
            validator.check(f"13. doc {doc_id} donor is lowest Jaccard",
                            _is_lowest_similarity(record, records))

        # 14. the token-noise replacements came from the rare-token pool
        if record["anomaly_type"] == "token_noise":
            replacements = record["meta"]["replacements"]
            validator.check(f"14. doc {doc_id} 3-5 replacements from rare pool",
                            isinstance(replacements, list)
                            and NOISE_MIN_REPLACEMENTS <= len(replacements)
                            <= NOISE_MAX_REPLACEMENTS
                            and all(r in rare_string_set for r in replacements),
                            f"got {replacements}")
            validator.check(f"14. doc {doc_id} token_noise sentence has a replacement",
                            any(r in record["anomaly_sentence"] for r in replacements))

    # 15. rerunning generation produces identical output
    rerun_rows, _, _ = build_rows()
    rerun_payload = serialize(rerun_rows)
    validator.check("15. rerun produces byte-identical output",
                    rerun_payload == payload)

    return validator.passed


def _is_lowest_similarity(record, records):
    """Confirm the chosen donor really is the least lexically similar doc."""
    target_terms = lexical_terms(record["source_text"])
    best_score = None
    for other in records:
        if other["doc_id"] == record["doc_id"]:
            continue
        score = jaccard(target_terms, lexical_terms(other["source_text"]))
        if best_score is None or score < best_score:
            best_score = score
    donor = records[record["meta"]["donor_doc_id"]]
    return abs(jaccard(target_terms, lexical_terms(donor["source_text"])) - best_score) < 1e-12


def _verify_contradiction(record, final_sentences, index):
    """The inserted sentence must differ from its source sentence in exactly one
    numeric literal (doubled, or 0 -> 1), and that source sentence must occur
    BEFORE the anomaly in the final document."""
    contradiction = record["contradiction"]
    origin = contradiction["source_sentence"]
    original_literal = contradiction["original_literal"]
    anomaly = record["anomaly_sentence"]

    if original_literal not in origin:
        return False
    if anomaly == origin:
        return False

    origin_numbers = NUMERIC_LITERAL_RE.findall(origin)
    anomaly_numbers = NUMERIC_LITERAL_RE.findall(anomaly)
    if not origin_numbers or len(origin_numbers) != len(anomaly_numbers):
        return False
    differing = [i for i, (a, b) in enumerate(zip(origin_numbers, anomaly_numbers))
                 if a != b]
    if len(differing) != 1:
        return False
    i = differing[0]
    if anomaly_numbers[i] != doubled_literal(origin_numbers[i]):
        return False

    # Exactly one numeric literal differs, and nothing else in the sentence does.
    if NUMERIC_LITERAL_RE.sub("\x00", origin) != NUMERIC_LITERAL_RE.sub("\x00", anomaly):
        return False

    # The contradicted sentence must appear earlier in the final document.
    source_sentences = sentences_text(record["source_text"],
                                      split_sentences(record["source_text"]))
    stripped = [s.strip() for s in source_sentences]
    try:
        origin_index = stripped.index(origin.strip())
    except ValueError:
        return False
    return origin_index < index


# -----------------------------------------------------------------------------#
# main
# -----------------------------------------------------------------------------#
def main():
    print(f"seed={SEED}")

    rows, records, rare_string_set = build_rows()
    payload = serialize(rows)
    write_output(payload)

    checks = validate(records, rare_string_set, payload)

    counts = Counter(r["anomaly_type"] for r in rows if r["condition"] == "anomalous")
    anomalous_rows = sum(1 for r in rows if r["condition"] == "anomalous")
    control_rows = sum(1 for r in rows if r["condition"] == "control")

    print()
    print("=" * 70)
    print("anomaly set build complete")
    print("=" * 70)
    print(f"valid source documents : {len(records)}")
    print(f"anomalous rows         : {anomalous_rows}")
    print(f"control rows           : {control_rows}")
    print(f"total rows             : {len(rows)}")
    print("anomaly counts         :")
    for anomaly_type in ANOMALY_TYPES:
        print(f"    {anomaly_type:<24} {counts.get(anomaly_type, 0)}")
    print(f"output path            : {OUTPUT_PATH}")
    print(f"checks passed          : {len(checks)}")
    print("validation             : PASS")


if __name__ == "__main__":
    main()
