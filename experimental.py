"""
Hybrid Lexical + Semantic Sentiment Tally Script

WHY THE PREVIOUS VERSION MISSED SO MUCH:
  1. Threshold of 0.45 is calibrated for sentence-vs-sentence comparison.
     When comparing a bare word to a full sentence, realistic cosine
     similarities for genuinely relevant sentences sit in the 0.25-0.40
     range - so 0.45 was silently discarding most real matches.
  2. No lexical fallback: sentences containing the literal word were still
     subject to the same threshold and could be dropped.
  3. Single-word embedding: "cybersecurity" as a lone token gives the model
     very little signal vs. a descriptive phrase.

THIS VERSION FIXES ALL THREE:
  - LEXICAL PASS: sentences containing the target word literally are ALWAYS
    included, no threshold required. The matched text is the "phrase".
  - SEMANTIC PASS: remaining sentences are embedded and compared against a
    richer query (target word + optional context hints you provide).
    Threshold is 0.25 - appropriate for word-to-sentence comparison.
  - PHRASE EXTRACTION: for semantic-only matches, candidate n-grams from
    the sentence are embedded and the single best-matching one is extracted.
    If even the best phrase scores < PHRASE_SIM_THRESHOLD (0.35) the
    sentence is discarded as a false positive.
  - OUTPUTS: two CSVs. The phrase/synonym CSV shows which terms are
    actually being used (exact text for lexical hits, extracted n-gram for
    semantic hits) and whether each was a direct mention or a paraphrase.
    No full sentences are written to any output file.

TUNING GUIDE (if results are off after a run):
  Too much noise / irrelevant phrases in output:
    -> Raise PHRASE_SIM_THRESHOLD (e.g. 0.35 -> 0.42)
  Missing things you know are in the filing:
    -> Lower SENTENCE_SIM_THRESHOLD (e.g. 0.25 -> 0.18)
    -> Add more relevant context hints when prompted
"""

import os
import re
import csv
import json
import time
import warnings
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader
from edgar import set_identity, Company
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")
set_identity("your.email@example.com")  # Update with your email

# ---------------------------------------------------------------------------
# TUNABLE THRESHOLDS
# ---------------------------------------------------------------------------
# Semantic sentence-level recall threshold. Lower = more recall, more noise.
# 0.25 is appropriate for a word/short phrase vs. a full sentence.
# If your query is itself a long descriptive sentence, raise toward 0.35.
SENTENCE_SIM_THRESHOLD = 0.25

# Phrase-level precision threshold. For semantic-only matches, the best
# extracted n-gram from the sentence must beat this to be kept.
# Phrase-vs-word comparison is more apples-to-apples so this can be higher.
PHRASE_SIM_THRESHOLD = 0.35

# Minimum number of words a semantically-extracted phrase must contain.
# Single words like "security", "risk", "technology" are too generic to be
# meaningful on their own - they could refer to anything. Requiring at least
# 2 words ("network security", "security breach") keeps phrases informative.
# NOTE: this only applies to SEMANTIC matches. Lexical matches report the
# exact keyword you searched for (e.g. "cybersecurity"), which is always
# unambiguous regardless of length.
MIN_SEMANTIC_PHRASE_WORDS = 2

# ---------------------------------------------------------------------------
# GPU THROTTLE
#
# set_per_process_memory_fraction(0.45) only limits VRAM (memory), NOT
# compute utilization. To actually limit how hard the GPU works:
#   1. Use smaller batch sizes (primary lever - reduces peak utilization)
#   2. torch.cuda.synchronize() after each batch (waits for GPU to actually
#      finish - without this, sleep() doesn't help because CUDA queues work
#      asynchronously and the GPU keeps running while the CPU thread sleeps)
#   3. time.sleep(GPU_THROTTLE_DELAY) after synchronize (breathing room)
#
# GPU_THROTTLE_DELAY of 0.15s drops utilization significantly. Raise it if
# still hitting too high; lower it (or set to 0.0) to go faster.
# ---------------------------------------------------------------------------
GPU_THROTTLE_DELAY = 0.15   # seconds to pause between GPU batches
EMBED_BATCH_SIZE = 16        # SentenceBERT batch size (lower = less utilization)
PHRASE_EMBED_BATCH_SIZE = 64 # Phrase candidate batch size
FINBERT_BATCH_SIZE = 8       # FinBERT batch size (lower = less utilization)

CACHE_DIR = "filing_cache"
EMBEDDING_MODEL = "all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# GPU SETUP
# ---------------------------------------------------------------------------

def setup_gpu():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        torch.cuda.set_per_process_memory_fraction(0.45, 0)
        print(f"\n[HARDWARE] CUDA GPU Detected: {device_name}")
        print("[HARDWARE] VRAM capped at 45%.")
        return 0
    else:
        print("\n[HARDWARE] No NVIDIA GPU/CUDA detected. Running on CPU.")
        return -1


class FilingDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


# ---------------------------------------------------------------------------
# ITEM-SECTION DETECTION (unchanged from previous scripts)
# ---------------------------------------------------------------------------

ITEM_LINE_PATTERN = re.compile(r'^\s*item\s+(\d{1,2}[a-c]?)\b\.?\s*(.*)$', re.IGNORECASE)


def classify_item_line(line):
    stripped = line.strip()
    if not stripped:
        return None
    match = ITEM_LINE_PATTERN.match(stripped)
    if not match:
        return None
    item_num = match.group(1).upper()
    rest = match.group(2).strip()
    has_dot_leader = bool(re.search(r'\.{2,}', stripped)) or bool(re.search(r'(\.\s){2,}', stripped))
    ends_with_pagenum = bool(re.search(r'\d{1,4}\s*$', stripped))
    if not rest:
        ends_with_pagenum = False
    is_toc_entry = has_dot_leader or ends_with_pagenum
    word_count = len(rest.split())
    looks_like_sentence = rest.endswith(('.', '!', '?')) and word_count > 6
    is_real_header = (not is_toc_entry) and (not looks_like_sentence) and word_count <= 15
    return item_num, is_real_header


def clean_section_lines(lines):
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s or re.fullmatch(r'\d{1,4}', s) or s.lower() in ('table of contents', 'index'):
            continue
        cleaned.append(s)
    return cleaned


def split_into_item_sections(raw_text):
    lines = raw_text.splitlines()
    anchor_idx = None
    for i, line in enumerate(lines):
        result = classify_item_line(line)
        if result and result[0] == '1' and result[1]:
            anchor_idx = i
            break
    if anchor_idx is None:
        return [("Unknown", raw_text)]
    boundaries = []
    for i in range(anchor_idx, len(lines)):
        result = classify_item_line(lines[i])
        if result and result[1]:
            boundaries.append((i, f"Item {result[0]}"))
    sections = []
    for idx, (line_idx, item_label) in enumerate(boundaries):
        next_idx = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        section_lines = clean_section_lines(lines[line_idx:next_idx])
        sections.append((item_label, "\n".join(section_lines)))
    return sections


def is_toc_or_header(sentence):
    s = sentence.strip()
    if not s:
        return True
    if re.match(r'^\s*(item|part)\s+\d+[a-z]?\.?\s*[-—:]?', s, re.IGNORECASE):
        if len(s.split()) < 12 or not s.endswith(('.', '!', '?')):
            return True
    if re.search(r'\b\d{1,4}\s+item\s+\d{1,2}[a-z]?\.?\s*$', s, re.IGNORECASE) or re.search(r'\.{3,}', s):
        return True
    if re.search(r'\b\d{1,4}\s*$', s) and not s.endswith(('.', '!', '?')) and len(s.split()) <= 10:
        return True
    if not s.endswith(('.', '!', '?')) and len(s.split()) <= 15:
        return True
    return False


# ---------------------------------------------------------------------------
# LEXICAL MATCHING
# ---------------------------------------------------------------------------

def build_lexical_pattern(term):
    """
    Builds a word-boundary-anchored regex for the target word.
    Same logic as the dictionary script: leading boundary always,
    trailing boundary only for very short terms (<=3 chars) to prevent
    matching mid-word.
    """
    parts = [p for p in re.split(r'[\s-]+', term.strip()) if p]
    escaped = [re.escape(p) for p in parts]
    body = r'[\s-]+'.join(escaped)
    if len(''.join(parts)) <= 3:
        return re.compile(r'\b' + body + r'\b', re.IGNORECASE)
    return re.compile(r'\b' + body, re.IGNORECASE)


# ---------------------------------------------------------------------------
# PHRASE EXTRACTION (for semantic-only matches)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "can", "may",
    "might", "must", "shall", "this", "that", "these", "those", "it", "its",
    "as", "we", "our", "us", "they", "their", "them", "he", "she", "his",
    "her", "you", "your", "i", "not", "no", "such", "than", "too", "very",
    "so", "also", "which", "who", "where", "when", "while", "any", "all",
    "each", "other", "more", "most", "some", "own", "same", "both",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def generate_phrase_candidates(sentence, max_n=3, min_words=2):
    """
    Generates min_words- to max_n-word candidate phrases from a sentence,
    filtering out pure-stopword grams. Returns a deduplicated list.

    min_words=2 by default for semantic matches: single words like "security",
    "risk", "technology" are too generic to be meaningful without context.
    "network security" or "security breach" are informative; "security" alone
    is not. Lexical matches bypass this entirely (they report the exact
    keyword text, not an extracted phrase).
    """
    words = WORD_RE.findall(sentence.lower())
    candidates = []
    seen = set()
    for n in range(min_words, max_n + 1):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if all(w in STOPWORDS for w in gram):
                continue
            phrase = " ".join(gram)
            if phrase not in seen:
                seen.add(phrase)
                candidates.append(phrase)
    return candidates


# ---------------------------------------------------------------------------
# QUERY EMBEDDING (with optional context expansion)
# ---------------------------------------------------------------------------

def build_query_embedding(target_word, context_hints, embedder, device):
    """
    Builds a query embedding by encoding the target word AND any optional
    context hints the user provided, then averaging the result.

    Why: encoding a bare single word gives the model very little signal.
    Encoding "cybersecurity" alongside user-supplied hints like
    "data breach, network security, information security" produces a richer
    vector that pulls in semantically related sentences more reliably,
    without locking the model into a hardcoded synonym list.

    If no context hints are given, just the target word is encoded.
    """
    to_encode = [target_word]
    if context_hints:
        # Each hint is encoded separately and averaged in with the target.
        to_encode.extend(context_hints)

    embeddings = embedder.encode(
        to_encode,
        convert_to_tensor=True,
        device=str(device),
        show_progress_bar=False
    )
    # Average across all encoded strings to get one combined query vector.
    return embeddings.mean(dim=0)


# ---------------------------------------------------------------------------
# CACHE / DOWNLOAD LAYER (unchanged)
# ---------------------------------------------------------------------------

def get_cache_path(ticker, target_year):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker}_{target_year}.json")


def get_legacy_cache_path(ticker, target_year):
    return os.path.join(CACHE_DIR, f"{ticker}_{target_year}.txt")


def fetch_company_data(ticker, fiscal_year_suffix):
    ticker = ticker.upper()
    target_year = f"20{fiscal_year_suffix}"
    cache_path = get_cache_path(ticker, target_year)
    legacy_path = get_legacy_cache_path(ticker, target_year)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            return {
                'ticker': ticker, 'company_name': cached['company_name'],
                'report_text': cached['report_text'], 'from_cache': True
            }
        except Exception as e:
            print(f"[!] Cache read failed for {ticker}, re-downloading: {e}")

    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
            try:
                company_name = Company(ticker).name
            except Exception:
                company_name = ticker
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'company_name': company_name, 'report_text': report_text}, f)
            except Exception:
                pass
            return {
                'ticker': ticker, 'company_name': company_name,
                'report_text': report_text, 'from_cache': True
            }
        except Exception as e:
            print(f"[!] Legacy cache read failed for {ticker}, re-downloading: {e}")

    try:
        company = Company(ticker)
        company_name = company.name
        filings = company.get_filings(form="10-K")
        target_filing = next(
            (f for f in filings if f.period_of_report.startswith(target_year)), None
        )
        if not target_filing:
            return {'ticker': ticker, 'company_name': company_name,
                    'error': f"No 10-K found for period {target_year}"}
        report_text = target_filing.text()
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'company_name': company_name, 'report_text': report_text}, f)
        except Exception as e:
            print(f"[!] Could not write cache for {ticker}: {e}")
        return {
            'ticker': ticker, 'company_name': company_name,
            'report_text': report_text, 'from_cache': False,
            'period': target_filing.period_of_report
        }
    except Exception as e:
        return {'ticker': ticker, 'company_name': ticker, 'error': str(e)}


# ---------------------------------------------------------------------------
# MAIN BATCH FUNCTION
# ---------------------------------------------------------------------------

def batch_hybrid_tally(tickers, fiscal_year_suffix, target_word,
                       context_hints, max_download_workers=4):
    """
    For each company, processes its 10-K using a hybrid lexical + semantic
    approach to find sentences related to target_word, then runs FinBERT
    sentiment on each, and writes:

      1. Master tally CSV: per-item sentiment counts + COMPANY TOTAL row.
      2. Phrase breakdown CSV: which phrases/terms were used and how often,
         labelled as "Lexical" (exact keyword match) or "Semantic" (synonym
         caught by embedding similarity), + TOTAL row matching Master.
    """
    if not tickers:
        return

    start_time = time.perf_counter()

    device_id = setup_gpu()
    device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

    print(f"\nLoading semantic embedding model ({EMBEDDING_MODEL})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=str(device))

    print("Loading FinBERT sentiment model...")
    finbert_name = "yiyanghkust/finbert-tone"
    tokenizer = BertTokenizer.from_pretrained(finbert_name)
    finbert = BertForSequenceClassification.from_pretrained(finbert_name, num_labels=3)
    finbert.to(device)
    if device.type == 'cuda':
        finbert.half()
        print("[GPU] FinBERT running in fp16.")
    finbert.eval()

    # Build the query vector once — reused for every company.
    print(f"\nBuilding query embedding for: '{target_word}'" +
          (f" + {len(context_hints)} context hint(s)" if context_hints else "") + "...")
    query_embedding = build_query_embedding(target_word, context_hints, embedder, device)

    # Lexical pattern for direct keyword matching.
    lexical_pattern = build_lexical_pattern(target_word)

    safe_name = re.sub(r'[^a-z0-9]+', '_', target_word.strip().lower()).strip('_')
    tally_csv = f"Master_20{fiscal_year_suffix}_{safe_name}_item_tally.csv"
    phrase_csv = f"Master_20{fiscal_year_suffix}_{safe_name}_phrase_breakdown.csv"

    with open(tally_csv, mode='w', newline='', encoding='utf-8-sig') as tf, \
         open(phrase_csv, mode='w', newline='', encoding='utf-8-sig') as pf:

        tally_writer = csv.writer(tf)
        tally_writer.writerow([
            'Company Name', 'Company Ticker', 'Target Word', 'Item Section',
            'Positive', 'Neutral', 'Negative', 'Total Mentions'
        ])

        phrase_writer = csv.writer(pf)
        phrase_writer.writerow([
            'Company Name', 'Company Ticker', 'Target Word',
            'Matched Phrase', 'Match Type', 'Occurrences'
        ])

        print(f"\n[+] Queuing {len(tickers)} download(s) across "
              f"{max_download_workers} background worker(s)...")

        with ThreadPoolExecutor(max_workers=max_download_workers) as executor:
            futures = {
                executor.submit(fetch_company_data, t, fiscal_year_suffix): t
                for t in tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    data = future.result()
                except Exception as e:
                    print(f"[!] Unexpected error fetching {ticker}: {e}")
                    continue

                if 'error' in data:
                    print(f"[-] {ticker}: {data['error']}. Skipping.")
                    continue

                company_name = data['company_name']
                report_text = data['report_text']
                source_note = "cache" if data.get('from_cache') else "downloaded"
                print(f"\n{'='*60}\n--- Processing {company_name} ({ticker}) [{source_note}] ---")

                compute_start = time.perf_counter()
                try:
                    sections = split_into_item_sections(report_text)
                    item_counts = {
                        item_label: {'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0}
                        for item_label, _ in sections
                    }
                    total_pos, total_neu, total_neg = 0, 0, 0
                    # phrase -> {'type': 'Lexical'|'Semantic', 'count': int}
                    phrase_tally = {}

                    for item_label, section_text in sections:
                        sentences = re.split(
                            r'(?<=[.!?])\s+',
                            " ".join(section_text.split())
                        )

                        # -- Filter to valid sentences first --
                        valid = []
                        for s in sentences:
                            c = s.strip()
                            if len(c) > 30 and not is_toc_or_header(c):
                                valid.append(c[:1500])

                        if not valid:
                            continue

                        # ---- STAGE 1: LEXICAL PASS ----
                        # Sentences containing the literal keyword are
                        # guaranteed inclusions regardless of any score.
                        lexical_hits = {}   # sentence -> matched_phrase
                        no_literal_match = []

                        for sentence in valid:
                            m = lexical_pattern.search(sentence)
                            if m:
                                # Use the exact text matched by the regex as
                                # the "phrase", lower-cased and stripped.
                                phrase = m.group(0).strip().lower()
                                lexical_hits[sentence] = phrase
                            else:
                                no_literal_match.append(sentence)

                        # ---- STAGE 2: SEMANTIC PASS ----
                        # Only run on sentences that didn't already match
                        # lexically, to avoid double-counting.
                        semantic_hits = {}  # sentence -> best_phrase

                        if no_literal_match:
                            sent_embeddings = embedder.encode(
                                no_literal_match,
                                convert_to_tensor=True,
                                device=str(device),
                                batch_size=EMBED_BATCH_SIZE,
                                show_progress_bar=False
                            )
                            # Throttle: wait for GPU to actually finish before
                            # continuing. Without synchronize(), the CPU thread
                            # races ahead while the GPU is still working.
                            if device.type == 'cuda':
                                torch.cuda.synchronize()
                                time.sleep(GPU_THROTTLE_DELAY)

                            sims = util.cos_sim(
                                sent_embeddings, query_embedding
                            ).squeeze(-1)

                            stage1_survivors = [
                                (no_literal_match[i], float(sims[i]))
                                for i in range(len(no_literal_match))
                                if float(sims[i]) >= SENTENCE_SIM_THRESHOLD
                            ]

                            # -- STAGE 2b: Phrase extraction for semantic survivors --
                            # Collect all candidate phrases from all survivors
                            # and embed them in ONE batch for efficiency.
                            if stage1_survivors:
                                all_candidates = []   # (sent_idx, phrase)
                                sent_lookup = []      # (sentence, sim)
                                for sent_idx, (sentence, sim) in enumerate(stage1_survivors):
                                    phrases = generate_phrase_candidates(
                                        sentence,
                                        min_words=MIN_SEMANTIC_PHRASE_WORDS
                                    )
                                    for phrase in phrases:
                                        all_candidates.append((sent_idx, phrase))
                                    sent_lookup.append((sentence, sim))

                                if all_candidates:
                                    phrases_only = [p for _, p in all_candidates]
                                    phrase_embeddings = embedder.encode(
                                        phrases_only,
                                        convert_to_tensor=True,
                                        device=str(device),
                                        batch_size=PHRASE_EMBED_BATCH_SIZE,
                                        show_progress_bar=False
                                    )
                                    if device.type == 'cuda':
                                        torch.cuda.synchronize()
                                        time.sleep(GPU_THROTTLE_DELAY)
                                    phrase_sims = util.cos_sim(
                                        phrase_embeddings, query_embedding
                                    ).squeeze(-1).tolist()

                                    # Group by sentence index, find best phrase per sentence
                                    best_per_sentence = {}
                                    for (sent_idx, phrase), psim in zip(all_candidates, phrase_sims):
                                        if sent_idx not in best_per_sentence or psim > best_per_sentence[sent_idx][1]:
                                            best_per_sentence[sent_idx] = (phrase, psim)

                                    for sent_idx, (sentence, _) in enumerate(sent_lookup):
                                        if sent_idx in best_per_sentence:
                                            best_phrase, best_sim = best_per_sentence[sent_idx]
                                            if best_sim >= PHRASE_SIM_THRESHOLD:
                                                semantic_hits[sentence] = best_phrase

                        # ---- STAGE 3: FINBERT SENTIMENT ----
                        # Combine lexical and semantic hits.
                        # Lexical takes precedence if a sentence was caught by both
                        # (shouldn't happen since semantics only ran on no-literal-match,
                        # but guard it anyway).
                        pending = []  # (sentence, phrase, match_type)
                        for sentence, phrase in lexical_hits.items():
                            pending.append((sentence, phrase, 'Lexical'))
                        for sentence, phrase in semantic_hits.items():
                            pending.append((sentence, phrase, 'Semantic'))

                        if not pending:
                            continue

                        texts_to_score = [p[0] for p in pending]
                        dataset = FilingDataset(texts_to_score)
                        dataloader = DataLoader(dataset, batch_size=FINBERT_BATCH_SIZE, shuffle=False)

                        labels = []
                        with torch.no_grad():
                            for batch_texts in dataloader:
                                inputs = tokenizer(
                                    batch_texts, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt"
                                )
                                inputs = {k: v.to(device) for k, v in inputs.items()}
                                outputs = finbert(**inputs)
                                preds = torch.argmax(outputs.logits, dim=-1)
                                for p_idx in preds:
                                    labels.append(
                                        finbert.config.id2label[p_idx.item()].upper()
                                    )
                                # Throttle between FinBERT mini-batches.
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                                    time.sleep(GPU_THROTTLE_DELAY)

                        for (_, phrase, match_type), label in zip(pending, labels):
                            item_counts[item_label][label] += 1
                            if label == 'POSITIVE':
                                total_pos += 1
                            elif label == 'NEUTRAL':
                                total_neu += 1
                            elif label == 'NEGATIVE':
                                total_neg += 1

                            key = (phrase, match_type)
                            if key not in phrase_tally:
                                phrase_tally[key] = 0
                            phrase_tally[key] += 1

                    # ---- WRITE MASTER TALLY ROWS ----
                    for item_label, _ in sections:
                        counts = item_counts[item_label]
                        tally_writer.writerow([
                            company_name, ticker, target_word, item_label,
                            counts['POSITIVE'], counts['NEUTRAL'], counts['NEGATIVE'],
                            sum(counts.values())
                        ])
                    company_total = total_pos + total_neu + total_neg
                    tally_writer.writerow([
                        company_name, ticker, target_word, "COMPANY TOTAL",
                        total_pos, total_neu, total_neg, company_total
                    ])

                    # ---- WRITE PHRASE BREAKDOWN ROWS ----
                    # Sorted by count descending so most-used phrases appear first.
                    for (phrase, match_type), count in sorted(
                        phrase_tally.items(), key=lambda x: -x[1]
                    ):
                        phrase_writer.writerow([
                            company_name, ticker, target_word,
                            phrase, match_type, count
                        ])
                    # This total equals the Master COMPANY TOTAL exactly.
                    phrase_writer.writerow([
                        company_name, ticker, target_word,
                        "TOTAL", "", company_total
                    ])

                    lexical_count = sum(
                        c for (_, mt), c in phrase_tally.items() if mt == 'Lexical'
                    )
                    semantic_count = sum(
                        c for (_, mt), c in phrase_tally.items() if mt == 'Semantic'
                    )

                    compute_elapsed = time.perf_counter() - compute_start
                    print(f"[+] Done: {company_total} total mentions "
                          f"({lexical_count} lexical / {semantic_count} semantic). "
                          f"({compute_elapsed:.2f}s)")

                except Exception as e:
                    print(f"[!] Error processing {ticker}: {e}")
                    continue

    end_time = time.perf_counter()
    mins, secs = divmod(end_time - start_time, 60)
    print(f"\n{'='*60}")
    print(f"[FINISHED] {len(tickers)} companies in {int(mins)}m {int(secs)}s.")
    print(f"Output files:")
    print(f"  {tally_csv}")
    print(f"  {phrase_csv}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    while True:
        print(f"\n{'='*60}")
        user_year = input("Enter fiscal year (e.g., 25) or 'q' to quit: ").strip().lower()
        if user_year == 'q':
            break

        target_word = input(
            "Enter target word or phrase (e.g., 'cybersecurity', 'AI', 'litigation'): "
        ).strip()
        if not target_word:
            continue

        print(
            "\nOptional: enter context hints to improve semantic matching accuracy."
        )
        print(
            "These are additional words/phrases that describe the same concept."
        )
        print(
            "For 'cybersecurity' you might add: data breach, network security, information security"
        )
        print("Press Enter with no input to skip.")
        hints_raw = input("Context hints (comma-separated, or Enter to skip): ").strip()
        context_hints = [h.strip() for h in hints_raw.split(',') if h.strip()] if hints_raw else []

        if context_hints:
            print(f"[*] Using context hints: {context_hints}")
        else:
            print("[*] No context hints provided - using bare word only.")

        input_mode = ""
        while input_mode not in ['m', 'e']:
            input_mode = input(
                "Input tickers manually or load from file? (M for manual / E for CSV): "
            ).strip().lower()

        user_tickers = []
        if input_mode == 'm':
            print("Enter tickers (type 'o' to finish):")
            while True:
                entry = input("Ticker: ").strip()
                if entry.lower() in ['o', 'over']:
                    break
                if entry:
                    user_tickers.append(entry)
        elif input_mode == 'e':
            csv_filename = input("Enter the path/name of your CSV file: ").strip()
            if not os.path.exists(csv_filename):
                print(f"[!] File '{csv_filename}' not found. Restarting...")
                continue
            try:
                with open(csv_filename, mode='r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    target_key = next(
                        h for h in reader.fieldnames if h.strip().lower() == 'ticker'
                    )
                    for row in reader:
                        val = row[target_key]
                        if val and val.strip():
                            user_tickers.append(val.strip().upper())
                print(f"[+] Loaded {len(user_tickers)} tickers.")
            except Exception as file_err:
                print(f"[!] Failed to parse file: {file_err}")
                continue

        if user_tickers:
            batch_hybrid_tally(
                user_tickers, user_year, target_word,
                context_hints, max_download_workers=4
            )