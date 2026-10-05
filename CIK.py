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

from synonym_concepts import CONCEPTS

warnings.filterwarnings("ignore")
set_identity("your.email@example.com")  # Update with your email

CACHE_DIR = "filing_cache"


def setup_gpu():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        torch.cuda.set_per_process_memory_fraction(0.45, 0)
        print(f"\n[HARDWARE] CUDA GPU Detected: {device_name}")
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
# Item-section detection
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


def build_term_pattern(term):
    raw = term.strip()
    parts = [p for p in re.split(r'[\s-]+', raw) if p]
    escaped_parts = [re.escape(p) for p in parts]
    pattern_body = r'[\s-]+'.join(escaped_parts)
    compact_len = len(''.join(parts))
    if compact_len <= 3:
        pattern = r'\b' + pattern_body + r'\b'
    else:
        pattern = r'\b' + pattern_body
    return re.compile(pattern, re.IGNORECASE)


# ---------------------------------------------------------------------------
# NETWORK / DOWNLOAD LAYER
# ---------------------------------------------------------------------------

def get_cache_path(cache_key, target_year):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{cache_key}_{target_year}.json")


def get_legacy_cache_path(cache_key, target_year):
    return os.path.join(CACHE_DIR, f"{cache_key}_{target_year}.txt")


def fetch_company_data(identifier, fiscal_year_suffix, lookup_type='ticker'):
    lookup_type = lookup_type.lower()
    target_year = f"20{fiscal_year_suffix}"

    if lookup_type == 'cik':
        raw_cik = str(identifier).strip().upper().replace("CIK", "").strip()
        try:
            cik_number = int(float(raw_cik))
        except ValueError:
            return {'ticker': str(identifier), 'company_name': str(identifier),
                    'error': f"Invalid CIK value: {identifier}"}
        cache_key = f"CIK{cik_number}"
    else:
        cik_number = None
        cache_key = str(identifier).strip().upper()

    cache_path = get_cache_path(cache_key, target_year)
    legacy_path = get_legacy_cache_path(cache_key, target_year)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            return {
                'ticker': cached.get('ticker', cache_key), 'company_name': cached['company_name'],
                'report_text': cached['report_text'], 'from_cache': True
            }
        except Exception as e:
            print(f"[!] Cache read failed for {cache_key}, will re-download: {e}")

    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
            try:
                company = Company(cik_number if lookup_type == 'cik' else identifier.upper())
                company_name = company.name
                display_ticker = (company.tickers[0] if getattr(company, 'tickers', None) else cache_key)
            except Exception:
                company_name = cache_key
                display_ticker = cache_key
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'company_name': company_name, 'ticker': display_ticker, 'report_text': report_text}, f)
            except Exception:
                pass
            return {
                'ticker': display_ticker, 'company_name': company_name,
                'report_text': report_text, 'from_cache': True
            }
        except Exception as e:
            print(f"[!] Legacy cache read failed for {cache_key}, will re-download: {e}")

    try:
        company = Company(cik_number if lookup_type == 'cik' else identifier.upper())
        company_name = company.name
        display_ticker = (company.tickers[0] if getattr(company, 'tickers', None) else cache_key)

        filings = company.get_filings(form="10-K")
        target_filing = next(
            (f for f in filings
             if f.form == "10-K" and f.period_of_report.startswith(target_year)),
            None
        )
        if not target_filing:
            return {'ticker': display_ticker, 'company_name': company_name,
                    'error': f"No 10-K found for period {target_year}"}

        report_text = target_filing.text()

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'company_name': company_name, 'ticker': display_ticker, 'report_text': report_text}, f)
        except Exception as e:
            print(f"[!] Could not write cache for {cache_key}: {e}")

        return {
            'ticker': display_ticker, 'company_name': company_name,
            'report_text': report_text, 'from_cache': False,
            'period': target_filing.period_of_report
        }
    except Exception as e:
        return {'ticker': cache_key, 'company_name': cache_key, 'error': str(e)}


# ---------------------------------------------------------------------------
# MAIN BATCH FUNCTION
# ---------------------------------------------------------------------------

def batch_concept_tally(entries, fiscal_year_suffix, target_concept, max_download_workers=4):
    if not entries:
        return

    start_time = time.perf_counter()

    concept_patterns = {target_concept: [(term, build_term_pattern(term)) for term in CONCEPTS[target_concept]]}
    device_id = setup_gpu()

    print("\nLoading local FinBERT AI Model into memory...")
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
    model.to(device)

    if device.type == 'cuda':
        model.half()
        print("[GPU] Using fp16 (half precision) for faster inference.")

    model.eval()

    tally_csv_filename = f"Master_20{fiscal_year_suffix}_{target_concept.replace(' ', '_')}_item_tally.csv"
    synonym_csv_filename = f"Master_20{fiscal_year_suffix}_{target_concept.replace(' ', '_')}_synonym_breakdown.csv"

    with open(tally_csv_filename, mode='w', newline='', encoding='utf-8-sig') as tally_file, \
         open(synonym_csv_filename, mode='w', newline='', encoding='utf-8-sig') as syn_file:

        tally_writer = csv.writer(tally_file)
        tally_writer.writerow(['Company Name', 'Company Ticker', '10-K Word Count', 'Concept', 'Item Section', 'Item Word Count', 'Positive', 'Neutral', 'Negative', 'Total Mentions'])
        syn_writer = csv.writer(syn_file)
        syn_writer.writerow(['Company Name', 'Company Ticker', '10-K Word Count', 'Concept', 'Synonym Term', 'Term Occurrences'])

        print(f"\n[+] Queuing {len(entries)} download(s) across {max_download_workers} "
              f"background worker(s). GPU will process each as it becomes ready...")

        with ThreadPoolExecutor(max_workers=max_download_workers) as executor:
            futures = {
                executor.submit(fetch_company_data, identifier, fiscal_year_suffix, lookup_type): identifier
                for identifier, lookup_type in entries
            }

            for future in as_completed(futures):
                identifier = futures[future]

                try:
                    data = future.result()
                except Exception as e:
                    print(f"[!] Unexpected error fetching {identifier}: {e}")
                    continue

                if 'error' in data:
                    print(f"[-] {identifier}: {data['error']}. Skipping.")
                    continue

                company_name = data['company_name']
                ticker = data['ticker']
                report_text = data['report_text']
                source_note = "cache" if data.get('from_cache') else "downloaded"
                print(f"\n{'='*60}\n--- Processing {company_name} ({ticker}) [{source_note}] ---")

                gpu_start = time.perf_counter()
                try:
                    word_count = len(report_text.split())
                    sections = split_into_item_sections(report_text)
                    item_concept_counts = {item_label: {'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0} for item_label, _ in sections}
                    item_word_counts = {item_label: len(section_text.split()) for item_label, section_text in sections}
                    synonym_totals = {term: 0 for term in CONCEPTS[target_concept]}
                    total_pos, total_neu, total_neg = 0, 0, 0
                    unique_sentence_count = 0

                    for item_label, section_text in sections:
                        sentences = re.split(r'(?<=[.!?])\s+', " ".join(section_text.split()))
                        pending = []
                        for sentence in sentences:
                            cleaned = sentence.strip()
                            if len(cleaned) <= 30 or is_toc_or_header(cleaned):
                                continue
                            matched_terms = [term for term, pat in concept_patterns[target_concept] if pat.search(cleaned)]
                            if matched_terms:
                                pending.append((cleaned[:1500], matched_terms))

                        if not pending:
                            continue

                        texts_to_analyze = [p[0] for p in pending]
                        dataset = FilingDataset(texts_to_analyze)
                        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

                        results_labels = []
                        with torch.no_grad():
                            for batch_texts in dataloader:
                                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                                inputs = {k: v.to(device) for k, v in inputs.items()}
                                outputs = model(**inputs)
                                preds = torch.argmax(outputs.logits, dim=-1)
                                for p_idx in preds:
                                    results_labels.append(model.config.id2label[p_idx.item()].upper())

                        for (_, matched_terms), label in zip(pending, results_labels):
                            item_concept_counts[item_label][label] += 1
                            unique_sentence_count += 1
                            if label == 'POSITIVE':
                                total_pos += 1
                            elif label == 'NEUTRAL':
                                total_neu += 1
                            elif label == 'NEGATIVE':
                                total_neg += 1
                            for term in matched_terms:
                                synonym_totals[term] += 1

                    for item_label, _ in sections:
                        counts = item_concept_counts[item_label]
                        tally_writer.writerow([company_name, ticker, word_count, target_concept, item_label, item_word_counts[item_label], counts['POSITIVE'], counts['NEUTRAL'], counts['NEGATIVE'], sum(counts.values())])

                    tally_writer.writerow([company_name, ticker, word_count, target_concept, "COMPANY TOTAL", word_count, total_pos, total_neu, total_neg, (total_pos + total_neu + total_neg)])

                    for term in CONCEPTS[target_concept]:
                        syn_writer.writerow([company_name, ticker, word_count, target_concept, term, synonym_totals[term]])
                    syn_writer.writerow([company_name, ticker, word_count, target_concept, "TOTAL TERM OCCURRENCES (sum of rows above)", sum(synonym_totals.values())])
                    syn_writer.writerow([company_name, ticker, word_count, target_concept, "UNIQUE SENTENCES (should match Master file's COMPANY TOTAL)", unique_sentence_count])

                    gpu_elapsed = time.perf_counter() - gpu_start
                    print(f"[+] Done with {ticker}. (GPU processing: {gpu_elapsed:.2f}s)")

                except Exception as e:
                    print(f"[!] Error processing {ticker}: {e}")
                    continue

    end_time = time.perf_counter()
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)
    print(f"\n{'='*60}")
    print(f"[FINISHED] Processed {len(entries)} companies in {int(minutes)}m {int(seconds)}s.")
    print(f"{'='*60}")


if __name__ == "__main__":
    while True:
        print(f"\n{'='*60}")
        user_year = input("Enter fiscal year (e.g., 25) or 'q' to quit: ").strip().lower()
        if user_year == 'q':
            break
        search_input = input("Enter any keyword (e.g., 'ransomware', 'AI', 'ESG'): ").strip().lower()
        chosen_concept = None
        for concept, synonyms in CONCEPTS.items():
            if search_input == concept.lower() or search_input in [s.lower() for s in synonyms]:
                chosen_concept = concept
                break
        if not chosen_concept:
            confirm = input(f"[!] '{search_input}' not found. Proceed with standalone search? (Y/N): ").strip().lower()
            if confirm == 'y':
                chosen_concept = search_input
                CONCEPTS[chosen_concept] = [search_input]
            else:
                continue
        print(f"\n[*] Concept identified: '{chosen_concept}'")

        input_mode = ""
        while input_mode not in ['m', 'c']:
            input_mode = input("Input tickers manually or load from CSV file? (M for manual / C for CSV file): ").strip().lower()

        entries = []
        if input_mode == 'm':
            print("Enter tickers (type 'o' to finish):")
            while True:
                entry = input("Ticker: ").strip()
                if entry.lower() in ['o', 'over']:
                    break
                if entry:
                    entries.append((entry.upper(), 'ticker'))
        elif input_mode == 'c':
            csv_filename = input("Enter the path/name of your CSV file (e.g., companies.csv): ").strip()
            if not os.path.exists(csv_filename):
                print(f"[!] File '{csv_filename}' not found. Restarting...")
                continue
            try:
                with open(csv_filename, mode='r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    
                    # Normalize headers to lowercase to handle case-insensitive header matching
                    fieldnames = [field.strip().lower() for field in reader.fieldnames] if reader.fieldnames else []
                    if 'cik' not in fieldnames:
                        print(f"[!] No 'cik' column header found in '{csv_filename}'. Restarting...")
                        continue

                    # Map normalized 'cik' back to exact column key name
                    cik_key = next(field for field in reader.fieldnames if field.strip().lower() == 'cik')

                    for row in reader:
                        cik_val = row.get(cik_key)
                        if cik_val and cik_val.strip():
                            entries.append((cik_val.strip(), 'cik'))
                            
                print(f"[+] Loaded {len(entries)} CIK(s) from '{csv_filename}'.")
            except Exception as file_err:
                print(f"[!] Failed to parse file: {file_err}")
                continue

        if entries:
            batch_concept_tally(entries, user_year, chosen_concept, max_download_workers=4)