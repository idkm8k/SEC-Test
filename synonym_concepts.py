"""
Concept -> synonym list configuration for batch_concept_sentiment_tally.py.

Each key is a human-readable "concept" name that will appear in the output
CSVs. Each value is a list of words/phrases that, if found in a sentence,
count as a mention of that concept.

WHY CURATED LISTS (vs. fuzzy matching or vector/embedding similarity):

- Fully transparent and auditable - you can always see exactly WHICH term
  caused a sentence to count, and quickly add/remove terms if something is
  over- or under-matching.
- Essentially free to run (just regex - no extra GPU pass beyond FinBERT),
  which matters a lot when scaling to ~2000 filings.
- Fuzzy string (edit-distance) matching mainly helps with spelling/
  hyphenation variants of the SAME term (e.g. "cyber-attack" vs
  "cyberattack") - it does NOT find true synonyms like "machine learning"
  for "AI", since the strings share almost no characters. The
  build_term_pattern() helper already handles the hyphen-vs-space case via
  [\\s-]+, so you don't need fuzzy matching for that.
- Vector/embedding similarity CAN catch paraphrases with no shared
  vocabulary at all, but requires an extra model pass over every sentence in
  every filing (expensive at 2000x scale), a similarity threshold that's
  hard to pick/verify without a labeled dataset, and is much harder to
  explain when someone asks "why did this count?".

PRACTICAL "~93% ACCURACY" STRATEGY: put the effort into making these lists
comprehensive up front - brainstorm domain-specific phrasings, skim a few
real 10-Ks for the exact wording companies actually use - then keep
refining this file over time as you spot-check results. A thorough list with
simple matching will beat a fuzzy/embedding approach with a badly-tuned
threshold, and it's something YOU can keep improving without touching the
matching code at all.

NOTES ON SPECIFIC TERMS (precision/recall tradeoffs - read before a big run):

- Short acronyms ("AI", "ML", "ESG", "GHG") are matched as WHOLE WORDS ONLY
  (both leading AND trailing word boundaries) by build_term_pattern() - this
  avoids "ai" matching inside "gain"/"main"/"maintain". Acronyms can still
  occasionally collide with unrelated meanings though (e.g. "ML" could in
  theory appear in some unrelated abbreviation). If you spot false positives
  for a short term in your output, just delete it from the list below.
- "sustainable" is a double-edged term: central to ESG language
  ("sustainable practices", "environmental sustainability") but ALSO common
  in purely financial phrases like "sustainable competitive advantage" or
  "sustainable revenue growth" that have nothing to do with the environment.
  Kept here for higher recall on ESG content - remove it if it's too noisy
  for what you're studying.
- "ml" (machine learning) has the same short-acronym caveat as above.
"""

CONCEPTS = {
    "Artificial Intelligence": [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural network",
        "generative ai",
        "large language model",
        "llm",
        "natural language processing",
        "computer vision",
        "ai model",
        "ai system",
        "ai-powered",
        "ai-driven",
        "automation",
        "ai",
        "ml",
    ],

    "Cybersecurity": [
        "cybersecurity",
        "cyber security",
        "cyberattack",
        "cyber attack",
        "data breach",
        "ransomware",
        "information security",
        "network security",
        "data privacy",
        "cyber incident",
        "cyber threat",
        "phishing",
        "malware",
        "security incident",
        "unauthorized access",
    ],

    "Environmental / ESG": [
        "climate change",
        "greenhouse gas",
        "ghg emission",
        "ghg",
        "carbon emission",
        "carbon footprint",
        "carbon neutral",
        "net zero",
        "sustainability",
        "sustainable",
        "renewable energy",
        "environmental, social and governance",
        "esg",
        "decarbonization",
        "emissions reduction",
        "scope 1",
        "scope 2",
        "scope 3",
    ],
}