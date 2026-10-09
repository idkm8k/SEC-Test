Hi :D

This project pulls 10-K reports from the SEC's EDGAR, searches them for mentions of topics you choose (AI, cybersecurity, ESG, etc.), and analyzes the sentiment expressed around those mentions using a transformer model.

I made this with a lot of version control in VS Code at the time, but I hadn't planned on uploading it to GitHub before, so the repo is a little messy and doesn't reflect the timeline. Here's a quick guide to what's what.

## Files

- `Datasetmore.py` is the best working version so far.
- `Dataset.py` and `experimental.py` are previous or alternate versions. `experimental.py` tries a semantic search approach (Sentence-BERT) on top of the keyword matching.
- `Synonyms.py` is one of the first versions. For now it only serves as a smaller piece of the overall project and is kept as legacy.   
- `synonym_concepts.py` is the list of search keywords for each topic. It has to be filled in manually, because English is a vast language and it's best for the user to choose their own keywords. 
- `a.csv` is a sample input file, for when you want to run a list of several companies instead of typing in just two stock tickers.
- `niccs.csv` is a cybersecurity terms list published by NICCS, included as a source for expanding the keyword lists.

## How to run it

```bash
pip install torch transformers edgartools
python Datasetmore.py
```

You'll need to set your own name/email in the `set_identity(...)` line near the top of the script, since the SEC asks for it when you access EDGAR.

## Why keyword lists instead of something fancier (i.e entering a topic and having AI find all related works automatically?

More powerful tools exist for analyzing semantics, but they're harder to tune and harder to explain. This version is simple, transparent, and free: you can always see exactly which term caused a sentence to count, and you can improve it just by editing the keyword list.

The tradeoff is that the lists take manual upkeep, and keyword matching can miss paraphrases or over-match some terms. Sentence-level sentiment is also not the same thing as a company's overall stance, so this is more for fun, or to be used as a starting point.

## What's next

- Include quarterly reports and incident reports so there's more information available to the user.
- Integrate (or at least adapt) more lists published by established authorities to expand the manual keyword lists, such as the Niccs.csv which isn't actually integrated yet.

## Credits

- Sentiment model: FinBERT (`yiyanghkust/finbert-tone`), Huang, Wang & Yang (2023), *FinBERT: A Large Language Model Approach to Extracting Information from Financial Text*, Contemporary Accounting Research.
- Semantic search in `experimental.py`: Sentence-BERT (`all-mpnet-base-v2`), Reimers & Gurevych (2019).
- Filings: SEC EDGAR, accessed with the `edgartools` package.
- Cybersecurity terms: NICCS glossary.
