# Charlotte SME Greenwashing Detection

A natural language processing pipeline that detects greenwashing signals on small business websites. Built for a study of 38 Charlotte, NC SMEs using ClimateBERT.

## What it does

1. Scrapes text from small business websites.
2. Runs four ClimateBERT models to detect climate-related content, environmental commitments, commitment specificity, and sentiment.
3. Calculates a Greenwashing Severity Index (GSI) for each business.

## The GSI formula

GSI = (density x 0.30) + (positivity x 0.35) + (vagueness x 0.35)

Density is the percent of total sentences that contain climate claims. Positivity is the percent of climate sentences with positive sentiment. Vagueness is the percent of commitment sentences that lack specifics. GSI ranges from 0.0 (no greenwashing signal) to 1.0 (strong signal).

## Files

smes.csv — List of 38 Charlotte SMEs with website URLs.
run_pipeline.py — Main script that scrapes websites and calculates GSI.
test_pipeline.py — Test script to verify the pipeline works.
results.csv — Output with GSI scores for all 38 businesses.
README.md — This file.

## Setup

You need Python 3.10 or later. Open Command Prompt (CMD) and run:

pip install torch transformers pandas beautifulsoup4 requests numpy tqdm scikit-learn scipy openpyxl

## How to run

python run_pipeline.py

The script prints progress as it scrapes each website, shows "Models ready" when ClimateBERT loads, and saves results.csv when done.

## Output format

business_name — Company name.
url — Website URL.
n_sentences — Total scraped sentences.
n_climate — Climate-related sentences found.
n_claims — Environmental commitment sentences.
density — Percent of sentences that are climate claims.
positivity — Sentiment score from 0 to 1.
vagueness — Percent of claims lacking specifics.
gsi — Greenwashing Severity Index from 0 to 1.

## Sample results

Empower Efficiency scored 1.0000. Home Clean Heroes scored 0.8306. Ekologicall scored 0.7696. Piedmont Plastics scored 0.5426. Jones Dry Cleaning scored 0.5500.

## Limitations

Only analyzes website text, not images or design. JavaScript-heavy sites may return empty results. GSI measures online messaging, not actual business practices. Most SMEs operate outside EPA regulatory scope with no compliance records to cross-reference.

## License

Educational or research use.
