# Charlotte SME Greenwashing Detection

## Overview

This model functions as a natural language processing pipeline designed to detect potential greenwashing signals on small business websites. Built using ClimateBERT, the project analyzes 38 SMEs in Charlotte, NC by extracting website text and evaluating environmental messaging patterns.

### What It Does

- Scrapes text from small business websites.
- Runs four ClimateBERT models to detect:
  - climate-related content,
  - environmental commitments,
  - commitment specificity,
  - and sentiment.
- Calculates a Greenwashing Severity Index (GSI) for each business.

## The GSI formula

GSI = (density x 0.30) + (positivity x 0.35) + (vagueness x 0.35)

The GSI combines three factors:
- Density: percent of all sentences on the site that contain climate claims
- Positivity: how positive the climate sentences are (0 to 1)
- Vagueness: percent of claims that lack any specific detail

GSI ranges from 0.0 (no greenwashing signal) to 1.0 (strong signal).


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

## Output Format

- **business_name** — Company name.
- **url** — Website URL.
- **n_sentences** — Total scraped sentences.
- **n_climate** — Climate-related sentences found.
- **n_claims** — Environmental commitment sentences.
- **density** — Percent of sentences that are climate claims.
- **positivity** — Sentiment score from 0 to 1.
- **vagueness** — Percent of claims lacking specifics.
- **gsi** — Greenwashing Severity Index from 0 to 1.

## Key Sample Results

| SME | GSI |
|---|---|
| Empower Efficiency | 1.0000 |
| Home Clean Heroes | 0.8306 |
| Ekologicall | 0.7696 |
| Piedmont Plastics | 0.5426 |
| Jones Dry Cleaning | 0.5500 |


## Research Projections

Based on preliminary outputs from 38 Charlotte SMEs, this model is expected to reveal measurable variation in greenwashing signals across different small business categories. Early patterns suggest that service-based SMEs tend to exhibit higher GSI values, potentially due to heavier reliance on marketing language and less standardized environmental reporting.

Future iterations of this model could expand the dataset to 100+ SMEs across multiple cities to test whether observed patterns generalize beyond Charlotte, NC. Additionally, integrating structured environmental compliance data (e.g., EPA records) would allow for stronger validation of whether high GSI scores correlate with actual environmental performance or are primarily reflective of marketing behavior.

Over time, this framework could also support comparative regional studies, enabling analysis of whether urban policy environments influence the prevalence or intensity of greenwashing signals in small business communication.

---

## Recommendations

## Future Research Recommendations

- Businesses with high positivity but low specificity should be further examined, since this combination may indicate emotionally appealing but vague sustainability messaging.
- Industry-specific patterns should be analyzed to determine whether certain sectors naturally produce higher GSI scores due to marketing style rather than actual intent to mislead.
- Future work should compare website claims with real-world environmental actions to better distinguish between genuine sustainability efforts and purely performative messaging.
- Regional comparisons should be conducted to see whether greenwashing signals vary based on local environmental regulations or public awareness.
- Time-based tracking of website content changes could help identify whether greenwashing increases during periods of heightened sustainability trends or policy pressure.

---

## Key Takeaway

This project demonstrates that natural language processing applied to SME websites can quantify greenwashing signals through a composite Greenwashing Severity Index (GSI), revealing how environmental messaging tends to reflect strategic communication patterns rather than verifiable sustainability performance.

## Limitations

- Only analyzes website claims, not images or design.
- JavaScript-heavy sites may return empty results.
- GSI measures online messaging, not actual business practices.
- Most SMEs operate outside EPA regulatory scope with no compliance records to cross-reference.

## Creator

This project was created by **Nikitha Hegde**. The work applies natural language processing techniques to evaluate environmental communication patterns in small business ecosystems, with a focus on transparency and accountability in sustainability messaging.

## License

Educational or research use.











