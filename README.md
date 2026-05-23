# Charlotte SME Greenwashing Detection

A natural language processing pipeline that detects greenwashing signals on small business websites. Built for a study of 38 Charlotte, NC SMEs using ClimateBERT.

## What it does

1. Scrapes text from small business websites
2. Runs four ClimateBERT models to detect:
   - Climate-related content
   - Environmental commitments
   - Commitment specificity
   - Sentiment (positive/negative)
3. Calculates a **Greenwashing Severity Index (GSI)** for each business

## The GSI formula

```python
GSI = (density × 0.30) + (positivity × 0.35) + (vagueness × 0.35)
