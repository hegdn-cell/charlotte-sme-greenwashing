import csv, re, warnings, time
warnings.filterwarnings("ignore")

import requests
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from bs4 import BeautifulSoup

device = -1

print("Loading models...")
tok_climate = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-detector")
mod_climate = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-detector")
tok_claim = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-commitment")
mod_claim = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-commitment")
tok_spec = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-specificity")
mod_spec = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-specificity")
sent_pipe = pipeline("text-classification", model="climatebert/distilroberta-base-climate-sentiment", device=device)
print("Models ready.\n")

QUANT_PATTERNS = [r"\d+%",r"\d+ percent",r"\d+ tons?",r"reduced\s+(by\s+)?\d+",r"saved\s+\d+",r"diverted\s+\d+",r"since\s+(19|20)\d{2}",r"by\s+(19|20)\d{2}",r"target\s+of\s+\d+",r"\d+\s*kW",r"\d+\s*kWh",r"scope\s*[123]",r"tCO2",r"CO2e"]

def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        return float(softmax(model(**inputs).logits.numpy()[0])[1])

def classify_specificity(text):
    inputs = tok_spec(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = softmax(mod_spec(**inputs).logits.numpy()[0])
    if probs[1] > 0.5:
        tl = text.lower()
        if any(re.search(p, tl) for p in QUANT_PATTERNS): return "quantified"
        return "qualitative"
    return "vague"

def get_homepage_text(url):
    """Simple scraper using requests + BeautifulSoup."""
    try:
        resp = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove junk tags
        for tag in soup(["script","style","nav","footer","header","noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    except Exception as e:
        return ""

# Read SMEs
smes = []
with open("smes.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        smes.append(row)

if not smes:
    print("No SMEs found in smes.csv. Make sure the file has a header row: business_name,url")
    exit()

results = []
for i, sme in enumerate(smes):
    name = sme.get("business_name", "").strip()
    url = sme.get("url", "").strip()
    if not url: continue

    print(f"[{i+1}/{len(smes)}] {name}")
    
    # Clean URL: remove tracking parameters
    clean_url = url.split("?")[0].split("#")[0]
    
    text = get_homepage_text(clean_url)
    if not text or len(text) < 200:
        # Try with the original URL if clean version failed
        text = get_homepage_text(url)
    
    if not text or len(text) < 200:
        print(f"  Could not extract text\n")
        results.append({"business_name": name, "url": url, "n_sentences": 0, "n_climate": 0, "n_claims": 0, "density": "", "positivity": "", "vagueness": "", "gsi": ""})
        continue

    print(f"  {len(text)} chars extracted")

    # Split into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if len(s.strip()) > 25]
    
    if len(sentences) < 3:
        print(f"  Too few sentences\n")
        results.append({"business_name": name, "url": url, "n_sentences": len(sentences), "n_climate": 0, "n_claims": 0, "density": "", "positivity": "", "vagueness": "", "gsi": ""})
        continue

    # Climate detection
    climate_sents = []
    for s in sentences:
        prob = predict(mod_climate, tok_climate, s)
        if prob >= 0.5:
            climate_sents.append(s)

    if not climate_sents:
        print(f"  No climate content (checked {len(sentences)} sentences)\n")
        results.append({"business_name": name, "url": url, "n_sentences": len(sentences), "n_climate": 0, "n_claims": 0, "density": "", "positivity": "", "vagueness": "", "gsi": 0})
        continue

    # Claim detection
    claim_sents = []
    for s in climate_sents:
        prob = predict(mod_claim, tok_claim, s)
        if prob >= 0.5:
            claim_sents.append(s)

    if not claim_sents:
        print(f"  Climate: {len(climate_sents)}  No environmental claims\n")
        results.append({"business_name": name, "url": url, "n_sentences": len(sentences), "n_climate": len(climate_sents), "n_claims": 0, "density": "", "positivity": "", "vagueness": "", "gsi": 0})
        continue

    # Specificity
    specs = [classify_specificity(s) for s in claim_sents]

    # Sentiment
    sentiments = sent_pipe([s[:512] for s in climate_sents])
    pos_scores = []
    for r in sentiments:
        label = r["label"].lower()
        if label == "opportunity": pos_scores.append(1.0)
        elif label == "neutral": pos_scores.append(0.5)
        else: pos_scores.append(0.0)
    avg_pos = np.mean(pos_scores) if pos_scores else 0.5

    # GSI
    density = len(claim_sents) / len(sentences)
    n_vague = specs.count("vague")
    vagueness = n_vague / len(claim_sents) if claim_sents else 0
    density_norm = min(density * 100, 30) / 30.0
    gsi = round(0.30 * density_norm + 0.35 * avg_pos + 0.35 * vagueness, 4)

    print(f"  {len(climate_sents)} climate sentences, {len(claim_sents)} claims, vagueness {vagueness:.0%}, GSI {gsi}")
    for i, s in enumerate(claim_sents[:3]):
        print(f"    [{specs[i]:12s}] {s[:80]}...")
    print()

    results.append({"business_name": name, "url": url, "n_sentences": len(sentences), "n_climate": len(climate_sents), "n_claims": len(claim_sents), "density": f"{density:.1%}", "positivity": f"{avg_pos:.2f}", "vagueness": f"{vagueness:.0%}", "gsi": gsi})

df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print(f"\nDone! Results saved to results.csv")
print(f"SMEs analyzed: {len(results)}")
if "gsi" in df.columns and df["gsi"].notna().sum() > 0:
    df_num = df[df["gsi"].notna() & (df["gsi"] != "")]
    df_num["gsi"] = pd.to_numeric(df_num["gsi"])
    top = df_num.nlargest(10, "gsi")
    print(f"\nTop 10 highest GSI (strongest greenwashing signals):")
    for _, r in top.iterrows():
        print(f"  {r['business_name']:30s} GSI {r['gsi']:.4f}")
input("Press Enter to exit...")
