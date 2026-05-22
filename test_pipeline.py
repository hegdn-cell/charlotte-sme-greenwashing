import logging, re, warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import torch
import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = -1

# ── Load the 4 climate models ──
print("Downloading models (one time only, 30-60 sec)...")
tok_climate = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-detector")
mod_climate = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-detector")

tok_claim = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-commitment")
mod_claim = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-commitment")

tok_spec = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-specificity")
mod_spec = AutoModelForSequenceClassification.from_pretrained("climatebert/distilroberta-base-climate-specificity")

from transformers import pipeline
sent_pipe = pipeline("text-classification", model="climatebert/distilroberta-base-climate-sentiment", device=device)
print("Models ready.\n")

# ── Helper functions ──
def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits.numpy()[0]
    return float(softmax(logits)[1])

QUANT_PATTERNS = [r"\d+%", r"\d+ percent", r"\d+ tons?", r"reduced\s+(by\s+)?\d+", r"saved\s+\d+", r"diverted\s+\d+", r"since\s+(19|20)\d{2}", r"by\s+(19|20)\d{2}", r"target\s+of\s+\d+", r"\d+ kW", r"\d+ kWh", r"scope\s*[123]", r"tCO2", r"CO2e"]
SPEC_PATTERNS = [r"(install|installed|adopted|launched)\s+(solar|wind|efficient|led|compost)", r"(certif|accredit|iso\s*14001|b corp|leed|energy star)", r"(solar\s*panel|wind\s*turbine|renewable\s*energy)", r"(recycl\w*\s+program|zero\s+waste)", r"(policy|policie)\s+(environment|sustainab)"]

def classify_specificity(text):
    inputs = tok_spec(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = softmax(mod_spec(**inputs).logits.numpy()[0])
    if probs[1] > 0.5:
        tl = text.lower()
        if any(re.search(p, tl) for p in QUANT_PATTERNS): return "quantified"
        return "qualitative"
    return "vague"

# ── Two fake companies ──
SME_DATA = [
    {"name": "GreenDemo Charlotte (genuine SME)", "text": "At GreenDemo Charlotte we are deeply committed to environmental sustainability. We believe every business has a responsibility to reduce its ecological footprint. Our company has implemented a comprehensive recycling program that diverts 85% of our waste from landfill. Since 2022 we have reduced our energy consumption by 32% through LED retrofits and smart HVAC controls. We also installed a 50 kW solar array on our facility roof generating approximately 65,000 kWh annually. All of our packaging is made from 100% recycled materials and is fully compostable. We are proud to be a certified B Corporation and hold ISO 14001 certification. Our electric vehicle fleet now accounts for 40% of our delivery vehicles with a target of 100% by 2030. Climate change is the defining challenge of our generation and we are doing our part to address it."},
    {"name": "PretendGreen Charlotte (greenwasher)", "text": "PretendGreen is a Charlotte-based company that cares about the environment. We are committed to sustainability and green initiatives. Our team works hard to protect the planet for our children and future generations. We believe in doing good while doing business. Our environmental policy guides everything we do. We support local environmental organizations and participate in community clean-up events. We are always looking for new ways to be more environmentally friendly. Together we can make a difference for our planet. Sustainability is at the heart of our mission and we are passionate about protecting the environment."}
]

# ── Run pipeline on each ──
for sme in SME_DATA:
    print("=" * 60)
    print(sme["name"])
    print("=" * 60)

    # Split into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", sme["text"]) if len(s.strip()) > 15]

    # Stage 2: Climate detection
    climate_results = [(predict(mod_climate, tok_climate, s), s) for s in sentences]
    climate_sents = [(prob, s) for prob, s in climate_results if prob >= 0.5]

    # Stage 3: Claim detection
    claim_results = [(predict(mod_claim, tok_claim, s), s) for _, s in climate_sents]
    claim_sents = [(prob, s) for prob, s in claim_results if prob >= 0.5]

    # Stages 4-5: Specificity + sentiment
    specs = [classify_specificity(s) for _, s in claim_sents]
    sentiments = sent_pipe([s[:512] for _, s in climate_sents])

    # Aggregate
    n_total = len(sentences)
    n_climate = len(climate_sents)
    n_claims = len(claim_sents)
    density = n_claims / n_total if n_total else 0
    n_vague = specs.count("vague")
    vagueness = n_vague / n_claims if n_claims else 0
    pos_scores = []
    for r in sentiments:
        label = r["label"].lower()
        if label == "opportunity": pos_scores.append(1.0)
        elif label == "neutral": pos_scores.append(0.5)
        else: pos_scores.append(0.0)
    avg_pos = np.mean(pos_scores) if pos_scores else 0.5

    # GSI
    density_norm = min(density * 100, 30) / 30.0
    gsi = round(0.30 * density_norm + 0.35 * avg_pos + 0.35 * vagueness, 4)

    print(f"  Total sentences:            {n_total}")
    print(f"  Climate sentences:          {n_climate}")
    print(f"  Environmental claims:       {n_claims}")
    print(f"  Claim density:              {density:.1%}")
    print(f"  Avg sentiment positivity:   {avg_pos:.2f}")
    print(f"  Vagueness ratio:            {vagueness:.0%}")
    print(f"  GSI (Greenwashing Severity): {gsi:.4f}")
    print()

    for i, (_, s) in enumerate(claim_sents[:5]):
        print(f"    [{specs[i]:12s}] {s[:90]}...")
    print()

print("Done! Higher GSI = stronger greenwashing signal.")
input("Press Enter to exit...")
