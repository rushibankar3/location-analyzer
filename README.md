# 📍 Location Review Analyzer

An AI-powered tool that scrapes real Google Maps reviews for any location and produces a deep, evidence-backed analysis using a two-model AI pipeline with deterministic Python scoring.

---

## How It Works

```
Google Maps URL or Place Name
          ↓
    Apify Scraper
    (real reviews)
          ↓
  ┌───────┴───────┐
  ↓               ↓
Model A         Model A
Sentiment      Guardrail
(llama-70B)    (llama-8B)
  ↓               ↓
  └───────┬───────┘
          ↓
       Model B
    Verification
     (llama-8B)
          ↓
  Python Scoring
  (deterministic)
          ↓
   Final Verdict
```

### Stage-by-Stage Breakdown

| Stage | Model | Task |
|-------|-------|------|
| 1 | `llama-3.3-70b-versatile` | Deep sentiment — aspects, keywords, themes, trends |
| 2 | `llama-3.1-8b-instant` | Guardrail — fake review detection, trust scoring |
| 3 | `llama-3.1-8b-instant` | Independent verification — cross-checks Stage 1 output |
| 4 | Python (deterministic) | Final score from 7 weighted components |
| 5 | `llama-3.1-8b-instant` | Explanation — pros, cons, visitor tips |

**AI models never decide the verdict or score.** Python scoring is hard-enforced. Models only surface evidence and flag corrections.

---

## Features

- **Search by name** (city + place) or paste a **Google Maps URL**
- Scrapes up to 100 real Google reviews via Apify
- Batched sentiment analysis (handles any review count)
- Aspect scoring: food, service, ambience, value, cleanliness, crowd
- Fake review detection and trust scoring
- Model B cross-validates Model A — corrections only applied with review evidence
- Hallucination detection across both models
- Temporal trend analysis (recent vs older reviews)
- Crowd profile and accessibility notes
- Contradiction detection across reviews
- Interactive Streamlit dashboard with charts and export

---

## Prerequisites

- Python 3.9+
- [Apify](https://apify.com) account — free tier works (scrapes reviews)
- [Groq](https://console.groq.com) account — free tier (100K–500K tokens/day)

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/rushibankar3/location-analyzer.git
cd location-analyzer
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure API keys**

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
APIFY_API_TOKEN=your_apify_token_here
GROQ_API_KEY=your_groq_api_key_here
```

**4. Run the app**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Getting API Keys

### Apify (free)
1. Sign up at [apify.com](https://apify.com)
2. Go to **Settings → Integrations → API tokens**
3. Copy your token

### Groq (free)
1. Sign up at [console.groq.com](https://console.groq.com)
2. Go to **API Keys → Create API Key**
3. Copy your key

Free tier limits:
- `llama-3.3-70b-versatile`: 100,000 tokens/day
- `llama-3.1-8b-instant`: 500,000 tokens/day

---

## Configuration

All configuration is via `.env`. The defaults work out of the box — only the API keys need to be set.

```env
# Required
APIFY_API_TOKEN=your_apify_token
GROQ_API_KEY=your_groq_api_key

# Model assignments (defaults shown)
STRONG_MODEL=llama-3.3-70b-versatile   # Model A sentiment
FAST_MODEL=llama-3.1-8b-instant        # Model A guardrail + explanation
VERIFIER_MODEL=llama-3.1-8b-instant    # Model B verification

# Set to true to block results if Model B is unavailable
REQUIRE_VERIFICATION=false
```

---

## Output

The dashboard shows:

- **Executive Verdict** — HIGHLY RECOMMENDED / RECOMMENDED / VISIT WITH CAUTION / NOT RECOMMENDED
- **Visit Score** — 0–10, calculated by Python from 7 components
- **Pros & Cons** — evidence-backed, with review IDs
- **Aspect Scores** — radar chart across food, service, value, etc.
- **Sentiment Trends** — recent vs older reviews
- **Guardrail Report** — fake detection, trust score, linguistic analysis
- **Model B Verification** — field-by-field accuracy check with corrections
- **Raw Data & Export** — full JSON download

---

## Project Structure

```
location-analyzer/
├── app.py              # Streamlit dashboard
├── location.py         # Core pipeline — scraping, AI, scoring
├── requirements.txt    # Python dependencies
├── .env.example        # Configuration template
└── .gitignore          # Excludes .env and secrets
```

---

## Scoring Components

The final visit score is computed entirely in Python from these weighted inputs:

| Component | Weight | Source |
|-----------|--------|--------|
| Sentiment score | 25% | Model A |
| Google rating | 20% | Apify |
| Trust score | 15% | Model A guardrail |
| Review quality | 10% | Model A guardrail |
| Aspect average | 15% | Model A |
| Fake review penalty | 10% | Model A guardrail |
| Verification confidence | 5% | Model B |

No AI model can override this calculation.

---

## Rate Limits

If you hit Groq's free tier limits (`429` errors), wait for the daily quota to reset or get a new API key from [console.groq.com](https://console.groq.com).

To avoid `413` (request too large) errors on `llama-3.1-8b-instant` (6K TPM limit), the guardrail analysis automatically caps at 18 reviews with 120 characters per review.

---

## License

MIT
