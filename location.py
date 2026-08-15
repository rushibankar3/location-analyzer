#!/usr/bin/env python3
"""
Location Review AI Analyzer
Pipeline: Apify (scrape) → Groq (sentiment) → Groq (guardrail) → Groq (recommend)
All APIs are free tier — no credit card required.
"""

import os, json, time, sys, re
import urllib.request
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

try:
    from apify_client import ApifyClient
    from groq import Groq
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("Run: pip install apify-client groq python-dotenv rich")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────────────────

APIFY_TOKEN  = os.getenv("APIFY_API_TOKEN") or os.getenv("APify_API_TOKEN", "")
GROQ_KEY     = os.getenv("GROQ_API_KEY", "")

# Free Apify actor: Google Maps scraper (5 USD/month credit on free plan)
ACTOR_ID     = "compass/crawler-google-places"

# Groq free models — if these are deprecated, check console.groq.com/models
FAST_MODEL   = "llama-3.1-8b-instant"       # sentiment pass (speed matters)
STRONG_MODEL = "llama-3.3-70b-versatile"    # guardrail + recommendation (quality matters)

console = Console()

if not APIFY_TOKEN or not GROQ_KEY:
    console.print("[red bold]ERROR:[/red bold] Missing API keys.")
    console.print("Create a .env file with APIFY_API_TOKEN and GROQ_API_KEY")
    sys.exit(1)

apify  = ApifyClient(APIFY_TOKEN)
groq   = Groq(api_key=GROQ_KEY)


# ── Shared Groq helper ───────────────────────────────────────────────────────

def call_groq(prompt: str, model: str = FAST_MODEL, max_tokens: int = 4096) -> dict:
    """Call Groq and parse the JSON response. Retries once on parse failure."""
    for attempt in range(2):
        try:
            res = groq.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Respond ONLY with valid JSON. No markdown, no explanation."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            raw = res.choices[0].message.content.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

            # Try direct parse first
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

            # Find the outermost {...} block
            start = raw.find("{")
            end   = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1])
                except json.JSONDecodeError:
                    pass

            # Last resort: fix truncated JSON by closing open structures
            if start != -1:
                fragment = raw[start:]
                # Count unclosed braces/brackets and close them
                opens = fragment.count("{") - fragment.count("}")
                arr_opens = fragment.count("[") - fragment.count("]")
                # Close any open string (odd number of unescaped quotes)
                in_str = False
                for ch in fragment:
                    if ch == '"':
                        in_str = not in_str
                if in_str:
                    fragment += '"'
                fragment += "]" * max(0, arr_opens) + "}" * max(0, opens)
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    pass

            if attempt == 0:
                time.sleep(1)
                continue

            console.print("[yellow]⚠  JSON parse failed — returning empty result[/yellow]")
            return {}

        except Exception as e:
            console.print(f"[red]Groq error: {e}[/red]")
            return {}


# ── Geocoder helpers (Photon / komoot) ──────────────────────────────────────

def _photon_request(params: dict, timeout: int = 6) -> dict:
    """Low-level helper: hit Photon API and return parsed JSON."""
    url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def geocode_city(city: str) -> tuple[float, float] | tuple[None, None]:
    """
    Return (lat, lon) for the given city name using Photon.
    Returns (None, None) on failure.
    """
    if not city or not city.strip():
        return None, None
    try:
        data = _photon_request({"q": city.strip(), "limit": 1, "lang": "en"})
        feats = data.get("features", [])
        if feats:
            coords = feats[0].get("geometry", {}).get("coordinates", [])
            if len(coords) == 2:
                return coords[1], coords[0]   # (lat, lon)
    except Exception as e:
        console.print(f"[yellow]City geocode error: {e}[/yellow]")
    return None, None


def get_place_suggestions(query: str, city: str = "", limit: int = 7) -> list[dict]:
    """
    Return up to `limit` place suggestions for `query`, biased towards `city`
    when provided, using the free Photon geocoder (photon.komoot.io).
    No API key required.

    Strategy:
      1. If city given, append it to the query so Photon ranks city-local results first.
      2. Also geocode the city → pass lat/lon bias.
      3. Soft-filter: prefer results whose location text contains the city name,
         but keep top results anyway so the list is never empty.

    Each result dict contains:
        display_name  – human-readable label for the dropdown
        search_name   – concise "Place, City, Country" for Apify
        lat / lon     – coordinates
    """
    if not query or len(query.strip()) < 2:
        return []

    # Embed city into the query itself — most reliable way to bias Photon
    q = f"{query.strip()} {city.strip()}" if city.strip() else query.strip()
    params: dict = {"q": q, "limit": limit * 2, "lang": "en"}

    # Also pass lat/lon bias when we can geocode the city
    city_lat, city_lon = None, None
    if city.strip():
        city_lat, city_lon = geocode_city(city)
        if city_lat is not None:
            params["lat"] = city_lat
            params["lon"] = city_lon

    try:
        data = _photon_request(params)
    except Exception as e:
        console.print(f"[yellow]Photon autocomplete error: {e}[/yellow]")
        return []

    city_lower = city.strip().lower()
    preferred, fallback = [], []
    seen: set[str] = set()

    for feat in data.get("features", []):
        props  = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None])

        name    = props.get("name", "").strip()
        p_city  = (props.get("city") or props.get("town") or
                   props.get("village") or props.get("county") or "").strip()
        state   = props.get("state", "").strip()
        country = props.get("country", "").strip()

        if not name:
            continue

        # Display label: "Place, City, State, Country"
        label_parts = [p for p in [name, p_city, state, country] if p]
        display     = ", ".join(label_parts)

        # Concise search name for Apify
        search_parts = [p for p in [name, p_city, country] if p]
        search_name  = ", ".join(search_parts) if search_parts else display

        if search_name in seen:
            continue
        seen.add(search_name)

        entry = {
            "display_name": display,
            "search_name":  search_name,
            "lat":          coords[1],
            "lon":          coords[0],
            "type":         props.get("type", ""),
            "category":     props.get("osm_key", ""),
        }

        # Soft city match — prefer results that mention the city, keep rest as fallback
        location_text = f"{name} {p_city} {state} {country}".lower()
        if city_lower and city_lower in location_text:
            preferred.append(entry)
        else:
            fallback.append(entry)

    # Return city-matched results first, pad with fallback if needed
    combined = preferred + fallback
    return combined[:limit]


def get_city_suggestions(query: str, limit: int = 6) -> list[str]:
    """
    Return city name suggestions for the city search box using Photon.
    Filters to results that look like cities/towns.
    """
    if not query or len(query.strip()) < 2:
        return []

    params = {"q": query.strip(), "limit": limit * 3, "lang": "en"}
    try:
        data = _photon_request(params)
    except Exception as e:
        console.print(f"[yellow]City suggestions error: {e}[/yellow]")
        return []

    city_types = {"city", "town", "village", "municipality", "borough",
                  "suburb", "district", "county", "state", "administrative"}
    results, seen = [], set()

    for feat in data.get("features", []):
        props   = feat.get("properties", {})
        name    = props.get("name", "").strip()
        country = props.get("country", "").strip()
        osm_val = props.get("osm_value", "").lower()
        osm_key = props.get("osm_key", "").lower()

        # Keep entries that are place/boundary types (cities, towns, etc.)
        if osm_key not in ("place", "boundary") and osm_val not in city_types:
            continue

        label = f"{name}, {country}" if country else name
        if label in seen or not name:
            continue
        seen.add(label)
        results.append(label)

        if len(results) >= limit:
            break

    return results


# ── URL expander ─────────────────────────────────────────────────────────────

def expand_maps_url(url: str) -> str:
    """
    Follow redirects on a short or full Google Maps URL and return the
    final expanded URL (e.g. https://www.google.com/maps/place/...).
    Falls back to the original string if anything goes wrong.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LocationAnalyzer/1.0)"},
        )
        # We don't need the body — just follow the redirect chain
        with urllib.request.urlopen(req, timeout=8) as resp:
            final_url = resp.url          # urllib follows all redirects automatically
        console.print(f"[dim]🔗 Expanded URL: {final_url[:80]}…[/dim]")
        return final_url
    except Exception as e:
        console.print(f"[yellow]URL expand warning: {e} — using original URL[/yellow]")
        return url


# ── MODULE 1: Apify Scraper ──────────────────────────────────────────────────

def scrape_reviews(location: str, max_reviews: int = 40) -> tuple[list, dict]:
    """
    Scrape Google Maps reviews for `location` using Apify.

    Strategy (most-reliable first):
      1. URL passed -> expand short URLs -> use startUrls  (most reliable)
      2. place_id:ChIJ... -> pass directly in searchStringsArray
      3. Free text -> geocode with Photon to get lat/lon, pass to Apify with
         zoom=16 so it lands on the exact location; picks the result with
         the most reviews. Falls back to plain text search if geocoding fails.
    """
    console.print(f"\n[bold cyan]📡  Step 1 — Scraping:[/bold cyan] {location}")

    is_url      = location.strip().startswith("http")
    is_place_id = location.strip().lower().startswith("place_id:")

    # Path A: URL
    if is_url:
        expanded = expand_maps_url(location.strip())
        run_input = {
            "startUrls":           [{"url": expanded}],
            "maxCrawledPlaces":    1,
            "maxReviews":          max_reviews,
            "reviewsSort":         "newest",
            "language":            "en",
            "includeOpeningHours": True,
        }

    # Path B: explicit Place ID
    elif is_place_id:
        run_input = {
            "searchStringsArray":  [location.strip()],
            "maxCrawledPlaces":    1,
            "maxReviews":          max_reviews,
            "reviewsSort":         "newest",
            "language":            "en",
            "includeOpeningHours": True,
        }

    # Path C: free-text name -> geocode the PLACE itself -> use its coordinates
    # as a Google Maps URL so Apify lands on exactly the right page
    else:
        # Try to get the exact place coordinates from Photon
        # Split city from location if format is "Place, City"
        parts = location.strip().split(",", 1)
        place_query = location.strip()

        sugg = get_place_suggestions(place_query, city="", limit=1)

        if sugg and sugg[0].get("lat") and sugg[0].get("lon"):
            place_lat = sugg[0]["lat"]
            place_lon = sugg[0]["lon"]
            place_name = sugg[0]["search_name"]
            console.print(f"[dim]🌐 Place geocoded: {place_name} -> lat={place_lat:.5f}, lon={place_lon:.5f}[/dim]")

            # Build a Google Maps search URL pinned to the exact coordinates
            # This is far more reliable than a text search
            maps_url = (
                f"https://www.google.com/maps/search/"
                f"{urllib.parse.quote(place_query)}"
                f"/@{place_lat},{place_lon},17z"
            )
            run_input = {
                "startUrls":           [{"url": maps_url}],
                "maxCrawledPlaces":    3,
                "maxReviews":          max_reviews,
                "reviewsSort":         "newest",
                "language":            "en",
                "includeOpeningHours": True,
            }
        else:
            # Photon couldn't geocode — fall back to plain text search with
            # a wider net and pick the best result by review count
            console.print("[yellow]⚠ Place geocoding failed — using plain text search[/yellow]")
            run_input = {
                "searchStringsArray":  [location.strip()],
                "maxCrawledPlaces":    5,
                "maxReviews":          max_reviews,
                "reviewsSort":         "newest",
                "language":            "en",
                "includeOpeningHours": True,
            }

    try:
        run = apify.actor(ACTOR_ID).call(run_input=run_input)
        if isinstance(run, dict):
            dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")
        else:
            dataset_id = getattr(run, "default_dataset_id", getattr(run, "defaultDatasetId", None))
            if not dataset_id and hasattr(run, "get"):
                dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            raise ValueError(f"Could not retrieve dataset ID from run: {run}")

        all_places = list(apify.dataset(dataset_id).iterate_items())

        if not all_places:
            console.print("[red]✗  No places returned by Apify.[/red]")
            return [], {}

        # Pick the place with the most reviews — most likely the intended one
        best = max(all_places, key=lambda p: p.get("reviewsCount") or 0)

        place_info = {
            "name":               best.get("title") or location,
            "address":            best.get("address") or "",
            "google_score":       best.get("totalScore") or 0,
            "review_count":       best.get("reviewsCount") or 0,
            "category":           best.get("categoryName") or "",
            "subtypes":           best.get("categories") or [],
            "description":        best.get("description") or "",
            "price":              best.get("price") or "",
            "opening_hours":      best.get("openingHours") or [],
            "website":            best.get("website") or "",
            "phone":              best.get("phone") or "",
            "location_type":      best.get("locationType") or "",
            "permanently_closed": best.get("permanentlyClosed") or False,
            "temporarily_closed": best.get("temporarilyClosed") or False,
        }

        reviews = []
        for r in best.get("reviews") or []:
            text = (r.get("text") or "").strip()
            if text:
                reviews.append({
                    "author": r.get("name") or "Anonymous",
                    "rating": r.get("stars") or 3,
                    "text":   text[:800],
                    "date":   r.get("publishedAtDate") or "",
                    "likes":  r.get("likesCount") or 0,
                })

        name = place_info.get("name", location)
        console.print(f"[green]✓  Scraped {len(reviews)} reviews for \"{name}\"[/green]")
        return reviews, place_info

    except Exception as e:
        console.print(f"[red]✗  Apify scraping failed: {e}[/red]")
        return [], {}

def analyze_sentiment(reviews: list) -> dict:
    """
    Deep sentiment analysis using Groq.

    Covers:
      • Per-review scoring with emotion intensity label
      • Aspect-based scores: Food, Service, Ambience, Value, Cleanliness,
        Accessibility, Crowd/Wait-time  (0–10 each, N/A if not applicable)
      • Recurring theme extraction with frequency count + representative quote
      • Temporal trend: sentiment of recent (last third) vs older reviews
      • Crowd profile: what kind of visitors dominate the reviews
      • Positive / negative keyword lists
      • Overall sentiment + weighted sentiment score (-1 to +1)
    """
    console.print("\n[bold cyan]🧠  Step 2 — Deep sentiment analysis (Groq)[/bold cyan]")

    sample = reviews[:20]  # Reduced from 30 to keep within token limits

    # ── Pre-compute temporal split ──
    third = max(1, len(sample) // 3)
    recent_reviews  = sample[:third]
    older_reviews   = sample[third:]

    def fmt(subset, label):
        return f"\n=== {label} ({len(subset)} reviews) ===\n" + "\n---\n".join(
            f"[R{i+1}] ⭐{r['rating']}/5  date:{r.get('date','?')[:10]}\n{r['text'][:300]}"
            for i, r in enumerate(subset)
        )

    reviews_block = fmt(recent_reviews, "RECENT") + "\n" + fmt(older_reviews, "OLDER")

    prompt = f"""You are an expert review analyst. Perform a DEEP multi-dimensional sentiment analysis
on the following location reviews. Return ONLY valid JSON matching this EXACT structure:

{{
  "per_review": [
    {{
      "id": 1,
      "sentiment": "Positive|Negative|Neutral|Mixed",
      "score": 0.75,
      "emotion": "Excited|Happy|Satisfied|Neutral|Disappointed|Frustrated|Angry",
      "intensity": "Low|Medium|High",
      "key_phrase": "one crisp sentence capturing the review"
    }}
  ],

  "aspect_scores": {{
    "food_quality":    {{"score": 7.5, "reviews_mentioning": 5, "summary": "brief note or null"}},
    "service":         {{"score": 8.0, "reviews_mentioning": 10, "summary": "brief note or null"}},
    "ambience":        {{"score": 6.5, "reviews_mentioning": 8, "summary": "brief note or null"}},
    "value_for_money": {{"score": 7.0, "reviews_mentioning": 6, "summary": "brief note or null"}},
    "cleanliness":     {{"score": 8.5, "reviews_mentioning": 4, "summary": "brief note or null"}},
    "accessibility":   {{"score": 7.0, "reviews_mentioning": 3, "summary": "brief note or null"}},
    "crowd_wait_time": {{"score": 5.5, "reviews_mentioning": 7, "summary": "brief note or null"}}
  }},

  "themes": [
    {{
      "name": "theme name",
      "sentiment": "Positive|Negative|Mixed",
      "frequency": 5,
      "representative_quote": "exact short quote from a review",
      "evidence": "1-sentence synthesis of what reviewers say about this theme"
    }}
  ],

  "temporal_trend": {{
    "recent_sentiment":  "Positive|Negative|Neutral|Mixed",
    "recent_score":      0.7,
    "older_sentiment":   "Positive|Negative|Neutral|Mixed",
    "older_score":       0.6,
    "trend":             "Improving|Declining|Stable",
    "trend_explanation": "1-sentence reason for the trend"
  }},

  "crowd_profile": {{
    "dominant_visitor_type": "Families|Couples|Solo travellers|Business visitors|Tourists|Locals|Mixed",
    "mention_evidence": "brief evidence from reviews",
    "accessibility_notes": "any mentions of disability, elderly, children accessibility"
  }},

  "emotion_distribution": {{
    "Excited": 0, "Happy": 0, "Satisfied": 0,
    "Neutral": 0, "Disappointed": 0, "Frustrated": 0, "Angry": 0
  }},

  "positive_keywords": ["keyword1", "keyword2"],
  "negative_keywords": ["keyword1", "keyword2"],
  "standout_positive_quote": "the single most positive review sentence verbatim",
  "standout_negative_quote": "the single most critical review sentence verbatim",

  "overall_sentiment": "Positive|Negative|Neutral|Mixed",
  "sentiment_score": 0.65,
  "emotional_tone": "Excited|Happy|Satisfied|Disappointed|Angry|Neutral",
  "review_diversity": "High|Medium|Low"
}}

IMPORTANT RULES:
- aspect scores: use null for score and 0 for reviews_mentioning if that aspect is never mentioned
- sentiment_score range: -1.0 (very negative) to +1.0 (very positive)
- all aspect scores range: 0–10
- frequency = how many of the analyzed reviews mention this theme
- Include at least 5 themes if the review count allows

REVIEWS:
{reviews_block}"""

    result = call_groq(prompt, model=STRONG_MODEL)

    # ── Always compute rating distribution from raw data ──
    counts = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}
    for r in reviews:
        k = str(max(1, min(5, int(r.get("rating", 3)))))
        counts[k] = counts.get(k, 0) + 1
    result["rating_counts"] = counts

    # ── Compute avg sentiment from per-review scores as a sanity fallback ──
    per = result.get("per_review", [])
    if per and not result.get("sentiment_score"):
        scores = [p.get("score", 0) for p in per if isinstance(p.get("score"), (int, float))]
        result["sentiment_score"] = round(sum(scores) / len(scores), 3) if scores else 0

    # ── Summary log ──
    s  = result.get("sentiment_score", 0)
    o  = result.get("overall_sentiment", "Unknown")
    tr = result.get("temporal_trend", {}).get("trend", "?")
    kw = ", ".join(result.get("positive_keywords", [])[:5]) or "—"
    console.print(
        f"[green]✓  Sentiment: {o}  |  Score: {s:.2f}  |  "
        f"Trend: {tr}  |  Top keywords: {kw}[/green]"
    )
    return result


# ── MODULE 3: Guardrail Analysis (Groq) ─────────────────────────────────────

def _heuristic_checks(reviews: list) -> dict:
    """
    Pure-Python pre-pass before the LLM call.
    Returns a dict of flags and stats that gets fed into the prompt.
    """
    n = len(reviews)
    flags = []

    if n == 0:
        return {"flags": ["No reviews to analyze"], "stats": {}}

    ratings   = [r["rating"] for r in reviews]
    texts     = [r["text"].lower()  for r in reviews]
    authors   = [r.get("author", "") for r in reviews]

    # 1. Rating distribution anomalies
    five_star_pct  = ratings.count(5) / n
    one_star_pct   = ratings.count(1) / n
    if five_star_pct > 0.75:
        flags.append(f"Unusually high 5-star ratio ({five_star_pct:.0%}) — potential rating manipulation")
    if one_star_pct > 0.40:
        flags.append(f"High 1-star ratio ({one_star_pct:.0%}) — possible targeted negative campaign")
    if len(set(ratings)) == 1:
        flags.append("All reviews share the exact same star rating — highly suspicious uniformity")

    # 2. Short review abuse (< 12 chars on 5-star)
    short_5star = [r for r in reviews if r["rating"] == 5 and len(r["text"]) < 12]
    if len(short_5star) / n > 0.30:
        flags.append(f"{len(short_5star)} very short 5-star reviews (< 12 chars) — likely filler boosts")

    # 3. Copy-paste / near-duplicate detection (Jaccard on word sets)
    duplicates = 0
    for i in range(min(n, 20)):
        for j in range(i + 1, min(n, 20)):
            w1 = set(texts[i].split())
            w2 = set(texts[j].split())
            union = w1 | w2
            if union:
                jaccard = len(w1 & w2) / len(union)
                if jaccard > 0.55:
                    duplicates += 1
    if duplicates >= 3:
        flags.append(f"{duplicates} near-duplicate review pairs detected (>55% word overlap) — copy-paste patterns")

    # 4. Repeated phrases across reviews (3+ word n-grams appearing in 4+ reviews)
    from collections import Counter
    ngram_counter: Counter = Counter()
    for txt in texts[:20]:
        words = txt.split()
        for k in range(len(words) - 2):
            ng = " ".join(words[k:k+3])
            if len(ng) > 8:
                ngram_counter[ng] += 1
    repeated_phrases = [ng for ng, cnt in ngram_counter.most_common(5) if cnt >= 4]
    if repeated_phrases:
        flags.append(f"Repeated phrases across reviews: {repeated_phrases[:3]} — coordinated review language")

    # 5. Single-review authors (sock-puppet signal)
    unique_authors    = len(set(authors))
    single_rev_ratio  = unique_authors / n if n > 0 else 1
    if single_rev_ratio < 0.7 and n > 8:
        flags.append(f"Low author diversity ({unique_authors} unique / {n} reviews) — possible sock-puppet accounts")

    # 6. Sample size warning
    if n < 6:
        flags.append("Too few reviews for high-confidence analysis — treat results with caution")

    # 7. Temporal clustering (many reviews in very short window)
    dates = [r.get("date", "")[:10] for r in reviews if r.get("date")]
    if len(dates) >= 5:
        unique_dates = len(set(dates))
        if unique_dates / len(dates) < 0.25:
            flags.append("Many reviews share the same date — possible coordinated review-bombing or boosting")

    avg_rating = sum(ratings) / n
    avg_len    = sum(len(t) for t in texts) / n

    stats = {
        "total_reviews":     n,
        "avg_rating":        round(avg_rating, 2),
        "five_star_pct":     round(five_star_pct, 2),
        "one_star_pct":      round(one_star_pct, 2),
        "avg_review_length": round(avg_len, 0),
        "unique_authors":    unique_authors,
        "duplicate_pairs":   duplicates,
        "repeated_phrases":  repeated_phrases[:3],
    }
    return {"flags": flags, "stats": stats}


def guardrail_analysis(reviews: list, sentiment: dict) -> dict:
    """
    Deep guardrail analysis using Groq (strong model) + Python heuristics.

    Covers:
      • Linguistic fingerprinting (copy-paste, templated language, n-gram repeats)
      • Reviewer behaviour signals (author diversity, temporal clustering)
      • Rating manipulation detection (star distribution anomalies)
      • Cross-review contradiction analysis (conflicting claims about same aspect)
      • Per-aspect credibility breakdown
      • Verified genuine positives AND genuine concerns with severity
      • Composite trust score (0–1) with transparent reasoning
    """
    console.print("\n[bold cyan]🛡   Step 3 — Deep guardrail analysis (Groq)[/bold cyan]")

    # ── Python heuristics first ──
    heuristics = _heuristic_checks(reviews)
    heuristic_flags = heuristics["flags"]
    heuristic_stats = heuristics["stats"]

    # ── Build review sample for LLM (include more context per review) ──
    review_sample = [
        {
            "id":     i + 1,
            "rating": r["rating"],
            "length": len(r["text"]),
            "author": r.get("author", "Anonymous"),
            "date":   r.get("date", "")[:10],
            "text":   r["text"][:400],
        }
        for i, r in enumerate(reviews[:25])
    ]

    prompt = f"""You are a senior review-integrity analyst with expertise in detecting fake, manipulated,
and biased reviews. Perform a DEEP forensic analysis of the reviews below.

=== SENTIMENT CONTEXT ===
{json.dumps({
    "overall":            sentiment.get("overall_sentiment"),
    "score":              sentiment.get("sentiment_score"),
    "positive_keywords":  sentiment.get("positive_keywords", [])[:10],
    "negative_keywords":  sentiment.get("negative_keywords", [])[:8],
    "themes":             sentiment.get("themes", [])[:5],
    "temporal_trend":     sentiment.get("temporal_trend", {}),
}, indent=2)}

=== PYTHON HEURISTIC FLAGS (pre-computed) ===
{json.dumps(heuristic_flags, indent=2)}

=== HEURISTIC STATISTICS ===
{json.dumps(heuristic_stats, indent=2)}

=== REVIEW SAMPLE (up to 25) ===
{json.dumps(review_sample, indent=2)}

Analyze ALL of the above and return ONLY this exact JSON — no markdown, no explanation:

{{
  "trust_score": 0.0,
  "credibility_score": 0.0,
  "fake_review_probability": 0.0,
  "review_quality": "High|Medium|Low",
  "bias_level": "Low|Medium|High",
  "bias_direction": "Positive|Negative|None",

  "linguistic_analysis": {{
    "templated_language_detected": true,
    "copy_paste_evidence": "brief description or null",
    "vocabulary_diversity": "High|Medium|Low",
    "writing_style_consistency": "Consistent (suspicious)|Varied (natural)|Mixed",
    "language_notes": "1-sentence observation"
  }},

  "rating_integrity": {{
    "distribution_natural": true,
    "anomalies": ["anomaly description"],
    "inflated_stars_estimate": 0,
    "suppressed_stars_estimate": 0,
    "adjusted_true_rating": 0.0,
    "rating_notes": "1-sentence assessment"
  }},

  "reviewer_behavior": {{
    "sock_puppet_risk": "Low|Medium|High",
    "coordinated_posting_risk": "Low|Medium|High",
    "evidence": "brief description of suspicious patterns or 'None detected'"
  }},

  "contradiction_analysis": [
    {{
      "aspect": "aspect name",
      "positive_claim": "what some reviewers say",
      "negative_claim": "what others say",
      "resolution": "which claim appears more credible and why"
    }}
  ],

  "aspect_credibility": {{
    "food_quality":    {{"credible": true,  "confidence": 0.9, "note": "brief note"}},
    "service":         {{"credible": true,  "confidence": 0.8, "note": "brief note"}},
    "ambience":        {{"credible": true,  "confidence": 0.7, "note": "brief note"}},
    "value_for_money": {{"credible": true,  "confidence": 0.8, "note": "brief note"}},
    "cleanliness":     {{"credible": true,  "confidence": 0.9, "note": "brief note"}},
    "crowd_wait_time": {{"credible": false, "confidence": 0.4, "note": "brief note"}}
  }},

  "genuine_positives": [
    {{
      "aspect":     "aspect name",
      "evidence":   "what real reviewers consistently say",
      "confidence": 0.85,
      "supporting_review_ids": [1, 3, 7]
    }}
  ],

  "genuine_concerns": [
    {{
      "aspect":   "concern name",
      "evidence": "what real reviewers consistently say",
      "severity": "Minor|Moderate|Major",
      "supporting_review_ids": [2, 5]
    }}
  ],

  "suspicious_patterns": ["pattern 1", "pattern 2"],
  "verified_facts": ["verified fact 1", "verified fact 2"],

  "guardrail_summary": "2-sentence honest assessment of overall review reliability and what to trust",
  "analyst_recommendation": "1 sentence on how much to trust these reviews"
}}

SCORING GUIDELINES:
- trust_score: 0.0 = completely fake, 1.0 = highly authentic
- credibility_score: overall data quality (0–1)
- fake_review_probability: 0.0 = no fakes, 1.0 = all fake
- adjusted_true_rating: your estimate of the real rating if fake reviews removed (out of 5)
- Be skeptical but fair. Flag real patterns, not coincidences."""

    result = call_groq(prompt, model=STRONG_MODEL)

    # Merge Python heuristic flags with LLM-detected patterns
    ai_flags  = result.get("suspicious_patterns", [])
    all_flags = list(dict.fromkeys(heuristic_flags + ai_flags))
    result["suspicious_patterns"]  = all_flags
    result["heuristic_stats"]      = heuristic_stats   # pass stats to dashboard

    ts  = result.get("trust_score", 0)
    rq  = result.get("review_quality", "?")
    fp  = result.get("fake_review_probability", 0)
    adj = result.get("rating_integrity", {}).get("adjusted_true_rating", "?")
    console.print(
        f"[green]✓  Trust: {ts:.0%}  |  Quality: {rq}  |  "
        f"Fake prob: {fp:.0%}  |  Adj. rating: {adj}/5[/green]"
    )
    return result


# ── MODULE 4: Recommendation Engine (Groq) ───────────────────────────────────

def generate_recommendation(
    location:   str,
    place_info: dict,
    reviews:    list,
    sentiment:  dict,
    guardrail:  dict,
) -> dict:
    """
    Uses Groq (strong Llama model) to synthesise all pipeline data into:
      - A visit_score (0–10)
      - A recommendation label: HIGHLY RECOMMENDED / RECOMMENDED /
                                VISIT WITH CAUTION / NOT RECOMMENDED
      - Pros, cons, visitor tips, full verdict
    """
    console.print("\n[bold cyan]💡  Step 4 — Recommendation (Groq)[/bold cyan]")

    avg_rating = sum(r["rating"] for r in reviews) / len(reviews) if reviews else 0

    context = {
        "location":            location,
        "place_name":          place_info.get("name", location),
        "category":            place_info.get("category", ""),
        "google_official":     place_info.get("google_score", 0),
        "reviews_analyzed":    len(reviews),
        "avg_scraped_rating":  round(avg_rating, 2),
        "overall_sentiment":   sentiment.get("overall_sentiment"),
        "sentiment_score":     sentiment.get("sentiment_score"),
        "emotional_tone":      sentiment.get("emotional_tone"),
        "top_positive_kw":     sentiment.get("positive_keywords", [])[:6],
        "top_negative_kw":     sentiment.get("negative_keywords", [])[:4],
        "themes":              [
            {"name": t.get("name"), "sentiment": t.get("sentiment"), "evidence": t.get("evidence", "")[:80]}
            for t in sentiment.get("themes", [])[:4]
            if isinstance(t, dict)
        ],
        "trust_score":         guardrail.get("trust_score"),
        "review_quality":      guardrail.get("review_quality"),
        "genuine_positives":   [
            {"aspect": p.get("aspect"), "evidence": (p.get("evidence",""))[:80]}
            for p in guardrail.get("genuine_positives", [])[:4]
            if isinstance(p, dict)
        ],
        "genuine_concerns":    [
            {"aspect": c.get("aspect"), "severity": c.get("severity"), "evidence": (c.get("evidence",""))[:80]}
            for c in guardrail.get("genuine_concerns", [])[:3]
            if isinstance(c, dict)
        ],
        "suspicious_patterns": guardrail.get("suspicious_patterns", [])[:3],
    }

    prompt = f"""You are an expert travel and experience advisor with 20 years of experience.
Based on the complete multi-stage AI analysis below, provide a final honest recommendation.

FULL ANALYSIS DATA:
{json.dumps(context, indent=2)}

Return EXACTLY this JSON:
{{
  "recommendation":   "HIGHLY RECOMMENDED|RECOMMENDED|VISIT WITH CAUTION|NOT RECOMMENDED",
  "confidence":       0.0,
  "visit_score":      0.0,
  "one_line_verdict": "25-word or less honest verdict",
  "full_verdict":     "3-4 sentence balanced assessment",
  "best_for":         ["visitor type 1", "visitor type 2"],
  "avoid_if":         ["avoid if reason 1"],
  "pros": [
    {{"point": "pro description", "weight": "High|Medium|Low"}}
  ],
  "cons": [
    {{"point": "con description", "weight": "High|Medium|Low"}}
  ],
  "visitor_tips":     ["tip 1", "tip 2", "tip 3"],
  "best_time":        "best time to visit, or null if not determinable",
  "data_reliability": "High|Medium|Low",
  "score_breakdown": {{
    "sentiment_score": 0.0,
    "rating_score":    0.0,
    "trust_score":     0.0,
    "composite":       0.0
  }}
}}"""

    result = call_groq(prompt, model=STRONG_MODEL)

    # Fallback composite score if AI didn't compute it
    bd = result.get("score_breakdown", {})
    if not bd.get("composite"):
        s = sentiment.get("sentiment_score", 0.5) * 10
        r = avg_rating * 2
        t = guardrail.get("trust_score", 0.5) * 10
        composite = round(s * 0.40 + r * 0.40 + t * 0.20, 1)
        result["score_breakdown"] = {
            "sentiment_score": round(s, 1),
            "rating_score":    round(r, 1),
            "trust_score":     round(t, 1),
            "composite":       composite,
        }
        if not result.get("visit_score"):
            result["visit_score"] = composite

    rec = result.get("recommendation", "Unknown")
    vs  = result.get("visit_score", 0)
    console.print(f"[green]✓  Verdict: {rec}  |  Visit score: {vs}/10[/green]")
    return result


# ── Report Display ────────────────────────────────────────────────────────────

VERDICT_COLOR = {
    "HIGHLY RECOMMENDED": "bold green",
    "RECOMMENDED":        "green",
    "VISIT WITH CAUTION": "yellow",
    "NOT RECOMMENDED":    "red",
}

def display_report(location, place_info, reviews, sentiment, guardrail, rec):
    console.print()
    console.rule("[bold yellow]📊  LOCATION REVIEW ANALYSIS REPORT[/bold yellow]")

    # Place info
    console.print(Panel(
        f"[bold]{place_info.get('name', location)}[/bold]\n"
        f"📍 {place_info.get('address') or 'Address unavailable'}\n"
        f"⭐ Google score: {place_info.get('google_score', 'N/A')}/5  "
        f"({place_info.get('review_count', len(reviews))} total reviews on Google)\n"
        f"🏷  Category: {place_info.get('category') or 'N/A'}",
        title="[cyan]Place info[/cyan]", expand=False,
    ))

    # Recommendation banner
    label = rec.get("recommendation", "UNKNOWN")
    color = VERDICT_COLOR.get(label, "white")
    bd    = rec.get("score_breakdown", {})
    console.print(Panel(
        f"[{color}]{label}[/{color}]\n\n"
        f"Visit score  [bold]{rec.get('visit_score', '?')}/10[/bold]   ·   "
        f"Confidence  [bold]{rec.get('confidence', 0):.0%}[/bold]   ·   "
        f"Data reliability  [bold]{rec.get('data_reliability', '?')}[/bold]\n\n"
        f"[italic]{rec.get('one_line_verdict', '')}[/italic]\n\n"
        f"Sentiment {bd.get('sentiment_score','?')}/10  ·  "
        f"Rating {bd.get('rating_score','?')}/10  ·  "
        f"Trust {bd.get('trust_score','?')}/10",
        title="[yellow]⚡ Recommendation[/yellow]", expand=False,
    ))

    # Pros / Cons table
    pros = rec.get("pros", [])
    cons = rec.get("cons", [])
    tbl  = Table(title="Pros vs cons", show_header=True, expand=False)
    tbl.add_column("✅ Pros", style="green", width=42)
    tbl.add_column("❌ Cons", style="red",   width=42)
    for i in range(min(max(len(pros), len(cons)), 5)):
        p_item = pros[i] if i < len(pros) else ""
        c_item = cons[i] if i < len(cons) else ""
        p = p_item.get("point", str(p_item)) if isinstance(p_item, dict) else str(p_item)
        c = c_item.get("point", str(c_item)) if isinstance(c_item, dict) else str(c_item)
        tbl.add_row(p, c)
    console.print(tbl)

    # Keywords & tone
    pos_kw = ", ".join(sentiment.get("positive_keywords", [])[:8]) or "—"
    neg_kw = ", ".join(sentiment.get("negative_keywords", [])[:5]) or "—"
    console.print(Panel(
        f"[green]Positive:[/green]  {pos_kw}\n"
        f"[red]Negative:[/red]  {neg_kw}\n"
        f"[yellow]Overall tone:[/yellow]  {sentiment.get('emotional_tone', 'N/A')}",
        title="[cyan]🔑 Sentiment keywords[/cyan]", expand=False,
    ))

    # Guardrail results
    flags = guardrail.get("suspicious_patterns", [])
    fp    = guardrail.get("fake_review_probability", 0)
    flag_text = f"⚠  {chr(10).join(flags)}" if flags else "✓  No suspicious patterns detected"
    console.print(Panel(
        f"Trust score [bold]{guardrail.get('trust_score', 0):.0%}[/bold]  ·  "
        f"Review quality [bold]{guardrail.get('review_quality', '?')}[/bold]  ·  "
        f"Fake probability [bold]{fp:.0%}[/bold]\n"
        f"Reviews analyzed: {len(reviews)}\n\n"
        + flag_text,
        title="[cyan]🛡  Guardrail results[/cyan]", expand=False,
    ))

    # Verified positives
    positives = guardrail.get("genuine_positives", [])
    if positives:
        pos_lines_list = []
        for p in positives[:5]:
            if isinstance(p, dict):
                asp = p.get('aspect', '')
                ev = p.get('evidence', '')
                conf = p.get('confidence', 0)
                pos_lines_list.append(f"• [bold]{asp}[/bold] — {ev}  (confidence {conf:.0%})")
            else:
                pos_lines_list.append(f"• {p}")
        pos_lines = "\n".join(pos_lines_list)
        console.print(Panel(pos_lines, title="[cyan]✅ Verified genuine positives[/cyan]", expand=False))

    # Visitor tips
    tips = rec.get("visitor_tips", [])
    if tips:
        console.print(Panel(
            "\n".join(f"• {t}" for t in tips[:5]),
            title="[cyan]💡 Visitor tips[/cyan]", expand=False,
        ))

    # Full verdict
    console.print(Panel(
        rec.get("full_verdict", "No verdict generated."),
        title="[yellow]📝 Full verdict[/yellow]", expand=False,
    ))

    console.rule("[bold yellow]End of report[/bold yellow]")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def analyze(location: str, max_reviews: int = 30, progress_callback=None) -> dict:
    """Run the complete 4-stage pipeline for a given location name."""

    t0 = time.time()
    console.print(Panel(
        f"[bold]Location:[/bold]     {location}\n"
        f"[bold]Max reviews:[/bold]  {max_reviews}\n"
        f"[bold]Pipeline:[/bold]     Apify → Groq (sentiment) → Groq (guardrail) → Groq (recommendation)\n"
        f"[bold]Cost:[/bold]         Apify free tier  +  Groq free tier  =  $0",
        title="[bold blue]🌐 Location Review AI Analyzer[/bold blue]",
        expand=False,
    ))

    # Stage 1: Scrape
    if progress_callback:
        progress_callback(1, "Scraping Reviews", f"Scraping Google Maps reviews for '{location}' via Apify...")
    reviews, place_info = scrape_reviews(location, max_reviews)
    if not reviews:
        console.print("[red]No reviews scraped. Try a more specific location name.[/red]")
        console.print("[dim]Example: 'India Gate New Delhi' or 'Colosseum Rome'[/dim]")
        return {}

    # Stage 2: Sentiment
    if progress_callback:
        progress_callback(2, "Sentiment Analysis", "Analyzing sentiment and extracting themes with Groq (Llama 8B)...")
    sentiment = analyze_sentiment(reviews)
    time.sleep(1)   # respect Groq rate limits (30 req/min on free tier)

    # Stage 3: Guardrail
    if progress_callback:
        progress_callback(3, "Guardrail Analysis", "Checking review authenticity and trust score with Groq (Llama 70B)...")
    guardrail = guardrail_analysis(reviews, sentiment)
    time.sleep(1)

    # Stage 4: Recommendation
    if progress_callback:
        progress_callback(4, "Generating Recommendation", "Synthesizing final verdict, visit score, and visitor tips...")
    rec = generate_recommendation(location, place_info, reviews, sentiment, guardrail)

    # Display formatted report
    display_report(location, place_info, reviews, sentiment, guardrail, rec)

    elapsed = time.time() - t0
    console.print(f"\n[dim]⏱  Total time: {elapsed:.1f}s[/dim]")

    # Save full JSON
    slug     = "".join(c if c.isalnum() or c in " _-" else "" for c in location)[:30].strip()
    out_file = f"report_{slug.replace(' ', '_')}.json"
    payload  = {
        "location":       location,
        "place_info":     place_info,
        "reviews":        reviews,
        "sentiment":      sentiment,
        "guardrail":      guardrail,
        "recommendation": rec,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    console.print(f"[dim]💾 Full JSON report saved → {out_file}[/dim]")

    return payload


if __name__ == "__main__":
    loc = ""
    n = 30

    if len(sys.argv) > 1:
        if len(sys.argv) > 2 and sys.argv[-1].isdigit():
            loc = " ".join(sys.argv[1:-1])
            n = int(sys.argv[-1])
        else:
            loc = " ".join(sys.argv[1:])

    if not loc:
        try:
            loc = console.input("\n[bold]Enter location to analyze:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            loc = ""
        if not loc:
            console.print("[red]No location entered.[/red]")
            sys.exit(1)

        try:
            n_input = console.input("[bold]Max reviews to scrape (default 30):[/bold] ").strip()
            n = int(n_input) if n_input.isdigit() else 30
        except (EOFError, KeyboardInterrupt):
            n = 30

    analyze(loc, n)