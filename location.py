#!/usr/bin/env python3
"""
Location Review AI Analyzer
Two-Model Architecture:
  Model A — Groq (llama-3.3-70b-versatile) — Primary analyst
  Model B — Groq (llama-3.1-8b-instant)    — Independent verifier (different model, same provider)
Final recommendation: deterministic Python scoring only.
"""

import os, json, time, sys, re
import urllib.request
import urllib.parse
from collections import Counter
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

# ── Configuration ─────────────────────────────────────────────────────────────

APIFY_TOKEN  = os.getenv("APIFY_API_TOKEN") or os.getenv("APify_API_TOKEN", "")
GROQ_KEY     = os.getenv("GROQ_API_KEY", "")

# Model B — Groq llama-3.1-8b-instant (independent verification, proven reliable)
# Using llama-3.1-8b-instant for Model B (most reliable JSON on Groq)
# While same model as guardrail, independence comes from: different stage, different context, cross-validation role
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "llama-3.1-8b-instant")

# Apify actor
ACTOR_ID     = "compass/crawler-google-places"

# ── Groq model assignments ────────────────────────────────────────────────────
# Spread across two Groq quota buckets to avoid hitting the 70B daily TPD cap.
#
#  llama-3.3-70b-versatile : 100K TPD, 14,400 TPM  → Model A sentiment (70B, deep analysis)
#  llama-3.1-8b-instant    : 500K TPD,  6,000 TPM  → Model A guardrail + Model B verifier (8B, reliable JSON)
# 
# Note: Using same model (llama-3.1-8b-instant) for guardrail + verification provides:
#   • Proven JSON reliability (most stable on Groq)
#   • Huge quota (500K TPD — plenty for both tasks)
#   • Independence through: different stage, different context, cross-validation role

STRONG_MODEL = os.getenv("STRONG_MODEL", "llama-3.3-70b-versatile")  # Model A sentiment
FAST_MODEL   = os.getenv("FAST_MODEL",   "llama-3.1-8b-instant")     # guardrail + explanation

REQUIRE_VERIFICATION = os.getenv("REQUIRE_VERIFICATION", "false").lower() == "true"

# Batch size when reviews exceed this count — avoids token overflows
SENTIMENT_BATCH_SIZE = 15

console = Console()

if not APIFY_TOKEN or not GROQ_KEY:
    console.print("[red bold]ERROR:[/red bold] Missing API keys.")
    console.print("Create a .env file with APIFY_API_TOKEN and GROQ_API_KEY")
    sys.exit(1)

apify = ApifyClient(APIFY_TOKEN)
groq  = Groq(api_key=GROQ_KEY)


# ── Shared Groq helper ────────────────────────────────────────────────────────

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

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

            start = raw.find("{")
            end   = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1])
                except json.JSONDecodeError:
                    pass

            if start != -1:
                fragment = raw[start:]
                opens     = fragment.count("{") - fragment.count("}")
                arr_opens = fragment.count("[") - fragment.count("]")
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
            if attempt == 0:
                time.sleep(2)
                continue
            return {}
    return {}


# ── Model B verifier call — Groq with JSON mode ──────────────────────────────

def call_verifier(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> dict:
    """
    Call Model B verifier (Groq with llama-3.1-8b-instant by default — most reliable JSON).
    Uses Groq's native JSON mode for reliable structured output.
    Returns parsed JSON dict, or raises RuntimeError on failure.
    """
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment.")

    client = Groq(api_key=GROQ_KEY)
    
    try:
        response = client.chat.completions.create(
            model=VERIFIER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},  # Native JSON mode
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Groq verifier call failed: {e}")

    if not raw:
        raise RuntimeError("Verifier returned empty content.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting first JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end+1])
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Verifier response is not valid JSON. Raw (first 400): {raw[:400]}")


# ── Geocoder helpers (Photon / komoot) ───────────────────────────────────────

def _photon_request(params: dict, timeout: int = 6) -> dict:
    url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def geocode_city(city: str) -> tuple:
    if not city or not city.strip():
        return None, None
    try:
        data = _photon_request({"q": city.strip(), "limit": 1, "lang": "en"})
        feats = data.get("features", [])
        if feats:
            coords = feats[0].get("geometry", {}).get("coordinates", [])
            if len(coords) == 2:
                return coords[1], coords[0]
    except Exception as e:
        console.print(f"[yellow]City geocode error: {e}[/yellow]")
    return None, None


def get_place_suggestions(query: str, city: str = "", limit: int = 7) -> list:
    if not query or len(query.strip()) < 2:
        return []

    q = f"{query.strip()} {city.strip()}" if city.strip() else query.strip()
    params: dict = {"q": q, "limit": limit * 2, "lang": "en"}

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
    seen: set = set()

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

        label_parts  = [p for p in [name, p_city, state, country] if p]
        display      = ", ".join(label_parts)
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

        location_text = f"{name} {p_city} {state} {country}".lower()
        if city_lower and city_lower in location_text:
            preferred.append(entry)
        else:
            fallback.append(entry)

    combined = preferred + fallback
    return combined[:limit]


def get_city_suggestions(query: str, limit: int = 6) -> list:
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


# ── URL expander ──────────────────────────────────────────────────────────────

def expand_maps_url(url: str) -> str:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LocationAnalyzer/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            final_url = resp.url
        console.print(f"[dim]🔗 Expanded URL: {final_url[:80]}…[/dim]")
        return final_url
    except Exception as e:
        console.print(f"[yellow]URL expand warning: {e} — using original URL[/yellow]")
        return url


# ── MODULE 1: Apify Scraper ───────────────────────────────────────────────────

def scrape_reviews(location: str, max_reviews: int = 40) -> tuple:
    """
    Scrape Google Maps reviews for `location` using Apify.
    Returns (reviews, place_info, review_stats).

    review_stats contains:
        raw_scraped_count       — total items returned by Apify for the best place
        usable_text_review_count — reviews that have non-empty text
        cleaned_review_count    — after dedup / spam filtering
        analyzed_review_count   — set later after cleaning (same as cleaned here)
    """
    console.print(f"\n[bold cyan]📡  Step 1 — Scraping:[/bold cyan] {location}")

    is_url      = location.strip().startswith("http")
    is_place_id = location.strip().lower().startswith("place_id:")

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
    elif is_place_id:
        run_input = {
            "searchStringsArray":  [location.strip()],
            "maxCrawledPlaces":    1,
            "maxReviews":          max_reviews,
            "reviewsSort":         "newest",
            "language":            "en",
            "includeOpeningHours": True,
        }
    else:
        parts       = location.strip().split(",", 1)
        place_query = location.strip()
        sugg        = get_place_suggestions(place_query, city="", limit=1)

        if sugg and sugg[0].get("lat") and sugg[0].get("lon"):
            place_lat  = sugg[0]["lat"]
            place_lon  = sugg[0]["lon"]
            place_name = sugg[0]["search_name"]
            console.print(f"[dim]🌐 Place geocoded: {place_name} -> lat={place_lat:.5f}, lon={place_lon:.5f}[/dim]")
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
            return [], {}, _empty_review_stats()

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

        # ── Build raw list ──
        raw_reviews = best.get("reviews") or []
        raw_scraped_count = len(raw_reviews)

        # ── Usable = has non-empty text ──
        usable = []
        for r in raw_reviews:
            text = (r.get("text") or "").strip()
            if text:
                usable.append({
                    "author": r.get("name") or "Anonymous",
                    "rating": r.get("stars") or 3,
                    "text":   text[:800],
                    "date":   r.get("publishedAtDate") or None,
                    "likes":  r.get("likesCount") or 0,
                })
        usable_text_review_count = len(usable)

        # ── Clean: remove near-duplicates (>70% Jaccard) and obvious spam ──
        cleaned = _clean_reviews(usable)
        cleaned_review_count = len(cleaned)

        review_stats = {
            "raw_scraped_count":        raw_scraped_count,
            "usable_text_review_count": usable_text_review_count,
            "cleaned_review_count":     cleaned_review_count,
            "analyzed_review_count":    cleaned_review_count,  # updated in analyze()
        }

        name = place_info.get("name", location)
        console.print(
            f"[green]✓  Scraped {raw_scraped_count} raw  |  "
            f"{usable_text_review_count} usable  |  "
            f"{cleaned_review_count} after cleaning  for \"{name}\"[/green]"
        )
        return cleaned, place_info, review_stats

    except Exception as e:
        console.print(f"[red]✗  Apify scraping failed: {e}[/red]")
        raise RuntimeError(f"Unable to collect reviews right now. Apify error: {e}")


def _empty_review_stats() -> dict:
    return {
        "raw_scraped_count":        0,
        "usable_text_review_count": 0,
        "cleaned_review_count":     0,
        "analyzed_review_count":    0,
    }


def _clean_reviews(reviews: list) -> list:
    """Remove near-duplicate reviews (>70% Jaccard word overlap) and keep unique ones."""
    if not reviews:
        return []
    kept   = []
    seen_texts = []
    for r in reviews:
        words_r = set(r["text"].lower().split())
        is_dup  = False
        for prev in seen_texts:
            union = words_r | prev
            if union and len(words_r & prev) / len(union) > 0.70:
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
            seen_texts.append(words_r)
    return kept


# ── MODULE 2: Model A — Sentiment Analysis (Groq, batched) ───────────────────

def _sentiment_prompt_for_batch(batch: list, batch_offset: int) -> str:
    """Build the sentiment analysis prompt for one batch of reviews."""
    reviews_block = "\n---\n".join(
        f"[r{batch_offset + i + 1}] ⭐{r['rating']}/5  date:{str(r.get('date',''))[:10]}\n{r['text'][:350]}"
        for i, r in enumerate(batch)
    )
    return f"""You are an expert review analyst. Perform DEEP multi-dimensional sentiment analysis
on the following location reviews. Review IDs are shown as [rN] — use them in evidence fields.
Return ONLY valid JSON matching this EXACT structure (no markdown, no explanation):

{{
  "per_review": [
    {{
      "id": "r1",
      "sentiment": "Positive|Negative|Neutral|Mixed",
      "score": 0.75,
      "emotion": "Excited|Happy|Satisfied|Neutral|Disappointed|Frustrated|Angry",
      "intensity": "Low|Medium|High",
      "key_phrase": "one crisp sentence capturing the review"
    }}
  ],
  "aspect_scores": {{
    "food_quality":    {{"score": 7.5, "reviews_mentioning": 5, "summary": "brief note or null", "evidence_review_ids": ["r1"]}},
    "service":         {{"score": 8.0, "reviews_mentioning": 3, "summary": "brief note or null", "evidence_review_ids": []}},
    "ambience":        {{"score": 6.5, "reviews_mentioning": 2, "summary": "brief note or null", "evidence_review_ids": []}},
    "value_for_money": {{"score": 7.0, "reviews_mentioning": 2, "summary": "brief note or null", "evidence_review_ids": []}},
    "cleanliness":     {{"score": 8.5, "reviews_mentioning": 1, "summary": "brief note or null", "evidence_review_ids": []}},
    "accessibility":   {{"score": 7.0, "reviews_mentioning": 1, "summary": "brief note or null", "evidence_review_ids": []}},
    "crowd_wait_time": {{"score": 5.5, "reviews_mentioning": 3, "summary": "brief note or null", "evidence_review_ids": []}}
  }},
  "themes": [
    {{
      "name": "theme name",
      "sentiment": "Positive|Negative|Mixed",
      "frequency": 3,
      "representative_quote": "exact short quote",
      "evidence": "1-sentence synthesis",
      "evidence_review_ids": ["r1", "r3"]
    }}
  ],
  "positive_points": [
    {{"claim": "Food is consistently praised", "evidence_review_ids": ["r1", "r4"]}}
  ],
  "negative_points": [
    {{"claim": "Parking is difficult", "evidence_review_ids": ["r2", "r7"]}}
  ],
  "positive_keywords": ["keyword1", "keyword2"],
  "negative_keywords": ["keyword1", "keyword2"],
  "standout_positive_quote": "most positive sentence verbatim",
  "standout_negative_quote": "most critical sentence verbatim",
  "overall_sentiment": "Positive|Negative|Neutral|Mixed",
  "sentiment_score": 0.65,
  "emotional_tone": "Excited|Happy|Satisfied|Disappointed|Angry|Neutral",
  "emotion_distribution": {{
    "Excited": 0, "Happy": 0, "Satisfied": 0,
    "Neutral": 0, "Disappointed": 0, "Frustrated": 0, "Angry": 0
  }},
  "crowd_profile": {{
    "dominant_visitor_type": "Families|Couples|Solo travellers|Business visitors|Tourists|Locals|Mixed",
    "mention_evidence": "brief evidence",
    "accessibility_notes": "any accessibility mentions"
  }},
  "temporal_trend": {{
    "recent_sentiment": "Positive|Negative|Neutral|Mixed",
    "recent_score": 0.7,
    "older_sentiment": "Positive|Negative|Neutral|Mixed",
    "older_score": 0.6,
    "trend": "Improving|Declining|Stable",
    "trend_explanation": "1-sentence reason"
  }},
  "review_diversity": "High|Medium|Low"
}}

RULES:
- Use null score and 0 reviews_mentioning if an aspect is never mentioned
- sentiment_score: -1.0 (very negative) to +1.0 (very positive)
- aspect scores: 0-10
- Include ALL evidence_review_ids that support each claim
- At least 3 themes if review count allows

REVIEWS:
{reviews_block}"""


def _merge_sentiment_batches(batch_results: list) -> dict:
    """Merge multiple batch sentiment results into one combined result."""
    if not batch_results:
        return {}
    if len(batch_results) == 1:
        return batch_results[0]

    merged = {
        "per_review":        [],
        "positive_keywords": [],
        "negative_keywords": [],
        "themes":            [],
        "positive_points":   [],
        "negative_points":   [],
        "emotion_distribution": {
            "Excited": 0, "Happy": 0, "Satisfied": 0,
            "Neutral": 0, "Disappointed": 0, "Frustrated": 0, "Angry": 0
        },
    }

    aspect_keys = ["food_quality","service","ambience","value_for_money",
                   "cleanliness","accessibility","crowd_wait_time"]
    aspect_accum = {k: {"scores": [], "mentions": 0, "evidence_ids": [], "summaries": []}
                    for k in aspect_keys}

    all_scores   = []
    trend_latest = None

    for br in batch_results:
        merged["per_review"].extend(br.get("per_review", []))

        # keywords — merge unique
        for kw in br.get("positive_keywords", []):
            if kw not in merged["positive_keywords"]:
                merged["positive_keywords"].append(kw)
        for kw in br.get("negative_keywords", []):
            if kw not in merged["negative_keywords"]:
                merged["negative_keywords"].append(kw)

        # themes — merge by name
        existing_theme_names = {t["name"] for t in merged["themes"] if isinstance(t, dict)}
        for t in br.get("themes", []):
            if isinstance(t, dict):
                if t.get("name") not in existing_theme_names:
                    merged["themes"].append(t)
                    existing_theme_names.add(t.get("name"))
                else:
                    # accumulate frequency
                    for mt in merged["themes"]:
                        if isinstance(mt, dict) and mt.get("name") == t.get("name"):
                            mt["frequency"] = mt.get("frequency", 0) + t.get("frequency", 0)
                            for eid in t.get("evidence_review_ids", []):
                                if eid not in mt.get("evidence_review_ids", []):
                                    mt.setdefault("evidence_review_ids", []).append(eid)
                            break

        # positive / negative points
        for pp in br.get("positive_points", []):
            merged["positive_points"].append(pp)
        for np_ in br.get("negative_points", []):
            merged["negative_points"].append(np_)

        # emotion distribution
        for emo, cnt in br.get("emotion_distribution", {}).items():
            if emo in merged["emotion_distribution"]:
                merged["emotion_distribution"][emo] += (cnt or 0)

        # aspect scores — weighted average
        for k in aspect_keys:
            asp = br.get("aspect_scores", {}).get(k, {})
            if isinstance(asp, dict) and asp.get("score") is not None:
                aspect_accum[k]["scores"].append(asp["score"])
                aspect_accum[k]["mentions"] += asp.get("reviews_mentioning", 0)
                aspect_accum[k]["evidence_ids"].extend(asp.get("evidence_review_ids", []))
                if asp.get("summary"):
                    aspect_accum[k]["summaries"].append(asp["summary"])

        # overall score
        s = br.get("sentiment_score")
        if s is not None:
            all_scores.append(s)

        # take temporal trend from first batch (most recent reviews)
        if trend_latest is None:
            trend_latest = br.get("temporal_trend")

        # standout quotes — take best from first non-empty
        if not merged.get("standout_positive_quote") and br.get("standout_positive_quote"):
            merged["standout_positive_quote"] = br["standout_positive_quote"]
        if not merged.get("standout_negative_quote") and br.get("standout_negative_quote"):
            merged["standout_negative_quote"] = br["standout_negative_quote"]

        # crowd profile — take first
        if not merged.get("crowd_profile") and br.get("crowd_profile"):
            merged["crowd_profile"] = br["crowd_profile"]

    # Finalise aspect scores
    merged_aspects = {}
    for k in aspect_keys:
        acc = aspect_accum[k]
        if acc["scores"]:
            avg_score = round(sum(acc["scores"]) / len(acc["scores"]), 2)
            summary   = acc["summaries"][0] if acc["summaries"] else None
            merged_aspects[k] = {
                "score":              avg_score,
                "reviews_mentioning": acc["mentions"],
                "summary":            summary,
                "evidence_review_ids": list(dict.fromkeys(acc["evidence_ids"])),
            }
        else:
            merged_aspects[k] = {"score": None, "reviews_mentioning": 0,
                                  "summary": None, "evidence_review_ids": []}
    merged["aspect_scores"] = merged_aspects

    # Overall sentiment score
    merged["sentiment_score"] = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

    # Overall sentiment label from score
    s = merged["sentiment_score"]
    if s >= 0.4:
        merged["overall_sentiment"] = "Positive"
    elif s <= -0.3:
        merged["overall_sentiment"] = "Negative"
    elif -0.15 <= s <= 0.15:
        merged["overall_sentiment"] = "Neutral"
    else:
        merged["overall_sentiment"] = "Mixed"

    if trend_latest:
        merged["temporal_trend"] = trend_latest

    return merged


def analyze_sentiment(reviews: list) -> dict:
    """
    MODEL A — Deep sentiment analysis using Groq (STRONG_MODEL).
    Processes ALL reviews via batching — no arbitrary [:20] truncation.
    Batch size is SENTIMENT_BATCH_SIZE (default 15) to stay within token limits.
    """
    console.print(
        f"\n[bold cyan]🧠  Model A — Sentiment analysis (Groq / {STRONG_MODEL})[/bold cyan]  "
        f"[dim]{len(reviews)} reviews[/dim]"
    )

    if not reviews:
        return {}

    # ── Split into batches ──
    batches      = [reviews[i:i+SENTIMENT_BATCH_SIZE]
                    for i in range(0, len(reviews), SENTIMENT_BATCH_SIZE)]
    batch_results = []

    for b_idx, batch in enumerate(batches):
        offset = b_idx * SENTIMENT_BATCH_SIZE
        console.print(
            f"[dim]  Batch {b_idx+1}/{len(batches)} — reviews "
            f"r{offset+1}–r{offset+len(batch)}[/dim]"
        )
        prompt = _sentiment_prompt_for_batch(batch, offset)
        result = call_groq(prompt, model=STRONG_MODEL, max_tokens=4096)
        batch_results.append(result)
        if b_idx < len(batches) - 1:
            time.sleep(1.2)   # respect Groq rate limits

    merged = _merge_sentiment_batches(batch_results)

    # ── Always compute rating distribution from raw data ──
    counts = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}
    for r in reviews:
        k = str(max(1, min(5, int(r.get("rating", 3)))))
        counts[k] = counts.get(k, 0) + 1
    merged["rating_counts"] = counts

    # ── Log summary ──
    s  = merged.get("sentiment_score", 0)
    o  = merged.get("overall_sentiment", "Unknown")
    tr = merged.get("temporal_trend", {}).get("trend", "?")
    kw = ", ".join(merged.get("positive_keywords", [])[:5]) or "—"
    console.print(
        f"[green]✓  Model A sentiment: {o}  |  Score: {s:.2f}  |  "
        f"Trend: {tr}  |  Top keywords: {kw}[/green]"
    )
    return merged


# ── MODULE 3: Guardrail Analysis (Model A / Groq) ────────────────────────────

def _heuristic_checks(reviews: list) -> dict:
    """Pure-Python pre-pass. Returns flags and stats fed into the guardrail prompt."""
    n = len(reviews)
    flags = []

    if n == 0:
        return {"flags": ["No reviews to analyze"], "stats": {}}

    ratings = [r["rating"] for r in reviews]
    texts   = [r["text"].lower() for r in reviews]
    authors = [r.get("author", "") for r in reviews]

    five_star_pct = ratings.count(5) / n
    one_star_pct  = ratings.count(1) / n
    if five_star_pct > 0.75:
        flags.append(f"Unusually high 5-star ratio ({five_star_pct:.0%}) — potential rating manipulation")
    if one_star_pct > 0.40:
        flags.append(f"High 1-star ratio ({one_star_pct:.0%}) — possible targeted negative campaign")
    if len(set(ratings)) == 1:
        flags.append("All reviews share the exact same star rating — highly suspicious uniformity")

    short_5star = [r for r in reviews if r["rating"] == 5 and len(r["text"]) < 12]
    if n > 0 and len(short_5star) / n > 0.30:
        flags.append(f"{len(short_5star)} very short 5-star reviews (< 12 chars) — likely filler boosts")

    duplicates = 0
    check_n    = min(n, 30)
    for i in range(check_n):
        for j in range(i + 1, check_n):
            w1 = set(texts[i].split())
            w2 = set(texts[j].split())
            union = w1 | w2
            if union and len(w1 & w2) / len(union) > 0.55:
                duplicates += 1
    if duplicates >= 3:
        flags.append(f"{duplicates} near-duplicate review pairs (>55% word overlap)")

    ngram_counter: Counter = Counter()
    for txt in texts[:30]:
        words = txt.split()
        for k in range(len(words) - 2):
            ng = " ".join(words[k:k+3])
            if len(ng) > 8:
                ngram_counter[ng] += 1
    repeated_phrases = [ng for ng, cnt in ngram_counter.most_common(5) if cnt >= 4]
    if repeated_phrases:
        flags.append(f"Repeated phrases across reviews: {repeated_phrases[:3]} — coordinated language")

    unique_authors   = len(set(authors))
    single_rev_ratio = unique_authors / n if n > 0 else 1
    if single_rev_ratio < 0.7 and n > 8:
        flags.append(f"Low author diversity ({unique_authors} unique / {n} reviews) — possible sock-puppets")

    if n < 6:
        flags.append("Too few reviews for high-confidence analysis — treat results with caution")

    dates = [r.get("date", "")[:10] for r in reviews if r.get("date")]
    if len(dates) >= 5:
        unique_dates = len(set(dates))
        if unique_dates / len(dates) < 0.25:
            flags.append("Many reviews share the same date — possible coordinated review-bombing")

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
    MODEL A — Deep guardrail / authenticity analysis using Groq STRONG_MODEL.
    Uses up to 30 reviews for the LLM pass; heuristics run on all reviews.
    """
    console.print(
        f"\n[bold cyan]🛡  Model A — Guardrail analysis "
        f"(Groq / {FAST_MODEL})[/bold cyan]"
    )

    heuristics      = _heuristic_checks(reviews)
    heuristic_flags = heuristics["flags"]
    heuristic_stats = heuristics["stats"]

    # Send up to 18 representative reviews to the LLM (reduced to fit 6K TPM reliably)
    # Cap each review text at 120 chars to stay comfortably under 6K TPM limit
    review_sample = [
        {
            "id":     f"r{i+1}",
            "rating": r["rating"],
            "text":   r["text"][:120],  # Reduced to 120 chars for reliable TPM fit
        }
        for i, r in enumerate(reviews[:18])  # Cap at 18 reviews to fit budget
    ]

    sentiment_context = {
        "overall":           sentiment.get("overall_sentiment"),
        "score":             sentiment.get("sentiment_score"),
        "positive_keywords": sentiment.get("positive_keywords", [])[:10],
        "negative_keywords": sentiment.get("negative_keywords", [])[:8],
        "themes":            sentiment.get("themes", [])[:5],
        "temporal_trend":    sentiment.get("temporal_trend", {}),
    }

    prompt = f"""Review integrity analysis. Sentiment context: {json.dumps(sentiment_context)}
Heuristic flags: {json.dumps(heuristic_flags)}
Stats: {json.dumps(heuristic_stats)}
Reviews (up to 20): {json.dumps(review_sample)}

Return ONLY this JSON:
{{
  "trust_score": 0.0,
  "credibility_score": 0.0,
  "fake_review_probability": 0.0,
  "review_quality": "High|Medium|Low",
  "bias_level": "Low|Medium|High",
  "bias_direction": "Positive|Negative|None",
  "linguistic_analysis": {{
    "templated_language_detected": false,
    "copy_paste_evidence": "description or null",
    "vocabulary_diversity": "High|Medium|Low",
    "writing_style_consistency": "Consistent (suspicious)|Varied (natural)|Mixed",
    "language_notes": "1-sentence observation"
  }},
  "rating_integrity": {{
    "distribution_natural": true,
    "anomalies": [],
    "inflated_stars_estimate": 0,
    "suppressed_stars_estimate": 0,
    "adjusted_true_rating": 0.0,
    "rating_notes": "1-sentence assessment"
  }},
  "reviewer_behavior": {{
    "sock_puppet_risk": "Low|Medium|High",
    "coordinated_posting_risk": "Low|Medium|High",
    "evidence": "description or None detected"
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
    "crowd_wait_time": {{"credible": true,  "confidence": 0.6, "note": "brief note"}}
  }},
  "genuine_positives": [
    {{
      "aspect": "aspect name",
      "evidence": "what reviewers say",
      "confidence": 0.85,
      "supporting_review_ids": ["r1", "r3"]
    }}
  ],
  "genuine_concerns": [
    {{
      "aspect": "concern name",
      "evidence": "what reviewers say",
      "severity": "Minor|Moderate|Major",
      "supporting_review_ids": ["r2"]
    }}
  ],
  "suspicious_patterns": [],
  "verified_facts": [],
  "guardrail_summary": "2-sentence honest assessment",
  "analyst_recommendation": "1 sentence on trust level"
}}"""

    # Guardrail is structured JSON — 8B model is sufficient and saves 70B daily quota
    result = call_groq(prompt, model=FAST_MODEL, max_tokens=4096)

    ai_flags  = result.get("suspicious_patterns", [])
    all_flags = list(dict.fromkeys(heuristic_flags + ai_flags))
    result["suspicious_patterns"] = all_flags
    result["heuristic_stats"]     = heuristic_stats

    ts  = result.get("trust_score", 0)
    rq  = result.get("review_quality", "?")
    fp  = result.get("fake_review_probability", 0)
    adj = result.get("rating_integrity", {}).get("adjusted_true_rating", "?")
    console.print(
        f"[green]✓  Trust: {ts:.0%}  |  Quality: {rq}  |  "
        f"Fake prob: {fp:.0%}  |  Adj. rating: {adj}/5[/green]"
    )
    return result


# ── MODULE 4: Model B — Groq Independent Verifier ────────────────────────────

VERIFIER_SYSTEM_PROMPT = """You are Model B, an independent verification model reviewing Model A's output.
Do NOT assume Model A is correct. Use the original reviews to verify every claim.
Check: sentiment accuracy, aspect scores, keywords, positive/negative claims, evidence IDs, hallucinations, exaggerations, missing signals.
For every error: explain why and provide corrected interpretation with review IDs.
Do NOT invent evidence. Every correction must reference real review IDs.
Return ONLY valid JSON. Do NOT include explanations, reasoning, or markdown. Output must start with { and end with }."""


def verify_analysis(reviews: list, model_a_sentiment: dict, model_a_guardrail: dict) -> dict:
    """
    MODEL B — Independent verification via Groq (VERIFIER_MODEL).
    Uses a different Groq model from Model A for independent verification.
    Receives ORIGINAL reviews + trimmed Model A analysis.
    Returns verification JSON or an UNAVAILABLE status dict on failure.
    Does NOT generate a recommendation or score.
    """
    console.print(
        f"\n[bold magenta]🔍  Model B — Independent verification "
        f"(Groq / {VERIFIER_MODEL})[/bold magenta]"
    )

    if not GROQ_KEY:
        console.print("[yellow]⚠  GROQ_API_KEY not set — Model B verification unavailable.[/yellow]")
        return _unavailable_verification("GROQ_API_KEY is not configured.")

    # ── Trim reviews: cap at 20, text at 150 chars ────────────────────────────
    review_sample = reviews[:20]
    review_list = [
        {
            "id":     f"r{i+1}",
            "rating": r["rating"],
            "text":   r["text"][:150],
        }
        for i, r in enumerate(review_sample)
    ]

    # ── Trim Model A summary to essentials only ───────────────────────────────
    # Aspects: keep score + mentions + first evidence ID only
    asp_trimmed = {}
    for k, v in model_a_sentiment.get("aspect_scores", {}).items():
        if isinstance(v, dict) and v.get("score") is not None:
            asp_trimmed[k] = {
                "score":    v.get("score"),
                "mentions": v.get("reviews_mentioning", 0),
                "ids":      (v.get("evidence_review_ids") or [])[:2],
            }

    # Positive / negative points: claim + first 2 IDs only
    def _trim_points(pts):
        out = []
        for p in (pts or [])[:4]:
            if isinstance(p, dict):
                out.append({"claim": (p.get("claim") or "")[:80],
                            "ids": (p.get("evidence_review_ids") or [])[:2]})
        return out

    # Concerns: aspect + severity + first 2 IDs
    concerns_trimmed = [
        {"aspect": c.get("aspect",""), "severity": c.get("severity",""),
         "ids": (c.get("supporting_review_ids") or [])[:2]}
        for c in model_a_guardrail.get("genuine_concerns", [])[:4]
        if isinstance(c, dict)
    ]

    model_a_summary = {
        "sentiment": {
            "overall": model_a_sentiment.get("overall_sentiment"),
            "score":   model_a_sentiment.get("sentiment_score"),
        },
        "aspects":        asp_trimmed,
        "pos_keywords":   model_a_sentiment.get("positive_keywords", [])[:6],
        "neg_keywords":   model_a_sentiment.get("negative_keywords", [])[:6],
        "positive_points": _trim_points(model_a_sentiment.get("positive_points", [])),
        "negative_points": _trim_points(model_a_sentiment.get("negative_points", [])),
        "trust_score":     model_a_guardrail.get("trust_score"),
        "concerns":        concerns_trimmed,
    }

    user_prompt = f"""=== REVIEWS (up to 20, verify Model A against these) ===
{json.dumps(review_list, indent=2)}

=== MODEL A OUTPUT (verify this) ===
{json.dumps(model_a_summary, indent=2)}

OUTPUT ONLY THE JSON BELOW. DO NOT include any explanation, reasoning, or markdown. Start with {{ and end with }}.

{{
  "verification_status": "PASS|CORRECTED|FAIL",
  "accuracy": 0.91,
  "sentiment":  {{"status": "PASS|FAIL",           "issues": []}},
  "aspects":    {{"status": "PASS|CORRECTED|FAIL",  "issues": []}},
  "keywords":   {{"status": "PASS|CORRECTED|FAIL",  "issues": []}},
  "guardrail":  {{"status": "PASS|CORRECTED|FAIL",  "issues": []}},
  "evidence":   {{"status": "PASS|FAIL", "unsupported_claims": []}},
  "hallucination_detected": false,
  "corrections": [
    {{
      "field": "e.g. aspects.food_quality",
      "original_claim": "Model A claim",
      "corrected_claim": "what reviews actually support",
      "reason": "brief reason",
      "evidence_review_ids": ["r1"]
    }}
  ],
  "missing_positive_evidence": [],
  "missing_negative_evidence": [],
  "verification_notes": "1 sentence summary"
}}

RULES: Only correct claims unsupported by review text. Every correction needs evidence_review_ids. No recommendation."""

    try:
        result = call_verifier(VERIFIER_SYSTEM_PROMPT, user_prompt, max_tokens=1500)
        status   = result.get("verification_status", "UNKNOWN")
        accuracy = result.get("accuracy", 0)
        hall     = result.get("hallucination_detected", False)
        n_corr   = len(result.get("corrections", []))
        console.print(
            f"[green]✓  Model B (Groq): {status}  |  Accuracy: {accuracy:.0%}  |  "
            f"Hallucination: {hall}  |  Corrections: {n_corr}[/green]"
        )
        return result
    except RuntimeError as e:
        err_msg = str(e)
        # If it's a null content error, it might be a transient issue - try once more with a simpler payload
        if "null content" in err_msg.lower() and len(review_list) > 10:
            console.print("[yellow]⚠  Model B returned null, retrying with fewer reviews...[/yellow]")
            # Retry with only first 10 reviews
            retry_list = review_list[:10]
            retry_prompt = user_prompt.replace(json.dumps(review_list, indent=2), json.dumps(retry_list, indent=2))
            try:
                result = call_openrouter_verifier(VERIFIER_SYSTEM_PROMPT, retry_prompt, max_tokens=1500)
                status   = result.get("verification_status", "UNKNOWN")
                accuracy = result.get("accuracy", 0)
                hall     = result.get("hallucination_detected", False)
                n_corr   = len(result.get("corrections", []))
                console.print(
                    f"[green]✓  Model B (OpenRouter, 10 reviews): {status}  |  Accuracy: {accuracy:.0%}  |  "
                    f"Hallucination: {hall}  |  Corrections: {n_corr}[/green]"
                )
                return result
            except RuntimeError as e2:
                console.print(f"[yellow]⚠  Model B unavailable: {e2}[/yellow]")
                return _unavailable_verification(str(e2))
        
        console.print(f"[yellow]⚠  Model B unavailable: {e}[/yellow]")
        return _unavailable_verification(str(e))


def _unavailable_verification(reason: str) -> dict:
    return {
        "verification_status": "UNAVAILABLE",
        "accuracy":            None,
        "reason":              reason,
        "sentiment":           {"status": "UNAVAILABLE", "issues": []},
        "aspects":             {"status": "UNAVAILABLE", "issues": []},
        "keywords":            {"status": "UNAVAILABLE", "issues": []},
        "guardrail":           {"status": "UNAVAILABLE", "issues": []},
        "evidence":            {"status": "UNAVAILABLE", "unsupported_claims": []},
        "hallucination_detected": None,
        "corrections":         [],
        "verification_notes":  "Independent verification is currently unavailable.",
    }


# ── MODULE 5: Apply Corrections ───────────────────────────────────────────────

def apply_corrections(sentiment: dict, guardrail: dict, verification: dict) -> tuple:
    """
    Apply evidence-supported corrections from Model B to Model A's output.
    Returns (verified_sentiment, verified_guardrail).

    Rules:
    - Only apply a correction if evidence_review_ids is non-empty.
    - Never apply a correction without review evidence.
    - Does not change the final score (that is Python's job).
    """
    if not verification or verification.get("verification_status") in ("PASS", "UNAVAILABLE"):
        # No corrections needed or verifier unavailable — use Model A as-is
        return sentiment.copy(), guardrail.copy()

    corrections = verification.get("corrections", [])
    if not corrections:
        return sentiment.copy(), guardrail.copy()

    v_sentiment = sentiment.copy()
    v_guardrail = guardrail.copy()

    for corr in corrections:
        # Only apply if there is actual review evidence
        evidence_ids = corr.get("evidence_review_ids", [])
        if not evidence_ids:
            console.print(
                f"[dim]  Skipping correction for '{corr.get('field')}' "
                f"— no evidence_review_ids provided[/dim]"
            )
            continue

        field   = corr.get("field", "")
        new_val = corr.get("corrected_claim", "")
        reason  = corr.get("reason", "")

        console.print(
            f"[cyan]  Applying correction: {field} — {reason[:80]}[/cyan]"
        )

        # ── Aspect sentiment corrections ──
        if field.startswith("aspects."):
            aspect_key = field.split(".", 1)[1]
            asp = v_sentiment.get("aspect_scores", {}).get(aspect_key)
            if isinstance(asp, dict) and new_val:
                # Only update if the corrected claim contains an explicit NEW score
                # that is meaningfully different from the existing one
                score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', new_val)
                if score_match:
                    candidate = float(score_match.group(1))
                    if 0.0 <= candidate <= 10.0:
                        existing_score = asp.get("score")
                        # Only apply if the correction actually changes the score by > 0.5
                        if existing_score is None or abs(candidate - existing_score) > 0.5:
                            asp["score"]   = candidate
                            asp["summary"] = new_val[:120]
                            v_sentiment["aspect_scores"][aspect_key] = asp

        # ── Overall sentiment correction ──
        elif field == "sentiment" or field == "sentiment.overall":
            if "negative" in new_val.lower():
                v_sentiment["overall_sentiment"] = "Negative"
                if v_sentiment.get("sentiment_score", 0) > 0:
                    v_sentiment["sentiment_score"] = -abs(v_sentiment["sentiment_score"])
            elif "positive" in new_val.lower():
                v_sentiment["overall_sentiment"] = "Positive"
                if v_sentiment.get("sentiment_score", 0) < 0:
                    v_sentiment["sentiment_score"] = abs(v_sentiment["sentiment_score"])
            elif "mixed" in new_val.lower():
                v_sentiment["overall_sentiment"] = "Mixed"

        # ── Guardrail / trust corrections ──
        elif field.startswith("guardrail"):
            # Add the correction as a new genuine concern with evidence
            new_concern = {
                "aspect":              corr.get("field", "verified concern"),
                "evidence":            new_val[:200],
                "severity":            "Moderate",
                "supporting_review_ids": evidence_ids,
                "source":              "Model B correction",
            }
            concerns = v_guardrail.get("genuine_concerns", [])
            concerns.append(new_concern)
            v_guardrail["genuine_concerns"] = concerns

        # ── Keyword corrections — only remove if corrected_claim says unsupported ──
        elif field == "keywords.positive":
            bad_kw = corr.get("original_claim", "").lower().strip()
            corrected = corr.get("corrected_claim", "").lower()
            if "not" in corrected or "unsupported" in corrected or "absent" in corrected or "incorrect" in corrected:
                v_sentiment["positive_keywords"] = [
                    kw for kw in v_sentiment.get("positive_keywords", [])
                    if kw.lower() != bad_kw
                ]
        elif field == "keywords.negative":
            bad_kw = corr.get("original_claim", "").lower().strip()
            corrected = corr.get("corrected_claim", "").lower()
            if "not" in corrected or "unsupported" in corrected or "absent" in corrected or "incorrect" in corrected:
                v_sentiment["negative_keywords"] = [
                    kw for kw in v_sentiment.get("negative_keywords", [])
                    if kw.lower() != bad_kw
                ]

    return v_sentiment, v_guardrail


# ── MODULE 6: Deterministic Python Scoring & Recommendation ──────────────────

def calculate_final_score(
    reviews:     list,
    place_info:  dict,
    sentiment:   dict,
    guardrail:   dict,
    verification: dict,
) -> dict:
    """
    DETERMINISTIC Python scoring engine.
    Neither Model A nor Model B decides the verdict.
    Weights:
        Sentiment score   30%
        Aspect score      25%
        Google rating     15%
        Trust/guardrail   15%
        Review consistency 10%
        Recency            5%
    """
    n = len(reviews)

    # ── 1. Sentiment component (0-10) ──
    raw_s          = sentiment.get("sentiment_score", 0.0)      # -1 to +1
    sentiment_comp = round((raw_s + 1) / 2 * 10, 2)             # → 0-10

    # ── 2. Aspect component (0-10) — weighted mean of scored aspects ──
    aspect_weights = {
        "food_quality":    1.5,
        "service":         1.5,
        "ambience":        1.2,
        "value_for_money": 1.0,
        "cleanliness":     1.0,
        "crowd_wait_time": 0.8,
        "accessibility":   0.5,
    }
    asp_scores = sentiment.get("aspect_scores", {})
    weighted_sum = 0.0
    weight_total = 0.0
    for k, w in aspect_weights.items():
        v = asp_scores.get(k, {})
        score = v.get("score") if isinstance(v, dict) else None
        if score is not None:
            weighted_sum += score * w
            weight_total += w
    aspect_comp = round(weighted_sum / weight_total, 2) if weight_total > 0 else 5.0

    # ── 3. Google rating component (0-10) ──
    google_score = place_info.get("google_score") or 0
    rating_comp  = round(float(google_score) * 2, 2)   # 5-star → 10-point

    # If no Google score, fall back to avg scraped rating
    if rating_comp == 0 and reviews:
        avg_scraped = sum(r["rating"] for r in reviews) / len(reviews)
        rating_comp = round(avg_scraped * 2, 2)

    # ── 4. Trust/guardrail component (0-10) ──
    trust_score = guardrail.get("trust_score", 0.7)
    fake_prob   = guardrail.get("fake_review_probability", 0.2)
    # Penalise high fake probability
    trust_comp  = round((trust_score - fake_prob * 0.5) * 10, 2)
    trust_comp  = max(0.0, min(10.0, trust_comp))

    # Apply verifier accuracy as a modifier on trust
    if verification.get("verification_status") not in ("UNAVAILABLE", None):
        verif_accuracy = verification.get("accuracy") or 1.0
        trust_comp     = round(trust_comp * (0.5 + verif_accuracy * 0.5), 2)
        trust_comp     = max(0.0, min(10.0, trust_comp))

    # ── 5. Guardrail risk penalties ──
    risk_penalty = 0.0
    concerns     = guardrail.get("genuine_concerns", [])
    for c in concerns:
        if not isinstance(c, dict): continue
        sev = c.get("severity", "Minor")
        if sev == "Major":
            risk_penalty += 1.5
        elif sev == "Moderate":
            risk_penalty += 0.8
        else:
            risk_penalty += 0.2
    # Also penalise from verification corrections
    for corr in verification.get("corrections", []):
        if corr.get("evidence_review_ids"):
            risk_penalty += 0.3
    risk_penalty = min(risk_penalty, 3.0)   # cap penalty

    # ── 6. Review consistency component (0-10) ──
    per_review    = sentiment.get("per_review", [])
    review_scores = [p.get("score", 0) for p in per_review if isinstance(p.get("score"), (int, float))]
    if len(review_scores) >= 3:
        mean_rs = sum(review_scores) / len(review_scores)
        variance = sum((s - mean_rs) ** 2 for s in review_scores) / len(review_scores)
        std_dev  = variance ** 0.5
        consistency_comp = round(max(0, 10 - std_dev * 5), 2)
    else:
        consistency_comp = 5.0

    # ── 7. Recency component (0-10) ──
    # Use temporal trend if available
    trend = sentiment.get("temporal_trend", {})
    trend_label = trend.get("trend", "Stable")
    recent_score_raw = trend.get("recent_score", raw_s)
    recency_comp = round((recent_score_raw + 1) / 2 * 10, 2)
    if trend_label == "Declining":
        recency_comp = max(0.0, recency_comp - 1.5)
    elif trend_label == "Improving":
        recency_comp = min(10.0, recency_comp + 1.0)

    # ── Weighted composite ──
    composite = (
        sentiment_comp  * 0.30 +
        aspect_comp     * 0.25 +
        rating_comp     * 0.15 +
        trust_comp      * 0.15 +
        consistency_comp * 0.10 +
        recency_comp    * 0.05
    )
    final_score = round(max(0.0, min(10.0, composite - risk_penalty)), 2)

    # ── Verdict ──
    if final_score >= 8.0:
        verdict = "HIGHLY RECOMMENDED"
    elif final_score >= 6.5:
        verdict = "RECOMMENDED"
    elif final_score >= 4.5:
        verdict = "VISIT WITH CAUTION"
    else:
        verdict = "NOT RECOMMENDED"

    # ── Confidence ──
    # Depends on: review count, consistency, evidence coverage, verifier accuracy, recency
    base_conf = min(1.0, n / 50)              # more reviews → higher confidence
    if consistency_comp >= 7:
        base_conf = min(1.0, base_conf + 0.15)
    if trust_score >= 0.8:
        base_conf = min(1.0, base_conf + 0.1)
    v_status = verification.get("verification_status")
    if v_status == "PASS":
        base_conf = min(1.0, base_conf + 0.1)
        verif_acc = verification.get("accuracy") or 1.0
        base_conf = min(1.0, base_conf * (0.7 + verif_acc * 0.3))
    elif v_status == "UNAVAILABLE":
        base_conf = max(0.0, base_conf - 0.1)   # slight penalty without verification
    elif v_status == "CORRECTED":
        base_conf = max(0.0, base_conf - 0.05)
    confidence = round(base_conf, 3)

    score_breakdown = {
        "sentiment_comp":    sentiment_comp,
        "aspect_comp":       aspect_comp,
        "rating_comp":       rating_comp,
        "trust_comp":        trust_comp,
        "consistency_comp":  consistency_comp,
        "recency_comp":      recency_comp,
        "risk_penalty":      risk_penalty,
        "composite_raw":     round(composite, 2),
        "final_score":       final_score,
    }

    console.print(
        f"[bold green]✓  Python scoring: {verdict}  |  "
        f"Score: {final_score}/10  |  Confidence: {confidence:.0%}[/bold green]"
    )
    return {
        "verdict":        verdict,
        "score":          final_score,
        "confidence":     confidence,
        "score_breakdown": score_breakdown,
    }


# ── MODULE 7: Recommendation (Groq explanation only) ─────────────────────────

def generate_recommendation(
    location:      str,
    place_info:    dict,
    reviews:       list,
    sentiment:     dict,
    guardrail:     dict,
    final_scoring: dict,
) -> dict:
    """
    Groq generates the EXPLANATION TEXT (pros, cons, tips, verdict text).
    It does NOT decide the score or the verdict label — those come from Python.
    """
    console.print(
        f"\n[bold cyan]💡  Generating explanation (Groq / {STRONG_MODEL})[/bold cyan]"
    )

    verdict    = final_scoring["verdict"]
    score      = final_scoring["score"]
    confidence = final_scoring["confidence"]

    context = {
        "location":          location,
        "place_name":        place_info.get("name", location),
        "category":          place_info.get("category", ""),
        "google_official":   place_info.get("google_score", 0),
        "reviews_analyzed":  len(reviews),
        "verdict":           verdict,           # Python-determined — do not override
        "visit_score":       score,             # Python-determined — do not override
        "confidence":        confidence,
        "overall_sentiment": sentiment.get("overall_sentiment"),
        "sentiment_score":   sentiment.get("sentiment_score"),
        "top_positive_kw":   sentiment.get("positive_keywords", [])[:6],
        "top_negative_kw":   sentiment.get("negative_keywords", [])[:4],
        "positive_points":   sentiment.get("positive_points", [])[:5],
        "negative_points":   sentiment.get("negative_points", [])[:5],
        "themes":            [
            {"name": t.get("name"), "sentiment": t.get("sentiment"),
             "evidence": (t.get("evidence",""))[:80]}
            for t in sentiment.get("themes", [])[:4]
            if isinstance(t, dict)
        ],
        "trust_score":       guardrail.get("trust_score"),
        "genuine_positives": [
            {"aspect": p.get("aspect"), "evidence": (p.get("evidence",""))[:80]}
            for p in guardrail.get("genuine_positives", [])[:4]
            if isinstance(p, dict)
        ],
        "genuine_concerns":  [
            {"aspect": c.get("aspect"), "severity": c.get("severity"),
             "evidence": (c.get("evidence",""))[:80]}
            for c in guardrail.get("genuine_concerns", [])[:3]
            if isinstance(c, dict)
        ],
        "score_breakdown":   final_scoring.get("score_breakdown", {}),
    }

    prompt = f"""You are an expert travel and experience advisor.
The Python scoring engine has ALREADY determined the verdict and score below.
Your job is to write the explanation text only — do NOT change the verdict or score.

ANALYSIS DATA:
{json.dumps(context, indent=2)}

Return EXACTLY this JSON (use the provided verdict and visit_score as-is):
{{
  "recommendation":   "{verdict}",
  "confidence":       {confidence},
  "visit_score":      {score},
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
  "visitor_tips":     ["tip 1", "tip 2"],
  "best_time":        "best time to visit or null",
  "data_reliability": "High|Medium|Low",
  "score_breakdown": {{
    "sentiment_score": {final_scoring["score_breakdown"].get("sentiment_comp", 0)},
    "rating_score":    {final_scoring["score_breakdown"].get("rating_comp", 0)},
    "trust_score":     {final_scoring["score_breakdown"].get("trust_comp", 0)},
    "composite":       {score}
  }}
}}"""

    # Explanation is prose generation — 8B model is sufficient and saves 70B daily quota
    result = call_groq(prompt, model=FAST_MODEL, max_tokens=2048)

    # Hard-enforce Python verdicts regardless of what Groq returned
    result["recommendation"] = verdict
    result["visit_score"]    = score
    result["confidence"]     = confidence
    result["score_breakdown"] = {
        "sentiment_score": final_scoring["score_breakdown"].get("sentiment_comp", 0),
        "rating_score":    final_scoring["score_breakdown"].get("rating_comp", 0),
        "trust_score":     final_scoring["score_breakdown"].get("trust_comp", 0),
        "composite":       score,
    }

    console.print(f"[green]✓  Explanation generated.[/green]")
    return result


# ── Report Display ─────────────────────────────────────────────────────────────

VERDICT_COLOR = {
    "HIGHLY RECOMMENDED": "bold green",
    "RECOMMENDED":        "green",
    "VISIT WITH CAUTION": "yellow",
    "NOT RECOMMENDED":    "red",
}

def display_report(location, place_info, reviews, sentiment, guardrail, rec, verification, review_stats):
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

    # Review stats
    rs = review_stats
    console.print(Panel(
        f"Raw scraped: {rs.get('raw_scraped_count', 0)}  |  "
        f"Usable text: {rs.get('usable_text_review_count', 0)}  |  "
        f"Cleaned: {rs.get('cleaned_review_count', 0)}  |  "
        f"Analyzed: {rs.get('analyzed_review_count', 0)}",
        title="[cyan]Review counts[/cyan]", expand=False,
    ))

    # Verification summary
    v_status  = verification.get("verification_status", "UNAVAILABLE")
    v_acc     = verification.get("accuracy")
    v_acc_str = f"{v_acc:.0%}" if v_acc is not None else "N/A"
    v_hall    = verification.get("hallucination_detected")
    v_corr    = len(verification.get("corrections", []))
    console.print(Panel(
        f"Model B status: {v_status}  |  Accuracy: {v_acc_str}  |  "
        f"Hallucination: {v_hall}  |  Corrections applied: {v_corr}",
        title="[magenta]Model B — Verification[/magenta]", expand=False,
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
        title="[yellow]⚡ Recommendation (Python scoring)[/yellow]", expand=False,
    ))

    # Pros / Cons
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

    console.print(Panel(
        rec.get("full_verdict", "No verdict generated."),
        title="[yellow]📝 Full verdict[/yellow]", expand=False,
    ))
    console.rule("[bold yellow]End of report[/bold yellow]")


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def analyze(location: str, max_reviews: int = 30, progress_callback=None) -> dict:
    """
    Complete two-model pipeline:
      1. Apify — scrape reviews
      2. Model A (Groq) — sentiment analysis  [batched, all reviews]
      3. Model A (Groq) — guardrail analysis
      4. Model B (OpenRouter) — independent verification
      5. apply_corrections — merge verified analysis
      6. Python scoring — deterministic final score + verdict
      7. Groq explanation — prose only, never overrides score
    """
    t0 = time.time()
    console.print(Panel(
        f"[bold]Location:[/bold]     {location}\n"
        f"[bold]Max reviews:[/bold]  {max_reviews}\n"
        f"[bold]Model A:[/bold]      Groq / {STRONG_MODEL} (sentiment) + {FAST_MODEL} (guardrail/explanation)\n"
        f"[bold]Model B:[/bold]      Gemini / {VERIFIER_MODEL} (independent verifier)\n"
        f"[bold]Verification:[/bold] {'required' if REQUIRE_VERIFICATION else 'preferred (not required)'}\n"
        f"[bold]Pipeline:[/bold]     Apify → Model A → Model B → Python scoring → Explanation",
        title="[bold blue]🌐 Location Review AI Analyzer — Two-Model Architecture[/bold blue]",
        expand=False,
    ))

    # ── Stage 1: Scrape ──
    if progress_callback:
        progress_callback(1, "Scraping Reviews", f"Collecting Google Maps reviews via Apify...")
    try:
        reviews, place_info, review_stats = scrape_reviews(location, max_reviews)
    except RuntimeError as e:
        return {"error": str(e)}

    if not reviews:
        return {"error": "No reviews found. Try a more specific location name."}

    # ── Stage 2: Model A — Sentiment (batched over all reviews) ──
    if progress_callback:
        progress_callback(2, "Model A — Sentiment", f"Analyzing {len(reviews)} reviews in batches...")
    sentiment = analyze_sentiment(reviews)
    time.sleep(1)

    # ── Stage 3: Model A — Guardrail ──
    if progress_callback:
        progress_callback(3, "Model A — Guardrail", "Checking review authenticity and trust score...")
    guardrail = guardrail_analysis(reviews, sentiment)
    time.sleep(1)

    # ── Stage 4: Model B — Independent Verification ──
    if progress_callback:
        progress_callback(4, "Model B — Verification", "OpenRouter independent verifier checking Model A...")
    verification = verify_analysis(reviews, sentiment, guardrail)

    # Handle REQUIRE_VERIFICATION
    if REQUIRE_VERIFICATION and verification.get("verification_status") == "UNAVAILABLE":
        return {
            "error": "Independent verification is currently unavailable.",
            "verification": verification,
            "place_info": place_info,
            "review_stats": review_stats,
        }

    # ── Stage 5: Apply corrections ──
    verified_sentiment, verified_guardrail = apply_corrections(
        sentiment, guardrail, verification
    )
    n_corrections = len([
        c for c in verification.get("corrections", [])
        if c.get("evidence_review_ids")
    ])
    if n_corrections:
        console.print(f"[cyan]  {n_corrections} evidence-supported correction(s) applied.[/cyan]")

    # Update analyzed count (= cleaned reviews actually fed to models)
    review_stats["analyzed_review_count"] = len(reviews)

    # ── Stage 6: Python scoring ──
    if progress_callback:
        progress_callback(5, "Python Scoring", "Computing deterministic final score and verdict...")
    final_scoring = calculate_final_score(
        reviews, place_info, verified_sentiment, verified_guardrail, verification
    )

    # ── Stage 7: Groq explanation ──
    if progress_callback:
        progress_callback(6, "Generating Explanation", "Writing pros, cons, visitor tips...")
    rec = generate_recommendation(
        location, place_info, reviews,
        verified_sentiment, verified_guardrail, final_scoring
    )

    # Display report
    display_report(
        location, place_info, reviews,
        verified_sentiment, verified_guardrail,
        rec, verification, review_stats
    )

    elapsed = time.time() - t0
    console.print(f"\n[dim]⏱  Total time: {elapsed:.1f}s[/dim]")

    # Save full JSON
    slug     = "".join(c if c.isalnum() or c in " _-" else "" for c in location)[:30].strip()
    out_file = f"report_{slug.replace(' ', '_')}.json"
    payload  = {
        "location":       location,
        "place_info":     place_info,
        "review_stats":   review_stats,
        "reviews":        reviews,
        "sentiment":      verified_sentiment,
        "guardrail":      verified_guardrail,
        "verification":   {
            "status":               verification.get("verification_status"),
            "accuracy":             verification.get("accuracy"),
            "hallucination_detected": verification.get("hallucination_detected"),
            "corrections_count":    n_corrections,
            "corrections":          verification.get("corrections", []),
            "verification_notes":   verification.get("verification_notes", ""),
        },
        "recommendation": rec,
        "model_info": {
            "model_a": {"provider": "Groq", "model": STRONG_MODEL,
                        "fast_model": FAST_MODEL},
            "model_b": {"provider": "Gemini", "model": VERIFIER_MODEL,
                        "status": verification.get("verification_status")},
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    console.print(f"[dim]💾 Report saved → {out_file}[/dim]")

    return payload


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    loc = ""
    n   = 30

    if len(sys.argv) > 1:
        if len(sys.argv) > 2 and sys.argv[-1].isdigit():
            loc = " ".join(sys.argv[1:-1])
            n   = int(sys.argv[-1])
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
            n_input = console.input("[bold]Max reviews (default 30):[/bold] ").strip()
            n = int(n_input) if n_input.isdigit() else 30
        except (EOFError, KeyboardInterrupt):
            n = 30

    analyze(loc, n)
