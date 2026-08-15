import streamlit as st
import os, json, time, re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

import location as loc_engine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Location Analyzer",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

    /* Verdict banner */
    .verdict-box {
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .verdict-highly-recommended {
        background: #052e16;
        border: 1px solid #10b981;
        color: #6ee7b7;
    }
    .verdict-recommended {
        background: #052e16;
        border: 1px solid #22c55e;
        color: #86efac;
    }
    .verdict-visit-with-caution {
        background: #431407;
        border: 1px solid #f97316;
        color: #fed7aa;
    }
    .verdict-not-recommended {
        background: #450a0a;
        border: 1px solid #ef4444;
        color: #fca5a5;
    }
    .verdict-title {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .verdict-quote {
        font-style: italic;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* Pro / Con cards */
    .pro-card {
        background: #052e1620;
        border-left: 3px solid #10b981;
        padding: 0.6rem 0.9rem;
        border-radius: 4px;
        margin-bottom: 0.4rem;
        color: #d1fae5;
        font-size: 0.9rem;
    }
    .con-card {
        background: #450a0a20;
        border-left: 3px solid #ef4444;
        padding: 0.6rem 0.9rem;
        border-radius: 4px;
        margin-bottom: 0.4rem;
        color: #fee2e2;
        font-size: 0.9rem;
    }

    /* Keyword pills */
    .tag-pill {
        display: inline-block;
        background: #1e293b;
        color: #cbd5e1;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem 0.2rem 0.2rem 0;
    }
    .tag-pos { background: #064e3b; color: #a7f3d0; }
    .tag-neg { background: #7f1d1d; color: #fecaca; }

    /* Place type card */
    .place-type-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }
    .place-type-icon { font-size: 2.2rem; line-height: 1; flex-shrink: 0; padding-top: 0.1rem; }
    .place-type-name { font-size: 1.2rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.1rem; }
    .place-type-category { font-size: 0.85rem; color: #38bdf8; font-weight: 600; margin-bottom: 0.3rem; }
    .place-type-subtypes { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: 0.35rem; }
    .place-type-tag {
        background: #1e293b;
        color: #94a3b8;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        font-size: 0.72rem;
    }
    .place-type-meta { font-size: 0.8rem; color: #475569; margin-top: 0.3rem; }
    .place-type-closed {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-left: 0.4rem;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📍 Location Analyzer")
    st.divider()

    apify_key = os.getenv("APIFY_API_TOKEN") or os.getenv("APify_API_TOKEN")
    groq_key  = os.getenv("GROQ_API_KEY")

    if not apify_key or not groq_key:
        st.warning("⚠️ Missing API keys in `.env`")

    max_reviews = st.slider(
        "Max reviews to scrape",
        min_value=10, max_value=100, value=30, step=5,
    )

    preset_clicked = None

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("## 📍 Location Review Analyzer")
st.caption("Search a place, scrape real Google Maps reviews, and get an AI-powered verdict.")
st.divider()


# ── Search ───────────────────────────────────────────────────────────────────

for _k, _v in {
    "city_val":           "",
    "city_sugg":          [],
    "confirmed_city":     "",
    "loc_val":            "",
    "loc_sugg":           [],
    "confirmed_location": "",
    "maps_url":           "",
    "search_mode":        "name",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Mode toggle
_mc1, _mc2 = st.columns(2)
with _mc1:
    if st.button("🔤 Search by Name", use_container_width=True,
                 type="primary" if st.session_state["search_mode"] == "name" else "secondary"):
        st.session_state["search_mode"] = "name"
with _mc2:
    if st.button("🔗 Paste Google Maps URL", use_container_width=True,
                 type="primary" if st.session_state["search_mode"] == "url" else "secondary"):
        st.session_state["search_mode"] = "url"

st.markdown("")

# ════════════════════════════════════════════════════════
# MODE A — City + Location
# ════════════════════════════════════════════════════════
if st.session_state["search_mode"] == "name":

    col_city, col_loc = st.columns([1, 2])

    # ── CITY ──────────────────────────────────────────────
    with col_city:
        # on_change fires on every keystroke in Streamlit 1.37+
        def _city_changed():
            val = st.session_state["_city_widget"]
            st.session_state["city_val"]       = val
            st.session_state["confirmed_city"] = ""
            # reset location when city changes
            st.session_state["loc_val"]            = ""
            st.session_state["confirmed_location"] = ""
            st.session_state["loc_sugg"]           = []
            if len(val.strip()) >= 2:
                st.session_state["city_sugg"] = loc_engine.get_city_suggestions(val.strip(), limit=6)
            else:
                st.session_state["city_sugg"] = []

        st.text_input(
            "🏙️ City / Area",
            value=st.session_state["city_val"],
            placeholder="e.g. Mumbai, Paris, Rome…",
            key="_city_widget",
            on_change=_city_changed,
            label_visibility="visible",
        )

        for _cs in st.session_state["city_sugg"]:
            _label = _cs if len(_cs) <= 32 else _cs[:29] + "…"
            if st.button(_label, key=f"csugg_{_cs}", use_container_width=True):
                _name = _cs.split(",")[0].strip()
                st.session_state.update({
                    "city_val": _name, "confirmed_city": _name,
                    "city_sugg": [], "loc_val": "",
                    "confirmed_location": "", "loc_sugg": [],
                })
                st.rerun()

    # ── LOCATION ──────────────────────────────────────────
    with col_loc:
        def _loc_changed():
            val  = st.session_state["_loc_widget"]
            city = st.session_state["confirmed_city"] or st.session_state["city_val"]
            st.session_state["loc_val"]            = val
            st.session_state["confirmed_location"] = ""
            if len(val.strip()) >= 2:
                st.session_state["loc_sugg"] = loc_engine.get_place_suggestions(
                    val.strip(), city=city.strip(), limit=7
                )
            else:
                st.session_state["loc_sugg"] = []

        st.text_input(
            "📍 Location / Place",
            value=st.session_state["loc_val"],
            placeholder="e.g. Cafe Goodluck, Eiffel Tower…",
            key="_loc_widget",
            on_change=_loc_changed,
            label_visibility="visible",
        )

        suggs = st.session_state["loc_sugg"]
        for i in range(0, min(len(suggs), 6), 2):
            row = suggs[i:i+2]
            _rc = st.columns(len(row))
            for _col, _s in zip(_rc, row):
                _lbl = _s["search_name"]
                _short = _lbl if len(_lbl) <= 34 else _lbl[:31] + "…"
                with _col:
                    if st.button(f"📍 {_short}", key=f"lsugg_{_lbl}", use_container_width=True):
                        st.session_state.update({
                            "confirmed_location": _lbl,
                            "loc_val": _lbl, "loc_sugg": [],
                        })
                        st.rerun()

    confirmed_loc  = st.session_state["confirmed_location"] or st.session_state["loc_val"].strip()
    confirmed_city = st.session_state["confirmed_city"]     or st.session_state["city_val"].strip()

    if confirmed_loc:
        if confirmed_city and confirmed_city.lower() not in confirmed_loc.lower():
            search_query = f"{confirmed_loc}, {confirmed_city}"
        else:
            search_query = confirmed_loc
        st.caption(f"🔎 Will analyze: **{search_query}**")
    else:
        search_query = ""

# ════════════════════════════════════════════════════════
# MODE B — Google Maps URL
# ════════════════════════════════════════════════════════
else:
    maps_url = st.text_input(
        "🔗 Google Maps URL",
        value=st.session_state["maps_url"],
        placeholder="https://www.google.com/maps/place/... or https://maps.app.goo.gl/...",
        key="maps_url_input",
    )
    st.session_state["maps_url"] = maps_url
    if maps_url.strip() and not maps_url.strip().startswith("http"):
        st.warning("⚠️ That doesn't look like a URL.")
    search_query = maps_url.strip()

col_btn, col_blank = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🚀 Analyze Location", type="primary", use_container_width=True)


# ── Execution Pipeline ────────────────────────────────────────────────────────
if analyze_btn:
    if not search_query.strip():
        st.warning("⚠️ Please enter a location name or Google Maps URL.")
    elif not apify_key or not groq_key:
        st.error("❌ Cannot start analysis. Missing API keys in `.env` file.")
    else:
        # Create progress container
        progress_box = st.container()
        with progress_box:
            st.subheader("⚡ Running AI Pipeline")
            pbar = st.progress(0)
            status_text = st.empty()
            
            def handle_progress(stage, title, desc):
                percent = int((stage / 4) * 100)
                pbar.progress(percent)
                status_text.markdown(f"**Stage {stage}/4 — {title}:** {desc}")
                
            try:
                t_start = time.time()
                report_data = loc_engine.analyze(search_query.strip(), max_reviews=max_reviews, progress_callback=handle_progress)
                t_elapsed = time.time() - t_start
                
                pbar.progress(100)
                status_text.success(f"✅ Analysis complete in {t_elapsed:.1f} seconds!")
                time.sleep(1)
                progress_box.empty() # clear progress bars
                
                st.session_state["report_data"] = report_data
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")


# ── Results Dashboard Display ─────────────────────────────────────────────────
if "report_data" in st.session_state and st.session_state["report_data"]:
    data = st.session_state["report_data"]
    place_info = data.get("place_info", {})
    reviews = data.get("reviews", [])
    sentiment = data.get("sentiment", {})
    guardrail = data.get("guardrail", {})
    rec = data.get("recommendation", {})
    
    st.divider()
    
    # 1. Executive Verdict Header Banner
    rec_label = rec.get("recommendation", "RECOMMENDED").upper()
    
    if "HIGHLY" in rec_label:
        v_class = "verdict-highly-recommended"
        v_icon = "🌟"
    elif "NOT" in rec_label:
        v_class = "verdict-not-recommended"
        v_icon = "🛑"
    elif "CAUTION" in rec_label:
        v_class = "verdict-visit-with-caution"
        v_icon = "⚠️"
    else:
        v_class = "verdict-recommended"
        v_icon = "✅"
        
    st.markdown(f"""
    <div class="verdict-box {v_class}">
        <div class="verdict-title">{v_icon} {rec_label}</div>
        <div class="verdict-quote">"{rec.get('one_line_verdict', 'Solid location based on review synthesis.')}"</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Place Type Card ────────────────────────────────────────────────────────
    category   = place_info.get("category", "")
    subtypes   = place_info.get("subtypes", [])
    price      = place_info.get("price", "")
    desc       = place_info.get("description", "")
    perm_closed = place_info.get("permanently_closed", False)
    temp_closed = place_info.get("temporarily_closed", False)

    # Pick an emoji based on the category text
    CAT_ICONS = {
        "restaurant": "🍽️", "food": "🍽️", "cafe": "☕", "coffee": "☕",
        "bar": "🍺", "pub": "🍺", "bakery": "🥐", "pizza": "🍕",
        "hotel": "🏨", "lodge": "🏨", "resort": "🏨",
        "mall": "🛍️", "shopping": "🛍️", "store": "🛒", "market": "🛒",
        "cinema": "🎬", "theatre": "🎭", "theater": "🎭", "movie": "🎬",
        "museum": "🏛️", "gallery": "🖼️", "art": "🖼️",
        "park": "🌳", "garden": "🌿", "beach": "🏖️", "nature": "🌿",
        "temple": "🛕", "church": "⛪", "mosque": "🕌", "religious": "🙏",
        "hospital": "🏥", "clinic": "🏥", "pharmacy": "💊",
        "gym": "💪", "fitness": "💪", "sport": "⚽", "stadium": "🏟️",
        "spa": "💆", "salon": "💇", "beauty": "💅",
        "school": "🏫", "university": "🎓", "college": "🎓",
        "bank": "🏦", "atm": "🏧",
        "airport": "✈️", "station": "🚉", "transit": "🚌",
        "amusement": "🎡", "play": "🎮", "game": "🎮", "arcade": "🕹️",
        "zoo": "🦁", "aquarium": "🐠",
        "monument": "🗽", "landmark": "🏛️", "historic": "🏰",
    }
    cat_lower  = (category + " " + " ".join(subtypes[:3])).lower()
    place_icon = "📍"
    for kw, icon in CAT_ICONS.items():
        if kw in cat_lower:
            place_icon = icon
            break

    # Build subtypes tags (deduplicate, skip the main category if already shown)
    tag_list = [s for s in subtypes if s.lower() != category.lower()][:6]
    tags_html = "".join(f'<span class="place-type-tag">{t}</span>' for t in tag_list)

    closed_badge = ""
    if perm_closed:
        closed_badge = '<span class="place-type-closed">🔴 PERMANENTLY CLOSED</span>'
    elif temp_closed:
        closed_badge = '<span class="place-type-closed">🟡 TEMPORARILY CLOSED</span>'

    meta_parts = []
    if price:
        meta_parts.append(f"💰 {price}")
    if place_info.get("phone"):
        meta_parts.append(f"📞 {place_info['phone']}")
    meta_str = "  ·  ".join(meta_parts)

    # Build optional inner blocks in Python first — avoids broken f-string HTML
    desc_html     = f'<div class="place-type-meta" style="color:#cbd5e1;font-size:0.9rem;margin-bottom:0.3rem;">{desc[:160]}{"…" if len(desc) > 160 else ""}</div>' if desc else ""
    subtypes_html = f'<div class="place-type-subtypes">{tags_html}</div>' if tag_list else ""
    meta_html     = f'<div class="place-type-meta">{meta_str}</div>' if meta_str else ""
    place_name    = place_info.get("name", search_query)

    st.markdown(
        f'<div class="place-type-card">'
        f'<div class="place-type-icon">{place_icon}</div>'
        f'<div class="place-type-body">'
        f'<div class="place-type-name">{place_name} {closed_badge}</div>'
        f'<div class="place-type-category">{category or "Place"}</div>'
        f'{desc_html}{subtypes_html}{meta_html}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 3. Place Meta Card ────────────────────────────────────────────────────
    with st.expander("📍 **Place Metadata**", expanded=True):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown(f"**Name:** {place_info.get('name', search_query)}")
            st.markdown(f"**Category:** {place_info.get('category') or 'N/A'}")
        with col_p2:
            st.markdown(f"**Google Score:** ⭐ {place_info.get('google_score', 'N/A')} / 5")
            rc = place_info.get('review_count', 'N/A')
            st.markdown(f"**Total Google Reviews:** {rc:,}" if isinstance(rc, (int, float)) else f"**Total Google Reviews:** {rc}")
        with col_p3:
            st.markdown(f"**Address:** {place_info.get('address') or 'N/A'}")
            web = place_info.get('website')
            if web:
                st.markdown(f"**Website:** [{web}]({web})")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4. Tabs ───────────────────────────────────────────────────────────────
    tab_verdict, tab_sentiment, tab_guardrail, tab_tips, tab_raw = st.tabs([
        "📋 Executive Verdict",
        "🧠 Deep Sentiment",
        "🛡️ Guardrail & Authenticity",
        "💡 Visitor Guide",
        "📄 Raw Data & Export",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Executive Verdict
    # ════════════════════════════════════════════════════════════════════════
    with tab_verdict:
        col_pro, col_con = st.columns(2)
        with col_pro:
            st.subheader("✅ Genuine Strengths")
            pros = rec.get("pros", [])
            if pros:
                for p in pros:
                    if isinstance(p, dict):
                        pt = p.get("point") or p.get("text") or str(p)
                        wt = p.get("weight", "")
                    else:
                        pt = str(p)
                        wt = ""
                    wt_badge = f" `[{wt}]`" if wt else ""
                    st.markdown(f'<div class="pro-card">✓ <b>{pt}</b>{wt_badge}</div>', unsafe_allow_html=True)
            else:
                st.info("No significant pros highlighted.")

        with col_con:
            st.subheader("❌ Concerns & Drawbacks")
            cons = rec.get("cons", [])
            if cons:
                for c in cons:
                    if isinstance(c, dict):
                        ct = c.get("point") or c.get("text") or str(c)
                        wt = c.get("weight", "")
                    else:
                        ct = str(c)
                        wt = ""
                    wt_badge = f" `[{wt}]`" if wt else ""
                    st.markdown(f'<div class="con-card">✗ <b>{ct}</b>{wt_badge}</div>', unsafe_allow_html=True)
            else:
                st.info("No major concerns reported.")

        st.subheader("📝 Balanced Assessment")
        st.info(rec.get("full_verdict", "Analysis completed."))

        # Guardrail summary inline
        gs = guardrail.get("guardrail_summary", "")
        ar = guardrail.get("analyst_recommendation", "")
        if gs or ar:
            note = " ".join(filter(None, [gs, ar]))
            st.warning(f"🔍 **Analyst Note:** {note}")

        # Score breakdown radar / bar
        st.subheader("📊 Score Breakdown")
        bd = rec.get("score_breakdown", {})
        asp = sentiment.get("aspect_scores", {})

        # Build unified score dataframe
        score_rows = []
        if bd:
            score_rows += [
                ("Sentiment",    bd.get("sentiment_score", 0)),
                ("Google Rating",bd.get("rating_score", 0)),
                ("Trust",        bd.get("trust_score", 0)),
                ("Composite",    bd.get("composite", 0)),
            ]
        for asp_key, asp_label in [
            ("food_quality","Food"), ("service","Service"),
            ("ambience","Ambience"), ("value_for_money","Value"),
            ("cleanliness","Cleanliness"), ("crowd_wait_time","Crowd/Wait"),
        ]:
            v = asp.get(asp_key, {})
            sc = v.get("score") if isinstance(v, dict) else None
            if sc is not None:
                score_rows.append((asp_label, sc))

        if score_rows:
            df_scores = pd.DataFrame(score_rows, columns=["Metric", "Score"])
            fig_scores = px.bar(
                df_scores, x="Metric", y="Score", text_auto=".1f",
                range_y=[0, 10],
                color="Score",
                color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
                color_continuous_midpoint=5,
            )
            fig_scores.update_layout(
                showlegend=False, coloraxis_showscale=False,
                height=320, margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        # Standout quotes
        sq_pos = sentiment.get("standout_positive_quote", "")
        sq_neg = sentiment.get("standout_negative_quote", "")
        if sq_pos or sq_neg:
            st.subheader("💬 Standout Reviewer Quotes")
            qc1, qc2 = st.columns(2)
            with qc1:
                if sq_pos:
                    st.markdown(
                        f'<div class="pro-card">🌟 <i>"{sq_pos}"</i></div>',
                        unsafe_allow_html=True,
                    )
            with qc2:
                if sq_neg:
                    st.markdown(
                        f'<div class="con-card">⚠️ <i>"{sq_neg}"</i></div>',
                        unsafe_allow_html=True,
                    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Deep Sentiment
    # ════════════════════════════════════════════════════════════════════════
    with tab_sentiment:

        # ── Row 1: Emotion distribution + Rating distribution ──
        sc1, sc2 = st.columns(2)

        with sc1:
            st.subheader("🎭 Emotion Distribution")
            emo_dist = sentiment.get("emotion_distribution", {})
            if emo_dist and any(emo_dist.values()):
                df_emo = pd.DataFrame({
                    "Emotion": list(emo_dist.keys()),
                    "Count":   list(emo_dist.values()),
                })
                df_emo = df_emo[df_emo["Count"] > 0]
                fig_emo = px.pie(
                    df_emo, names="Emotion", values="Count",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4,
                )
                fig_emo.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_emo, use_container_width=True)
            else:
                st.info("Emotion data not available.")

        with sc2:
            st.subheader("⭐ Rating Distribution")
            counts = sentiment.get("rating_counts", {})
            if counts:
                df_r = pd.DataFrame({
                    "Stars": [f"⭐ {k}" for k in sorted(counts.keys(), reverse=True)],
                    "Count": [counts[k] for k in sorted(counts.keys(), reverse=True)],
                })
                fig_r = px.bar(
                    df_r, x="Stars", y="Count", text_auto=True,
                    color="Count",
                    color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
                )
                fig_r.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    height=280, margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_r, use_container_width=True)

        # ── Row 2: Aspect scores radar ──
        st.subheader("🔬 Aspect-Based Scores (0 – 10)")
        asp = sentiment.get("aspect_scores", {})
        asp_map = {
            "food_quality": "Food", "service": "Service",
            "ambience": "Ambience", "value_for_money": "Value",
            "cleanliness": "Cleanliness", "accessibility": "Accessibility",
            "crowd_wait_time": "Crowd / Wait",
        }
        asp_rows = []
        for key, label in asp_map.items():
            v = asp.get(key, {})
            sc  = v.get("score")    if isinstance(v, dict) else None
            cnt = v.get("reviews_mentioning", 0) if isinstance(v, dict) else 0
            summ= v.get("summary", "") if isinstance(v, dict) else ""
            if sc is not None:
                asp_rows.append((label, sc, cnt, summ or "—"))

        if asp_rows:
            # Radar chart
            labels_r = [r[0] for r in asp_rows]
            values_r = [r[1] for r in asp_rows]
            fig_radar = go.Figure(go.Scatterpolar(
                r=values_r + [values_r[0]],
                theta=labels_r + [labels_r[0]],
                fill="toself",
                fillcolor="rgba(56,189,248,0.2)",
                line=dict(color="#38bdf8", width=2),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 10], tickfont=dict(size=10))),
                showlegend=False,
                height=320,
                margin=dict(l=30, r=30, t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Aspect detail table
            for label, sc, cnt, summ in asp_rows:
                color = "#22c55e" if sc >= 7 else "#f59e0b" if sc >= 5 else "#ef4444"
                bar_w = int(sc * 10)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'<span style="width:110px;font-size:0.85rem;color:#94a3b8;">{label}</span>'
                    f'<div style="flex:1;background:#1e293b;border-radius:4px;height:14px;">'
                    f'<div style="width:{bar_w}%;background:{color};height:14px;border-radius:4px;"></div></div>'
                    f'<span style="width:32px;text-align:right;font-weight:700;color:{color};font-size:0.9rem;">{sc:.1f}</span>'
                    f'<span style="font-size:0.75rem;color:#64748b;width:80px;">({cnt} reviews)</span>'
                    f'<span style="font-size:0.8rem;color:#cbd5e1;flex:2;">{summ}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Aspect scores not available.")

        # ── Row 3: Temporal trend ──
        st.subheader("📈 Temporal Sentiment Trend")
        tt = sentiment.get("temporal_trend", {})
        if tt:
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.metric("Recent Sentiment", tt.get("recent_sentiment", "—"),
                          delta=f"{tt.get('recent_score', 0):.2f}")
            with tc2:
                st.metric("Older Sentiment", tt.get("older_sentiment", "—"),
                          delta=f"{tt.get('older_score', 0):.2f}")
            with tc3:
                trend_icon = {"Improving": "📈", "Declining": "📉", "Stable": "➡️"}.get(
                    tt.get("trend", ""), "—")
                st.metric("Overall Trend", f"{trend_icon} {tt.get('trend', '—')}")
            if tt.get("trend_explanation"):
                st.caption(f"💡 {tt['trend_explanation']}")

        # ── Row 4: Crowd profile ──
        st.subheader("👥 Crowd Profile")
        cp = sentiment.get("crowd_profile", {})
        if cp:
            st.markdown(f"**Dominant visitor type:** {cp.get('dominant_visitor_type', '—')}")
            if cp.get("mention_evidence"):
                st.caption(f"Evidence: {cp['mention_evidence']}")
            if cp.get("accessibility_notes"):
                st.info(f"♿ Accessibility: {cp['accessibility_notes']}")

        # ── Row 5: Themes ──
        st.subheader("🎯 Key Themes")
        themes = sentiment.get("themes", [])
        if themes:
            for t in themes:
                t_name  = t.get("name", "Theme")       if isinstance(t, dict) else str(t)
                t_sent  = t.get("sentiment", "Neutral") if isinstance(t, dict) else ""
                t_ev    = t.get("evidence", "")         if isinstance(t, dict) else ""
                t_quote = t.get("representative_quote","") if isinstance(t, dict) else ""
                t_freq  = t.get("frequency", 0)         if isinstance(t, dict) else 0
                color   = "#22c55e" if "Pos" in t_sent else "#ef4444" if "Neg" in t_sent else "#f59e0b"
                st.markdown(
                    f'<div style="border-left:4px solid {color};padding:0.6rem 1rem;'
                    f'background:#1e293b;border-radius:0 6px 6px 0;margin-bottom:8px;">'
                    f'<b style="color:{color};">{t_name}</b>'
                    f'<span style="font-size:0.75rem;color:#64748b;margin-left:8px;">'
                    f'{t_sent} · {t_freq} mentions</span><br>'
                    f'<span style="color:#cbd5e1;font-size:0.85rem;">{t_ev}</span>'
                    + (f'<br><i style="color:#94a3b8;font-size:0.8rem;">"{t_quote}"</i>' if t_quote else "")
                    + '</div>',
                    unsafe_allow_html=True,
                )

        # ── Row 6: Keywords ──
        kc1, kc2 = st.columns(2)
        with kc1:
            st.subheader("👍 Positive Keywords")
            pos_kws = sentiment.get("positive_keywords", [])
            if pos_kws:
                st.markdown("".join(f'<span class="tag-pill tag-pos">👍 {kw}</span>' for kw in pos_kws),
                            unsafe_allow_html=True)
        with kc2:
            st.subheader("👎 Negative Keywords")
            neg_kws = sentiment.get("negative_keywords", [])
            if neg_kws:
                st.markdown("".join(f'<span class="tag-pill tag-neg">👎 {kw}</span>' for kw in neg_kws),
                            unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Guardrail & Authenticity
    # ════════════════════════════════════════════════════════════════════════
    with tab_guardrail:

        # ── Top metrics row ──
        gm1, gm2, gm3, gm4, gm5 = st.columns(5)
        with gm1:
            st.metric("Trust Score",    f"{guardrail.get('trust_score', 0):.0%}")
        with gm2:
            st.metric("Fake Risk",      f"{guardrail.get('fake_review_probability', 0):.0%}")
        with gm3:
            st.metric("Review Quality", guardrail.get("review_quality", "—"))
        with gm4:
            st.metric("Bias Level",     guardrail.get("bias_level", "—"))
        with gm5:
            adj = guardrail.get("rating_integrity", {}).get("adjusted_true_rating", "—")
            st.metric("Adj. True Rating", f"{adj}/5" if adj != "—" else "—")

        st.divider()

        # ── Heuristic stats ──
        hs = guardrail.get("heuristic_stats", {})
        if hs:
            st.subheader("🔢 Review Statistics")
            hc1, hc2, hc3, hc4 = st.columns(4)
            with hc1: st.metric("Total Analyzed",   hs.get("total_reviews", "—"))
            with hc2: st.metric("Avg Star Rating",  hs.get("avg_rating", "—"))
            with hc3: st.metric("Avg Review Length",f"{int(hs.get('avg_review_length', 0))} chars")
            with hc4: st.metric("Duplicate Pairs",  hs.get("duplicate_pairs", 0))

        # ── Linguistic & rating analysis side by side ──
        la_col, ri_col = st.columns(2)

        with la_col:
            st.subheader("🔤 Linguistic Analysis")
            la = guardrail.get("linguistic_analysis", {})
            if la:
                for label, val in [
                    ("Templated language", "✅ Detected" if la.get("templated_language_detected") else "✗ Not detected"),
                    ("Vocabulary diversity", la.get("vocabulary_diversity", "—")),
                    ("Writing style",        la.get("writing_style_consistency", "—")),
                ]:
                    col_color = "#ef4444" if "Detected" in str(val) or "Consistent" in str(val) else "#22c55e"
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                        f'border-bottom:1px solid #1e293b;">'
                        f'<span style="color:#94a3b8;font-size:0.85rem;">{label}</span>'
                        f'<span style="color:{col_color};font-size:0.85rem;font-weight:600;">{val}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if la.get("copy_paste_evidence"):
                    st.warning(f"📋 Copy-paste evidence: {la['copy_paste_evidence']}")
                if la.get("language_notes"):
                    st.caption(f"💬 {la['language_notes']}")

        with ri_col:
            st.subheader("⭐ Rating Integrity")
            ri = guardrail.get("rating_integrity", {})
            if ri:
                natural = ri.get("distribution_natural", True)
                st.markdown(
                    f'<div style="color:{"#22c55e" if natural else "#ef4444"};font-weight:600;margin-bottom:8px;">'
                    f'{"✅ Distribution appears natural" if natural else "⚠️ Unnatural rating distribution"}</div>',
                    unsafe_allow_html=True,
                )
                inf = ri.get("inflated_stars_estimate", 0)
                sup = ri.get("suppressed_stars_estimate", 0)
                if inf:
                    st.markdown(f"**~{inf} inflated** 5-star reviews estimated")
                if sup:
                    st.markdown(f"**~{sup} suppressed** low-star reviews estimated")
                for anomaly in ri.get("anomalies", []):
                    st.warning(f"⚠️ {anomaly}")
                if ri.get("rating_notes"):
                    st.caption(ri["rating_notes"])

        # ── Reviewer behavior ──
        st.subheader("👤 Reviewer Behaviour")
        rb = guardrail.get("reviewer_behavior", {})
        if rb:
            rbc1, rbc2 = st.columns(2)
            with rbc1:
                risk = rb.get("sock_puppet_risk", "—")
                color = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}.get(risk, "#94a3b8")
                st.markdown(f'<b>Sock-puppet risk:</b> <span style="color:{color};font-weight:700;">{risk}</span>', unsafe_allow_html=True)
            with rbc2:
                risk2 = rb.get("coordinated_posting_risk", "—")
                color2 = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}.get(risk2, "#94a3b8")
                st.markdown(f'<b>Coordinated posting risk:</b> <span style="color:{color2};font-weight:700;">{risk2}</span>', unsafe_allow_html=True)
            if rb.get("evidence"):
                st.caption(f"Evidence: {rb['evidence']}")

        st.divider()

        # ── Contradictions ──
        contradictions = guardrail.get("contradiction_analysis", [])
        if contradictions:
            st.subheader("⚖️ Contradictions in Reviews")
            for c in contradictions:
                if not isinstance(c, dict): continue
                with st.expander(f"⚖️ **{c.get('aspect', 'Aspect')}**", expanded=False):
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.markdown(f'<div class="pro-card">👍 {c.get("positive_claim","—")}</div>', unsafe_allow_html=True)
                    with cc2:
                        st.markdown(f'<div class="con-card">👎 {c.get("negative_claim","—")}</div>', unsafe_allow_html=True)
                    st.info(f"🔍 Resolution: {c.get('resolution','—')}")

        # ── Per-aspect credibility ──
        ac = guardrail.get("aspect_credibility", {})
        if ac:
            st.subheader("🔬 Per-Aspect Credibility")
            ac_rows = []
            for key in ["food_quality","service","ambience","value_for_money","cleanliness","crowd_wait_time"]:
                v = ac.get(key, {})
                if isinstance(v, dict):
                    ac_rows.append({
                        "Aspect":     key.replace("_"," ").title(),
                        "Credible":   "✅" if v.get("credible") else "❌",
                        "Confidence": f"{v.get('confidence',0):.0%}",
                        "Note":       v.get("note", "—"),
                    })
            if ac_rows:
                st.dataframe(pd.DataFrame(ac_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Suspicious patterns ──
        patterns = guardrail.get("suspicious_patterns", [])
        st.subheader("🚨 Suspicious Patterns")
        if patterns:
            for pat in patterns:
                st.warning(f"⚠️ {pat}")
        else:
            st.success("✅ No suspicious review manipulation patterns detected.")

        # ── Verified genuine positives & concerns ──
        vp_col, vc_col = st.columns(2)
        with vp_col:
            st.subheader("✅ Verified Genuine Positives")
            for p in guardrail.get("genuine_positives", []):
                if not isinstance(p, dict): continue
                ids = p.get("supporting_review_ids", [])
                ids_str = f" *(reviews {ids})*" if ids else ""
                st.markdown(
                    f'<div class="pro-card"><b>{p.get("aspect","—")}</b>'
                    f' <span style="opacity:0.7;font-size:0.8rem;">conf {p.get("confidence",0):.0%}{ids_str}</span><br>'
                    f'{p.get("evidence","")}</div>',
                    unsafe_allow_html=True,
                )
        with vc_col:
            st.subheader("⚠️ Verified Genuine Concerns")
            for c in guardrail.get("genuine_concerns", []):
                if not isinstance(c, dict): continue
                sev = c.get("severity", "Minor")
                sev_color = {"Major": "#ef4444", "Moderate": "#f59e0b", "Minor": "#94a3b8"}.get(sev, "#94a3b8")
                ids = c.get("supporting_review_ids", [])
                ids_str = f" *(reviews {ids})*" if ids else ""
                st.markdown(
                    f'<div class="con-card"><b>{c.get("aspect","—")}</b>'
                    f' <span style="color:{sev_color};font-size:0.8rem;font-weight:700;">[{sev}]{ids_str}</span><br>'
                    f'{c.get("evidence","")}</div>',
                    unsafe_allow_html=True,
                )

        # ── Verified facts ──
        vf = guardrail.get("verified_facts", [])
        if vf:
            st.subheader("📌 Verified Facts")
            for f in vf:
                st.markdown(f"📌 {f}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Visitor Guide
    # ════════════════════════════════════════════════════════════════════════
    with tab_tips:
        tg1, tg2 = st.columns(2)
        with tg1:
            st.subheader("🎯 Best For")
            for b in rec.get("best_for", []):
                st.markdown(f"• 👥 {b}")
        with tg2:
            st.subheader("🚫 Avoid If")
            for a in rec.get("avoid_if", []):
                st.markdown(f"• ⚠️ {a}")

        st.divider()
        st.subheader("⏰ Best Time to Visit")
        st.info(f"🕒 {rec.get('best_time') or 'Not determinable from reviews.'}")

        st.subheader("💡 Actionable Visitor Tips")
        for tip in rec.get("visitor_tips", []):
            st.markdown(
                f'<div style="background:#1e293b;border-left:4px solid #38bdf8;padding:0.6rem 1rem;'
                f'border-radius:0 6px 6px 0;margin-bottom:6px;color:#e2e8f0;">💡 {tip}</div>',
                unsafe_allow_html=True,
            )

        # Crowd profile repeat here for convenience
        cp = sentiment.get("crowd_profile", {})
        if cp:
            st.divider()
            st.subheader("👥 Visitor Profile")
            st.markdown(f"**Typical visitor:** {cp.get('dominant_visitor_type','—')}")
            if cp.get("accessibility_notes"):
                st.info(f"♿ {cp['accessibility_notes']}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — Raw Data & Export
    # ════════════════════════════════════════════════════════════════════════
    with tab_raw:
        st.subheader("📄 Scraped Reviews")
        if reviews:
            df_revs = pd.DataFrame(reviews)
            st.dataframe(df_revs, use_container_width=True, height=350)

        st.divider()
        st.subheader("💾 Export Full Report")
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        loc_slug = "".join(c if c.isalnum() or c in " _-" else "" for c in search_query)[:30].strip().replace(" ", "_")
        st.download_button(
            label="📥 Download Full Report (.json)",
            data=json_str,
            file_name=f"report_{loc_slug}.json",
            mime="application/json",
            type="primary",
        )
