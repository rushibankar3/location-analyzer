import streamlit as st
import os, json, time, re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

import location as loc_engine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Location Analyzer",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

    .verdict-box { border-radius: 10px; padding: 1.2rem 1.6rem; margin-bottom: 1.2rem; }
    .verdict-highly-recommended { background:#052e16; border:1px solid #10b981; color:#6ee7b7; }
    .verdict-recommended        { background:#052e16; border:1px solid #22c55e; color:#86efac; }
    .verdict-visit-with-caution { background:#431407; border:1px solid #f97316; color:#fed7aa; }
    .verdict-not-recommended    { background:#450a0a; border:1px solid #ef4444; color:#fca5a5; }
    .verdict-title  { font-size:1.4rem; font-weight:700; letter-spacing:0.5px; margin-bottom:0.3rem; }
    .verdict-quote  { font-style:italic; font-size:0.95rem; opacity:0.9; }

    .pro-card {
        background:#052e1620; border-left:3px solid #10b981;
        padding:0.6rem 0.9rem; border-radius:4px; margin-bottom:0.4rem;
        color:#d1fae5; font-size:0.9rem;
    }
    .con-card {
        background:#450a0a20; border-left:3px solid #ef4444;
        padding:0.6rem 0.9rem; border-radius:4px; margin-bottom:0.4rem;
        color:#fee2e2; font-size:0.9rem;
    }

    .tag-pill { display:inline-block; background:#1e293b; color:#cbd5e1;
                padding:0.25rem 0.65rem; border-radius:20px; font-size:0.8rem;
                margin:0.2rem 0.2rem 0.2rem 0; }
    .tag-pos  { background:#064e3b; color:#a7f3d0; }
    .tag-neg  { background:#7f1d1d; color:#fecaca; }

    .place-type-card {
        background:#0f172a; border:1px solid #1e293b; border-radius:10px;
        padding:1rem 1.4rem; margin-bottom:1.2rem;
        display:flex; align-items:flex-start; gap:1rem;
    }
    .place-type-icon     { font-size:2.2rem; line-height:1; flex-shrink:0; padding-top:0.1rem; }
    .place-type-name     { font-size:1.2rem; font-weight:700; color:#f1f5f9; margin-bottom:0.1rem; }
    .place-type-category { font-size:0.85rem; color:#38bdf8; font-weight:600; margin-bottom:0.3rem; }
    .place-type-subtypes { display:flex; flex-wrap:wrap; gap:0.25rem; margin-top:0.35rem; }
    .place-type-tag      { background:#1e293b; color:#94a3b8; padding:0.15rem 0.5rem;
                           border-radius:20px; font-size:0.72rem; }
    .place-type-meta     { font-size:0.8rem; color:#475569; margin-top:0.3rem; }
    .place-type-closed   { background:#7f1d1d; color:#fca5a5; padding:0.1rem 0.5rem;
                           border-radius:4px; font-size:0.72rem; font-weight:700; margin-left:0.4rem; }

    [data-testid="stSidebar"] { background:#0f172a; }
    [data-testid="stSidebar"] .stMarkdown p { color:#94a3b8; font-size:0.82rem; }
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

    st.markdown("**Model A — Analyst**")
    st.caption(f"Groq / {os.getenv('STRONG_MODEL','llama-3.3-70b-versatile')} + {os.getenv('FAST_MODEL','llama-3.1-8b-instant')}")
    st.markdown("**Model B — Verifier**")
    verifier_model = os.getenv("VERIFIER_MODEL", "llama-3.1-8b-instant")
    groq_key = os.getenv("GROQ_API_KEY", "")
    st.caption(f"Groq / {verifier_model}")
    if not groq_key:
        st.warning("⚠️ GROQ_API_KEY not set")
    st.divider()

    max_reviews = st.slider(
        "Max reviews to scrape",
        min_value=10, max_value=100, value=30, step=5,
    )

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("## 📍 Location Review Analyzer")
st.caption("Search a place, scrape real Google Maps reviews, and get a two-model AI verdict.")
st.divider()

# ── Session state defaults ────────────────────────────────────────────────────
for _k, _v in {
    "city_val": "", "city_sugg": [], "confirmed_city": "",
    "loc_val": "", "loc_sugg": [], "confirmed_location": "",
    "maps_url": "", "search_mode": "name",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Search mode toggle ────────────────────────────────────────────────────────
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
# MODE A — City + Location name search
# ════════════════════════════════════════════════════════
if st.session_state["search_mode"] == "name":

    col_city, col_loc = st.columns([1, 2])

    with col_city:
        def _city_changed():
            val = st.session_state["_city_widget"]
            st.session_state.update({
                "city_val": val, "confirmed_city": "",
                "loc_val": "", "confirmed_location": "", "loc_sugg": [],
            })
            st.session_state["city_sugg"] = (
                loc_engine.get_city_suggestions(val.strip(), limit=6)
                if len(val.strip()) >= 2 else []
            )

        st.text_input("🏙️ City / Area", value=st.session_state["city_val"],
                      placeholder="e.g. Mumbai, Paris, Rome…",
                      key="_city_widget", on_change=_city_changed)

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

    with col_loc:
        def _loc_changed():
            val  = st.session_state["_loc_widget"]
            city = st.session_state["confirmed_city"] or st.session_state["city_val"]
            st.session_state.update({"loc_val": val, "confirmed_location": ""})
            st.session_state["loc_sugg"] = (
                loc_engine.get_place_suggestions(val.strip(), city=city.strip(), limit=7)
                if len(val.strip()) >= 2 else []
            )

        st.text_input("📍 Location / Place", value=st.session_state["loc_val"],
                      placeholder="e.g. Cafe Goodluck, Eiffel Tower…",
                      key="_loc_widget", on_change=_loc_changed)

        suggs = st.session_state["loc_sugg"]
        for i in range(0, min(len(suggs), 6), 2):
            row = suggs[i:i+2]
            _rc = st.columns(len(row))
            for _col, _s in zip(_rc, row):
                _lbl   = _s["search_name"]
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
        search_query = (
            f"{confirmed_loc}, {confirmed_city}"
            if confirmed_city and confirmed_city.lower() not in confirmed_loc.lower()
            else confirmed_loc
        )
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
        st.warning("⚠️ That doesn't look like a valid URL.")
    search_query = maps_url.strip()

col_btn, col_blank = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🚀 Analyze Location", type="primary", use_container_width=True)

# ── Execution Pipeline ────────────────────────────────────────────────────────
if analyze_btn:
    if not search_query.strip():
        st.warning("⚠️ Please enter a location name or Google Maps URL.")
    elif not apify_key or not groq_key:
        st.error("❌ Cannot start. Missing APIFY_API_TOKEN or GROQ_API_KEY in `.env`.")
    else:
        progress_box = st.container()
        with progress_box:
            st.subheader("⚡ Running Two-Model AI Pipeline")
            pbar        = st.progress(0)
            status_text = st.empty()

            STAGE_LABELS = {
                1: ("Scraping Reviews",      "Collecting Google Maps reviews via Apify…"),
                2: ("Model A — Sentiment",   "Analyzing all reviews in batches (llama-3.3-70b)…"),
                3: ("Model A — Guardrail",   "Checking review authenticity and trust score…"),
                4: ("Model B — Verification","Independent verifier checking Model A (llama-3.1-8b)…"),
                5: ("Python Scoring",        "Deterministic final score and verdict…"),
                6: ("Explanation",           "Writing pros, cons, visitor tips…"),
            }

            def handle_progress(stage, title, desc):
                pbar.progress(int(stage / 6 * 100))
                status_text.markdown(
                    f"**Stage {stage}/6 — {title}:** {desc}"
                )

            try:
                t_start     = time.time()
                report_data = loc_engine.analyze(
                    search_query.strip(),
                    max_reviews=max_reviews,
                    progress_callback=handle_progress,
                )
                t_elapsed = time.time() - t_start

                if not report_data:
                    st.error("❌ Analysis returned no data. Check the location name or URL.")
                elif report_data.get("error"):
                    pbar.progress(100)
                    st.error(f"❌ {report_data['error']}")
                    if report_data.get("verification"):
                        v = report_data["verification"]
                        st.warning(v.get("verification_notes") or v.get("reason", ""))
                else:
                    pbar.progress(100)
                    status_text.success(f"✅ Complete in {t_elapsed:.1f}s")
                    time.sleep(0.8)
                    progress_box.empty()
                    st.session_state["report_data"] = report_data

            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")

# ── Results Dashboard ─────────────────────────────────────────────────────────
if "report_data" in st.session_state and st.session_state["report_data"]:
    data = st.session_state["report_data"]

    place_info   = data.get("place_info", {})
    reviews      = data.get("reviews", [])
    sentiment    = data.get("sentiment", {})
    guardrail    = data.get("guardrail", {})
    rec          = data.get("recommendation", {})
    review_stats = data.get("review_stats", {})
    verification = data.get("verification", {})
    model_info   = data.get("model_info", {})

    st.divider()

    # ── Review counts ─────────────────────────────────────────────────────────
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1: st.metric("Raw Scraped",    review_stats.get("raw_scraped_count",        len(reviews)))
    with rc2: st.metric("Usable Reviews", review_stats.get("usable_text_review_count", len(reviews)))
    with rc3: st.metric("After Cleaning", review_stats.get("cleaned_review_count",     len(reviews)))
    with rc4: st.metric("Analyzed",       review_stats.get("analyzed_review_count",    len(reviews)))

    # ── Model info strip ──────────────────────────────────────────────────────
    ma = model_info.get("model_a", {})
    mb = model_info.get("model_b", {})
    st.markdown(
        f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;'
        f'padding:0.5rem 1rem;margin:0.4rem 0;display:flex;gap:2.5rem;flex-wrap:wrap;">'
        f'<span style="color:#64748b;font-size:0.8rem;">🧠 <b style="color:#38bdf8;">Model A</b> '
        f'{ma.get("provider","Groq")} / {ma.get("model","llama-3.3-70b-versatile")}'
        f' + {ma.get("fast_model","llama-3.1-8b-instant")}</span>'
        f'<span style="color:#64748b;font-size:0.8rem;">🔍 <b style="color:#a78bfa;">Model B</b> '
        f'{mb.get("provider","Groq")} / {mb.get("model","llama-3.1-8b-instant")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Verification banner ───────────────────────────────────────────────────
    v_status   = verification.get("status", "UNAVAILABLE")
    v_accuracy = verification.get("accuracy")
    v_hall     = verification.get("hallucination_detected")
    v_corr     = verification.get("corrections_count", 0)
    v_notes    = verification.get("verification_notes", "")

    V_STYLE = {
        "PASS":        ("#064e3b", "#10b981", "✅"),
        "CORRECTED":   ("#422006", "#f59e0b", "🔧"),
        "FAIL":        ("#450a0a", "#ef4444", "❌"),
        "UNAVAILABLE": ("#1e1b4b", "#6366f1", "⚠️"),
    }
    v_bg, v_border, v_icon = V_STYLE.get(v_status, ("#1e293b", "#64748b", "ℹ️"))
    v_acc_str  = f"{v_accuracy:.0%}" if v_accuracy is not None else "N/A"
    v_hall_str = ("Yes 🚨" if v_hall else "No ✅") if v_hall is not None else "N/A"

    st.markdown(
        f'<div style="background:{v_bg};border:1px solid {v_border};border-radius:8px;'
        f'padding:0.7rem 1.2rem;margin:0.5rem 0 1rem 0;display:flex;flex-wrap:wrap;gap:1.5rem;align-items:center;">'
        f'<span style="color:{v_border};font-weight:700;font-size:0.95rem;">{v_icon} Model B: {v_status}</span>'
        f'<span style="color:#94a3b8;font-size:0.82rem;">Accuracy <b style="color:#e2e8f0;">{v_acc_str}</b></span>'
        f'<span style="color:#94a3b8;font-size:0.82rem;">Hallucination <b style="color:#e2e8f0;">{v_hall_str}</b></span>'
        f'<span style="color:#94a3b8;font-size:0.82rem;">Corrections applied <b style="color:#e2e8f0;">{v_corr}</b></span>'
        + (f'<span style="color:#64748b;font-size:0.78rem;font-style:italic;">{v_notes}</span>' if v_notes else "")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Verdict banner ────────────────────────────────────────────────────────
    rec_label = rec.get("recommendation", "RECOMMENDED").upper()
    if "HIGHLY" in rec_label:
        v_class, v_icon_v = "verdict-highly-recommended", "🌟"
    elif "NOT" in rec_label:
        v_class, v_icon_v = "verdict-not-recommended",    "🛑"
    elif "CAUTION" in rec_label:
        v_class, v_icon_v = "verdict-visit-with-caution", "⚠️"
    else:
        v_class, v_icon_v = "verdict-recommended",        "✅"

    st.markdown(
        f'<div class="verdict-box {v_class}">'
        f'<div class="verdict-title">{v_icon_v} {rec_label}</div>'
        f'<div class="verdict-quote">"{rec.get("one_line_verdict","Solid location based on review synthesis.")}"</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Place type card ───────────────────────────────────────────────────────
    category    = place_info.get("category", "")
    subtypes    = place_info.get("subtypes", [])
    perm_closed = place_info.get("permanently_closed", False)
    temp_closed = place_info.get("temporarily_closed", False)
    desc        = place_info.get("description", "")
    price       = place_info.get("price", "")

    CAT_ICONS = {
        "restaurant":"🍽️","food":"🍽️","cafe":"☕","coffee":"☕",
        "bar":"🍺","pub":"🍺","bakery":"🥐","pizza":"🍕",
        "hotel":"🏨","lodge":"🏨","resort":"🏨",
        "mall":"🛍️","shopping":"🛍️","store":"🛒","market":"🛒",
        "cinema":"🎬","theatre":"🎭","theater":"🎭","movie":"🎬",
        "museum":"🏛️","gallery":"🖼️","art":"🖼️",
        "park":"🌳","garden":"🌿","beach":"🏖️","nature":"🌿",
        "temple":"🛕","church":"⛪","mosque":"🕌","religious":"🙏",
        "hospital":"🏥","clinic":"🏥","pharmacy":"💊",
        "gym":"💪","fitness":"💪","sport":"⚽","stadium":"🏟️",
        "spa":"💆","salon":"💇","beauty":"💅",
        "school":"🏫","university":"🎓","college":"🎓",
        "bank":"🏦","atm":"🏧","airport":"✈️","station":"🚉",
        "amusement":"🎡","zoo":"🦁","aquarium":"🐠",
        "monument":"🗽","landmark":"🏛️","historic":"🏰",
    }
    cat_lower  = (category + " " + " ".join(subtypes[:3])).lower()
    place_icon = next((icon for kw, icon in CAT_ICONS.items() if kw in cat_lower), "📍")

    tag_list    = [s for s in subtypes if s.lower() != category.lower()][:6]
    tags_html   = "".join(f'<span class="place-type-tag">{t}</span>' for t in tag_list)
    closed_badge = (
        '<span class="place-type-closed">🔴 PERMANENTLY CLOSED</span>' if perm_closed else
        '<span class="place-type-closed">🟡 TEMPORARILY CLOSED</span>'  if temp_closed else ""
    )
    meta_parts  = ([f"💰 {price}"] if price else []) + ([f"📞 {place_info['phone']}"] if place_info.get("phone") else [])
    desc_html     = f'<div class="place-type-meta" style="color:#cbd5e1;font-size:0.9rem;margin-bottom:0.3rem;">{desc[:160]}{"…" if len(desc)>160 else ""}</div>' if desc else ""
    subtypes_html = f'<div class="place-type-subtypes">{tags_html}</div>' if tag_list else ""
    meta_html     = f'<div class="place-type-meta">{"  ·  ".join(meta_parts)}</div>' if meta_parts else ""

    st.markdown(
        f'<div class="place-type-card">'
        f'<div class="place-type-icon">{place_icon}</div>'
        f'<div class="place-type-body">'
        f'<div class="place-type-name">{place_info.get("name", search_query)} {closed_badge}</div>'
        f'<div class="place-type-category">{category or "Place"}</div>'
        f'{desc_html}{subtypes_html}{meta_html}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Place meta expander ───────────────────────────────────────────────────
    with st.expander("📍 **Place Metadata**", expanded=True):
        pm1, pm2, pm3 = st.columns(3)
        with pm1:
            st.markdown(f"**Name:** {place_info.get('name', search_query)}")
            st.markdown(f"**Category:** {place_info.get('category') or 'N/A'}")
        with pm2:
            st.markdown(f"**Google Score:** ⭐ {place_info.get('google_score', 'N/A')} / 5")
            rc_cnt = place_info.get("review_count", "N/A")
            st.markdown(f"**Total Google Reviews:** {rc_cnt:,}" if isinstance(rc_cnt, (int, float)) else f"**Total Google Reviews:** {rc_cnt}")
        with pm3:
            st.markdown(f"**Address:** {place_info.get('address') or 'N/A'}")
            if place_info.get("website"):
                st.markdown(f"**Website:** [{place_info['website']}]({place_info['website']})")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_verdict, tab_sentiment, tab_guardrail, tab_tips, tab_verif, tab_raw = st.tabs([
        "📋 Executive Verdict",
        "🧠 Deep Sentiment",
        "🛡️ Guardrail & Authenticity",
        "💡 Visitor Guide",
        "🔍 Model B Verification",
        "📄 Raw Data & Export",
    ])

    # ══════════════════════════════════════════════════════
    # TAB 1 — Executive Verdict
    # ══════════════════════════════════════════════════════
    with tab_verdict:
        col_pro, col_con = st.columns(2)
        with col_pro:
            st.subheader("✅ Genuine Strengths")
            for p in rec.get("pros", []):
                pt = p.get("point") or p.get("text") or str(p) if isinstance(p, dict) else str(p)
                wt = p.get("weight", "") if isinstance(p, dict) else ""
                st.markdown(
                    f'<div class="pro-card">✓ <b>{pt}</b>'
                    + (f' <code>[{wt}]</code>' if wt else "") + "</div>",
                    unsafe_allow_html=True,
                )
            if not rec.get("pros"):
                st.info("No significant strengths highlighted.")

        with col_con:
            st.subheader("❌ Concerns & Drawbacks")
            for c in rec.get("cons", []):
                ct = c.get("point") or c.get("text") or str(c) if isinstance(c, dict) else str(c)
                wt = c.get("weight", "") if isinstance(c, dict) else ""
                st.markdown(
                    f'<div class="con-card">✗ <b>{ct}</b>'
                    + (f' <code>[{wt}]</code>' if wt else "") + "</div>",
                    unsafe_allow_html=True,
                )
            if not rec.get("cons"):
                st.info("No major concerns reported.")

        st.subheader("📝 Balanced Assessment")
        st.info(rec.get("full_verdict", "Analysis completed."))

        gs = guardrail.get("guardrail_summary", "")
        ar = guardrail.get("analyst_recommendation", "")
        if gs or ar:
            st.warning(f"🔍 **Analyst Note:** {' '.join(filter(None, [gs, ar]))}")

        # Score breakdown bar chart
        st.subheader("📊 Score Breakdown")
        bd  = rec.get("score_breakdown", {})
        asp = sentiment.get("aspect_scores", {})
        score_rows = []
        if bd:
            score_rows += [
                ("Sentiment",     bd.get("sentiment_score", 0)),
                ("Google Rating", bd.get("rating_score",    0)),
                ("Trust",         bd.get("trust_score",     0)),
                ("Final Score",   bd.get("composite",       0)),
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
            df_sc = pd.DataFrame(score_rows, columns=["Metric", "Score"])
            fig_sc = px.bar(
                df_sc, x="Metric", y="Score", text_auto=".1f",
                range_y=[0, 10], color="Score",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                color_continuous_midpoint=5,
            )
            fig_sc.update_layout(showlegend=False, coloraxis_showscale=False,
                                 height=320, margin=dict(l=10, r=10, t=10, b=40))
            st.plotly_chart(fig_sc, use_container_width=True)

        # Standout quotes
        sq_pos = sentiment.get("standout_positive_quote", "")
        sq_neg = sentiment.get("standout_negative_quote", "")
        if sq_pos or sq_neg:
            st.subheader("💬 Standout Reviewer Quotes")
            qc1, qc2 = st.columns(2)
            with qc1:
                if sq_pos:
                    st.markdown(f'<div class="pro-card">🌟 <i>"{sq_pos}"</i></div>', unsafe_allow_html=True)
            with qc2:
                if sq_neg:
                    st.markdown(f'<div class="con-card">⚠️ <i>"{sq_neg}"</i></div>', unsafe_allow_html=True)

        # Positive / Negative points with evidence IDs
        pp_col, np_col = st.columns(2)
        with pp_col:
            st.subheader("👍 Positive Points")
            for pp in sentiment.get("positive_points", []):
                if isinstance(pp, dict):
                    ids = pp.get("evidence_review_ids", [])
                    ids_str = f' <span style="color:#64748b;font-size:0.75rem;">({", ".join(ids)})</span>' if ids else ""
                    st.markdown(
                        f'<div class="pro-card">✓ {pp.get("claim","")}{ids_str}</div>',
                        unsafe_allow_html=True,
                    )
        with np_col:
            st.subheader("👎 Negative Points")
            for np_ in sentiment.get("negative_points", []):
                if isinstance(np_, dict):
                    ids = np_.get("evidence_review_ids", [])
                    ids_str = f' <span style="color:#64748b;font-size:0.75rem;">({", ".join(ids)})</span>' if ids else ""
                    st.markdown(
                        f'<div class="con-card">✗ {np_.get("claim","")}{ids_str}</div>',
                        unsafe_allow_html=True,
                    )

    # ══════════════════════════════════════════════════════
    # TAB 2 — Deep Sentiment
    # ══════════════════════════════════════════════════════
    with tab_sentiment:
        sc1, sc2 = st.columns(2)

        with sc1:
            st.subheader("🎭 Emotion Distribution")
            emo_dist = sentiment.get("emotion_distribution", {})
            if emo_dist and any(emo_dist.values()):
                df_emo = pd.DataFrame({"Emotion": list(emo_dist.keys()), "Count": list(emo_dist.values())})
                df_emo = df_emo[df_emo["Count"] > 0]
                fig_emo = px.pie(df_emo, names="Emotion", values="Count",
                                 color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
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
                    "Count": [counts[k]  for k in sorted(counts.keys(), reverse=True)],
                })
                fig_r = px.bar(df_r, x="Stars", y="Count", text_auto=True,
                               color="Count", color_continuous_scale=["#ef4444","#f59e0b","#22c55e"])
                fig_r.update_layout(showlegend=False, coloraxis_showscale=False,
                                    height=280, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_r, use_container_width=True)

        # Aspect radar + detail bars
        st.subheader("🔬 Aspect-Based Scores (0 – 10)")
        asp     = sentiment.get("aspect_scores", {})
        asp_map = {
            "food_quality":"Food", "service":"Service", "ambience":"Ambience",
            "value_for_money":"Value", "cleanliness":"Cleanliness",
            "accessibility":"Accessibility", "crowd_wait_time":"Crowd / Wait",
        }
        asp_rows = []
        for key, label in asp_map.items():
            v = asp.get(key, {})
            sc   = v.get("score")             if isinstance(v, dict) else None
            cnt  = v.get("reviews_mentioning", 0) if isinstance(v, dict) else 0
            summ = v.get("summary", "")        if isinstance(v, dict) else ""
            ids  = v.get("evidence_review_ids",[]) if isinstance(v, dict) else []
            if sc is not None:
                asp_rows.append((label, sc, cnt, summ or "—", ids))

        if asp_rows:
            labels_r = [r[0] for r in asp_rows]
            values_r = [r[1] for r in asp_rows]
            fig_radar = go.Figure(go.Scatterpolar(
                r=values_r + [values_r[0]], theta=labels_r + [labels_r[0]],
                fill="toself", fillcolor="rgba(56,189,248,0.2)",
                line=dict(color="#38bdf8", width=2),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0,10], tickfont=dict(size=10))),
                showlegend=False, height=320, margin=dict(l=30,r=30,t=20,b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            for label, sc, cnt, summ, ids in asp_rows:
                color = "#22c55e" if sc >= 7 else "#f59e0b" if sc >= 5 else "#ef4444"
                bar_w = int(sc * 10)
                ids_str = f'<span style="color:#64748b;font-size:0.72rem;"> · {", ".join(ids)}</span>' if ids else ""
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'<span style="width:110px;font-size:0.85rem;color:#94a3b8;">{label}</span>'
                    f'<div style="flex:1;background:#1e293b;border-radius:4px;height:14px;">'
                    f'<div style="width:{bar_w}%;background:{color};height:14px;border-radius:4px;"></div></div>'
                    f'<span style="width:32px;text-align:right;font-weight:700;color:{color};font-size:0.9rem;">{sc:.1f}</span>'
                    f'<span style="font-size:0.75rem;color:#64748b;width:80px;">({cnt} reviews)</span>'
                    f'<span style="font-size:0.8rem;color:#cbd5e1;flex:2;">{summ}{ids_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Aspect scores not available.")

        # Temporal trend
        st.subheader("📈 Temporal Sentiment Trend")
        tt = sentiment.get("temporal_trend", {})
        if tt:
            tc1, tc2, tc3 = st.columns(3)
            with tc1: st.metric("Recent Sentiment", tt.get("recent_sentiment","—"), delta=f'{tt.get("recent_score",0):.2f}')
            with tc2: st.metric("Older Sentiment",  tt.get("older_sentiment","—"),  delta=f'{tt.get("older_score",0):.2f}')
            with tc3:
                trend_icon = {"Improving":"📈","Declining":"📉","Stable":"➡️"}.get(tt.get("trend",""),"—")
                st.metric("Overall Trend", f'{trend_icon} {tt.get("trend","—")}')
            if tt.get("trend_explanation"):
                st.caption(f"💡 {tt['trend_explanation']}")

        # Crowd profile
        cp = sentiment.get("crowd_profile", {})
        if cp:
            st.subheader("👥 Crowd Profile")
            st.markdown(f"**Dominant visitor type:** {cp.get('dominant_visitor_type','—')}")
            if cp.get("mention_evidence"):
                st.caption(f"Evidence: {cp['mention_evidence']}")
            if cp.get("accessibility_notes"):
                st.info(f"♿ Accessibility: {cp['accessibility_notes']}")

        # Themes
        themes = sentiment.get("themes", [])
        if themes:
            st.subheader("🎯 Key Themes")
            for t in themes:
                if not isinstance(t, dict): continue
                t_sent  = t.get("sentiment","Neutral")
                t_color = "#22c55e" if "Pos" in t_sent else "#ef4444" if "Neg" in t_sent else "#f59e0b"
                t_ids   = t.get("evidence_review_ids",[])
                t_ids_str = f' <span style="color:#64748b;font-size:0.72rem;">({", ".join(t_ids)})</span>' if t_ids else ""
                st.markdown(
                    f'<div style="border-left:4px solid {t_color};padding:0.6rem 1rem;'
                    f'background:#1e293b;border-radius:0 6px 6px 0;margin-bottom:8px;">'
                    f'<b style="color:{t_color};">{t.get("name","")}</b>'
                    f'<span style="font-size:0.75rem;color:#64748b;margin-left:8px;">'
                    f'{t_sent} · {t.get("frequency",0)} mentions{t_ids_str}</span><br>'
                    f'<span style="color:#cbd5e1;font-size:0.85rem;">{t.get("evidence","")}</span>'
                    + (f'<br><i style="color:#94a3b8;font-size:0.8rem;">"{t["representative_quote"]}"</i>' if t.get("representative_quote") else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

        # Keywords
        kc1, kc2 = st.columns(2)
        with kc1:
            st.subheader("👍 Positive Keywords")
            pos_kws = sentiment.get("positive_keywords", [])
            if pos_kws:
                st.markdown("".join(f'<span class="tag-pill tag-pos">👍 {kw}</span>' for kw in pos_kws), unsafe_allow_html=True)
        with kc2:
            st.subheader("👎 Negative Keywords")
            neg_kws = sentiment.get("negative_keywords", [])
            if neg_kws:
                st.markdown("".join(f'<span class="tag-pill tag-neg">👎 {kw}</span>' for kw in neg_kws), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # TAB 3 — Guardrail & Authenticity
    # ══════════════════════════════════════════════════════
    with tab_guardrail:
        gm1, gm2, gm3, gm4, gm5 = st.columns(5)
        with gm1: st.metric("Trust Score",    f"{guardrail.get('trust_score',0):.0%}")
        with gm2: st.metric("Fake Risk",       f"{guardrail.get('fake_review_probability',0):.0%}")
        with gm3: st.metric("Review Quality",  guardrail.get("review_quality","—"))
        with gm4: st.metric("Bias Level",      guardrail.get("bias_level","—"))
        with gm5:
            adj = guardrail.get("rating_integrity",{}).get("adjusted_true_rating","—")
            st.metric("Adj. True Rating", f"{adj}/5" if adj != "—" else "—")

        st.divider()

        # Review statistics — from review_stats (real counts, not heuristic)
        hs = guardrail.get("heuristic_stats", {})
        st.subheader("🔢 Review Statistics")
        hc1, hc2, hc3, hc4, hc5 = st.columns(5)
        with hc1: st.metric("Raw Scraped",     review_stats.get("raw_scraped_count",        hs.get("total_reviews","—")))
        with hc2: st.metric("Usable Text",     review_stats.get("usable_text_review_count", "—"))
        with hc3: st.metric("Cleaned",         review_stats.get("cleaned_review_count",     "—"))
        with hc4: st.metric("Analyzed",        review_stats.get("analyzed_review_count",    hs.get("total_reviews","—")))
        with hc5: st.metric("Duplicate Pairs", hs.get("duplicate_pairs", 0))

        la_col, ri_col = st.columns(2)
        with la_col:
            st.subheader("🔤 Linguistic Analysis")
            la = guardrail.get("linguistic_analysis", {})
            if la:
                for label, val in [
                    ("Templated language",  "✅ Detected" if la.get("templated_language_detected") else "✗ Not detected"),
                    ("Vocabulary diversity", la.get("vocabulary_diversity","—")),
                    ("Writing style",        la.get("writing_style_consistency","—")),
                ]:
                    col_c = "#ef4444" if "Detected" in str(val) or "Consistent" in str(val) else "#22c55e"
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e293b;">'
                        f'<span style="color:#94a3b8;font-size:0.85rem;">{label}</span>'
                        f'<span style="color:{col_c};font-size:0.85rem;font-weight:600;">{val}</span></div>',
                        unsafe_allow_html=True,
                    )
                if la.get("copy_paste_evidence"):
                    st.warning(f"📋 {la['copy_paste_evidence']}")
                if la.get("language_notes"):
                    st.caption(la["language_notes"])

        with ri_col:
            st.subheader("⭐ Rating Integrity")
            ri = guardrail.get("rating_integrity", {})
            if ri:
                natural = ri.get("distribution_natural", True)
                st.markdown(
                    f'<div style="color:{"#22c55e" if natural else "#ef4444"};font-weight:600;margin-bottom:8px;">'
                    f'{"✅ Distribution appears natural" if natural else "⚠️ Unnatural distribution"}</div>',
                    unsafe_allow_html=True,
                )
                if ri.get("inflated_stars_estimate"):
                    st.markdown(f"**~{ri['inflated_stars_estimate']} inflated** 5-star reviews estimated")
                if ri.get("suppressed_stars_estimate"):
                    st.markdown(f"**~{ri['suppressed_stars_estimate']} suppressed** low-star reviews")
                for anomaly in ri.get("anomalies", []):
                    st.warning(f"⚠️ {anomaly}")
                if ri.get("rating_notes"):
                    st.caption(ri["rating_notes"])

        # Reviewer behaviour
        st.subheader("👤 Reviewer Behaviour")
        rb = guardrail.get("reviewer_behavior", {})
        if rb:
            rbc1, rbc2 = st.columns(2)
            with rbc1:
                risk  = rb.get("sock_puppet_risk","—")
                color = {"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"}.get(risk,"#94a3b8")
                st.markdown(f'<b>Sock-puppet risk:</b> <span style="color:{color};font-weight:700;">{risk}</span>', unsafe_allow_html=True)
            with rbc2:
                risk2  = rb.get("coordinated_posting_risk","—")
                color2 = {"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"}.get(risk2,"#94a3b8")
                st.markdown(f'<b>Coordinated posting:</b> <span style="color:{color2};font-weight:700;">{risk2}</span>', unsafe_allow_html=True)
            if rb.get("evidence"):
                st.caption(f"Evidence: {rb['evidence']}")

        st.divider()

        # Contradictions
        contradictions = guardrail.get("contradiction_analysis", [])
        if contradictions:
            st.subheader("⚖️ Contradictions in Reviews")
            for c in contradictions:
                if not isinstance(c, dict): continue
                with st.expander(f"⚖️ **{c.get('aspect','Aspect')}**", expanded=False):
                    cc1, cc2 = st.columns(2)
                    with cc1: st.markdown(f'<div class="pro-card">👍 {c.get("positive_claim","—")}</div>', unsafe_allow_html=True)
                    with cc2: st.markdown(f'<div class="con-card">👎 {c.get("negative_claim","—")}</div>', unsafe_allow_html=True)
                    st.info(f"🔍 Resolution: {c.get('resolution','—')}")

        # Per-aspect credibility
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
                        "Note":       v.get("note","—"),
                    })
            if ac_rows:
                st.dataframe(pd.DataFrame(ac_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Suspicious patterns
        patterns = guardrail.get("suspicious_patterns", [])
        st.subheader("🚨 Suspicious Patterns")
        if patterns:
            for pat in patterns:
                st.warning(f"⚠️ {pat}")
        else:
            st.success("✅ No suspicious review manipulation patterns detected.")

        # Verified positives & concerns
        vp_col, vc_col = st.columns(2)
        with vp_col:
            st.subheader("✅ Verified Genuine Positives")
            for p in guardrail.get("genuine_positives", []):
                if not isinstance(p, dict): continue
                ids = p.get("supporting_review_ids",[])
                st.markdown(
                    f'<div class="pro-card"><b>{p.get("aspect","—")}</b>'
                    f' <span style="opacity:0.7;font-size:0.8rem;">conf {p.get("confidence",0):.0%}'
                    + (f' · {ids}' if ids else "") + f'</span><br>{p.get("evidence","")}</div>',
                    unsafe_allow_html=True,
                )
        with vc_col:
            st.subheader("⚠️ Verified Genuine Concerns")
            for c in guardrail.get("genuine_concerns", []):
                if not isinstance(c, dict): continue
                sev   = c.get("severity","Minor")
                s_col = {"Major":"#ef4444","Moderate":"#f59e0b","Minor":"#94a3b8"}.get(sev,"#94a3b8")
                ids   = c.get("supporting_review_ids",[])
                st.markdown(
                    f'<div class="con-card"><b>{c.get("aspect","—")}</b>'
                    f' <span style="color:{s_col};font-size:0.8rem;font-weight:700;">[{sev}]'
                    + (f' · {ids}' if ids else "") + f'</span><br>{c.get("evidence","")}</div>',
                    unsafe_allow_html=True,
                )

        vf = guardrail.get("verified_facts", [])
        if vf:
            st.subheader("📌 Verified Facts")
            for f in vf:
                st.markdown(f"📌 {f}")

    # ══════════════════════════════════════════════════════
    # TAB 4 — Visitor Guide
    # ══════════════════════════════════════════════════════
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
        st.info(f"🕒 {rec.get('best_time') or 'Not determinable from available reviews.'}")

        st.subheader("💡 Actionable Visitor Tips")
        for tip in rec.get("visitor_tips", []):
            st.markdown(
                f'<div style="background:#1e293b;border-left:4px solid #38bdf8;'
                f'padding:0.6rem 1rem;border-radius:0 6px 6px 0;margin-bottom:6px;color:#e2e8f0;">'
                f'💡 {tip}</div>',
                unsafe_allow_html=True,
            )

        cp = sentiment.get("crowd_profile", {})
        if cp:
            st.divider()
            st.subheader("👥 Visitor Profile")
            st.markdown(f"**Typical visitor:** {cp.get('dominant_visitor_type','—')}")
            if cp.get("accessibility_notes"):
                st.info(f"♿ {cp['accessibility_notes']}")

    # ══════════════════════════════════════════════════════
    # TAB 5 — Model B Verification
    # ══════════════════════════════════════════════════════
    with tab_verif:
        vf_status = verification.get("status", "UNAVAILABLE")

        if vf_status == "UNAVAILABLE":
            st.warning(
                f"⚠️ Independent verification was not available for this analysis.\n\n"
                f"{verification.get('verification_notes','Set GROQ_API_KEY in .env to enable Model B.')}"
            )
        else:
            vc1, vc2, vc3, vc4 = st.columns(4)
            v_acc = verification.get("accuracy")
            with vc1: st.metric("Accuracy",   f"{v_acc:.0%}" if v_acc is not None else "N/A")
            with vc2: st.metric("Status",     vf_status)
            with vc3:
                hall = verification.get("hallucination_detected")
                st.metric("Hallucination", ("Yes 🚨" if hall else "No ✅") if hall is not None else "N/A")
            with vc4: st.metric("Corrections Applied", verification.get("corrections_count", 0))

            st.divider()
            st.subheader("📋 Field-by-Field Verification")

            field_checks = [
                ("Sentiment", verification.get("sentiment", {})),
                ("Aspects",   verification.get("aspects",   {})),
                ("Keywords",  verification.get("keywords",  {})),
                ("Guardrail", verification.get("guardrail", {})),
                ("Evidence",  verification.get("evidence",  {})),
            ]
            for field_name, fdata in field_checks:
                if not isinstance(fdata, dict): continue
                fstatus = fdata.get("status","UNAVAILABLE")
                issues  = fdata.get("issues",[]) or fdata.get("unsupported_claims",[])
                badge   = {"PASS":"✅","CORRECTED":"🔧","FAIL":"❌"}.get(fstatus,"⚠️")
                with st.expander(
                    f"{badge} **{field_name}** — {fstatus}"
                    + (f" ({len(issues)} issue{'s' if len(issues)!=1 else ''})" if issues else ""),
                    expanded=(fstatus not in ("PASS","UNAVAILABLE")),
                ):
                    if issues:
                        for issue in issues:
                            st.markdown(f'<div class="con-card">⚠️ {issue}</div>', unsafe_allow_html=True)
                    else:
                        st.success("No issues found.")

            # Corrections detail
            corrections = verification.get("corrections", [])
            if corrections:
                st.divider()
                st.subheader("🔧 Corrections Detail")
                for i, corr in enumerate(corrections):
                    has_ev = bool(corr.get("evidence_review_ids"))
                    badge  = "✅ Applied" if has_ev else "⏭️ Skipped (no evidence)"
                    with st.expander(f"**{i+1}. {corr.get('field','Field')}** — {badge}", expanded=has_ev):
                        ca, cb = st.columns(2)
                        with ca:
                            st.markdown("**Original (Model A):**")
                            st.markdown(f'<div class="con-card">{corr.get("original_claim","—")}</div>', unsafe_allow_html=True)
                        with cb:
                            st.markdown("**Corrected (Model B):**")
                            st.markdown(f'<div class="pro-card">{corr.get("corrected_claim","—")}</div>', unsafe_allow_html=True)
                        st.caption(f"**Reason:** {corr.get('reason','—')}")
                        ids = corr.get("evidence_review_ids", [])
                        if ids:
                            st.caption(f"**Evidence review IDs:** {', '.join(str(x) for x in ids)}")
                        else:
                            st.warning("No review IDs provided — correction was skipped.")
            else:
                st.success("✅ No corrections necessary — Model A analysis verified clean.")

            # Missing evidence
            miss_pos = verification.get("missing_positive_evidence", [])
            miss_neg = verification.get("missing_negative_evidence", [])
            if miss_pos or miss_neg:
                st.divider()
                st.subheader("🔎 Missing Evidence Flagged by Model B")
                mp_col, mn_col = st.columns(2)
                with mp_col:
                    st.markdown("**Missed Positive Signals**")
                    for item in miss_pos:
                        st.markdown(f'<div class="pro-card">+ {item}</div>', unsafe_allow_html=True)
                with mn_col:
                    st.markdown("**Missed Negative Signals**")
                    for item in miss_neg:
                        st.markdown(f'<div class="con-card">- {item}</div>', unsafe_allow_html=True)

            if verification.get("verification_notes"):
                st.divider()
                st.info(f"📝 **Model B Notes:** {verification['verification_notes']}")

    # ══════════════════════════════════════════════════════
    # TAB 6 — Raw Data & Export
    # ══════════════════════════════════════════════════════
    with tab_raw:
        st.subheader("📊 Review Statistics")
        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("Raw Scraped",    review_stats.get("raw_scraped_count",        len(reviews)))
        with s2: st.metric("Usable Reviews", review_stats.get("usable_text_review_count", len(reviews)))
        with s3: st.metric("After Cleaning", review_stats.get("cleaned_review_count",     len(reviews)))
        with s4: st.metric("Analyzed",       review_stats.get("analyzed_review_count",    len(reviews)))

        st.divider()
        st.subheader("📄 Scraped Reviews")
        if reviews:
            df_revs = pd.DataFrame(reviews)
            st.dataframe(df_revs, use_container_width=True, height=350)

        st.divider()
        st.subheader("🔍 Verification JSON")
        st.json(verification)

        st.divider()
        st.subheader("💾 Export Full Report")
        json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        loc_slug = "".join(
            c if c.isalnum() or c in " _-" else ""
            for c in search_query
        )[:30].strip().replace(" ", "_")
        st.download_button(
            label="📥 Download Full Report (.json)",
            data=json_str,
            file_name=f"report_{loc_slug}.json",
            mime="application/json",
            type="primary",
        )
