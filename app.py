"""app.py – Streamlit UI for AI Hospital Supply-Chain Disruption Analyst.

Tabs:
  1. Dashboard  – risk table for all items at selected facility
  2. Analyse Email – paste email → extract → forecast → agent recommendation
  3. Agent Trace   – tool-call steps from the last analysis
  4. Guidelines    – RAG search over SOPs
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ── Page config ───────────────────────────────────────────────────────────────
_CFG = yaml.safe_load(Path("config.yaml").read_text())
st.set_page_config(
    page_title=_CFG["app"]["title"],
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.risk-high {
    background: linear-gradient(135deg, #ff4757 0%, #c0392b 100%);
    color: white; padding: 4px 10px; border-radius: 6px;
    font-weight: 600; font-size: 0.8rem; display: inline-block;
}
.risk-medium {
    background: linear-gradient(135deg, #ffa502 0%, #e67e22 100%);
    color: white; padding: 4px 10px; border-radius: 6px;
    font-weight: 600; font-size: 0.8rem; display: inline-block;
}
.risk-low {
    background: linear-gradient(135deg, #2ed573 0%, #27ae60 100%);
    color: white; padding: 4px 10px; border-radius: 6px;
    font-weight: 600; font-size: 0.8rem; display: inline-block;
}
.risk-critical {
    background: linear-gradient(135deg, #8e44ad 0%, #6c3483 100%);
    color: white; padding: 4px 10px; border-radius: 6px;
    font-weight: 600; font-size: 0.8rem; display: inline-block;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460; border-radius: 12px;
    padding: 16px 20px; text-align: center;
}

.trace-thought { border-left: 3px solid #3498db; padding-left: 10px; margin: 6px 0; }
.trace-tool { border-left: 3px solid #9b59b6; padding-left: 10px; margin: 6px 0; }
.trace-obs { border-left: 3px solid #27ae60; padding-left: 10px; margin: 6px 0; }
.trace-answer { border-left: 3px solid #f39c12; padding-left: 10px; margin: 6px 0; }

.citation-card {
    background: #f8f9fa; border-left: 4px solid #3498db;
    padding: 10px 14px; border-radius: 4px; margin: 6px 0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_badge(level: str) -> str:
    cls = {
        "HIGH": "risk-high", "MEDIUM": "risk-medium",
        "LOW": "risk-low", "CRITICAL": "risk-critical",
        "WARNING": "risk-medium", "OK": "risk-low", "UNKNOWN": "risk-low",
    }.get(level, "risk-low")
    return f'<span class="{cls}">{level}</span>'


@st.cache_data(ttl=30)
def _get_facilities() -> list[dict]:
    from core.db import get_all_facilities
    return get_all_facilities()


@st.cache_data(ttl=30)
def _get_items() -> list[dict]:
    from core.db import get_all_items
    return get_all_items()


def _build_dashboard_df(facility_id: str) -> pd.DataFrame:
    from core.db import get_all_stock, get_consumption, find_surplus_facilities
    from core.forecast import forecast_days_of_stock
    from core.risk import assess_stock_risk, assess_expiry_risk

    stock_rows = get_all_stock(facility_id)
    # Group by item
    items_seen: dict[str, dict] = {}
    for row in stock_rows:
        iid = row["item_id"]
        if iid not in items_seen:
            items_seen[iid] = {"name": row["item_name"], "unit": row["unit"],
                               "total_qty": 0.0, "rows": []}
        items_seen[iid]["total_qty"] += row["qty"]
        items_seen[iid]["rows"].append(row)

    records = []
    for iid, info in items_seen.items():
        consumption = get_consumption(facility_id, iid, days=30)
        fc = forecast_days_of_stock(
            consumption_records=consumption,
            current_stock=info["total_qty"],
            item_id=iid,
            facility_id=facility_id,
            unit=info["unit"],
        )
        stock_risk = assess_stock_risk(fc.days_of_stock, new_eta_days=0)
        expiry_risk, nearest_expiry, days_until_expiry = assess_expiry_risk(info["rows"])
        surplus = find_surplus_facilities(iid, facility_id)

        records.append({
            "Item": info["name"],
            "On Hand": f"{info['total_qty']:.0f} {info['unit']}",
            "Daily Use": f"{fc.daily_consumption:.1f}",
            "Days Left": (
                "∞" if fc.days_of_stock == float("inf")
                else f"{fc.days_of_stock:.1f}"
            ),
            "Stock Risk": stock_risk,
            "Expiry Risk": expiry_risk,
            "Nearest Expiry": nearest_expiry or "N/A",
            "Days to Expiry": days_until_expiry if days_until_expiry is not None else "N/A",
            "Surplus At": ", ".join(f["name"] for f in surplus) if surplus else "None",
            "_item_id": iid,
        })

    return pd.DataFrame(records)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=64)
    st.title("⚕️ Supply-Chain Analyst")
    st.caption(f"v{_CFG['app']['version']} · Open-Source · Local AI")
    st.divider()

    # Model info
    st.subheader("🤖 Model")
    model_name = st.text_input(
        "Ollama model", value=_CFG["ollama"]["chat_model"],
        help="Change and re-run to switch models without touching code."
    )
    # Update in-memory config for this session
    _CFG["ollama"]["chat_model"] = model_name

    st.divider()

    # Facility selector
    st.subheader("🏥 Facility")
    try:
        facilities = _get_facilities()
        facility_map = {f["name"]: f["id"] for f in facilities}
        selected_facility_name = st.selectbox(
            "Select facility",
            list(facility_map.keys()),
            index=0,
        )
        selected_facility_id = facility_map[selected_facility_name]
    except Exception:
        st.error("Database not seeded. Run: `python seed.py`")
        st.stop()

    st.divider()
    st.caption(
        "⚠️ **Synthetic data only.** "
        "All data in this system is fictional for demonstration purposes. "
        "AI recommendations require human review before action."
    )

    # Ollama status
    st.divider()
    st.subheader("⚡ Ollama Status")
    try:
        import httpx
        r = httpx.get(f"{_CFG['ollama']['base_url']}/api/version", timeout=3)
        r.raise_for_status()
        v = r.json().get("version", "?")
        st.success(f"Running v{v}")
    except Exception:
        st.error("Not running. Start: `ollama serve`")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "📧 Analyse Email", "🔍 Agent Trace", "📚 Guidelines"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 – Dashboard
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header(f"📊 Risk Dashboard — {selected_facility_name}")
    st.caption("Stock-out risk, expiry risk, and transfer recommendations for all items.")

    with st.spinner("Loading stock data …"):
        df = _build_dashboard_df(selected_facility_id)

    if df.empty:
        st.warning("No stock data found. Run `python seed.py` first.")
    else:
        # Summary metrics
        n_high = (df["Stock Risk"] == "HIGH").sum()
        n_exp_crit = (df["Expiry Risk"] == "CRITICAL").sum()
        n_exp_warn = (df["Expiry Risk"] == "WARNING").sum()
        n_total = len(df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", n_total)
        with col2:
            st.metric("🔴 HIGH Stock Risk", n_high, delta=None,
                      delta_color="inverse" if n_high > 0 else "normal")
        with col3:
            st.metric("🟣 Critical Expiry", n_exp_crit)
        with col4:
            st.metric("🟡 Expiry Warning", n_exp_warn)

        st.divider()

        # Color-coded table
        def _color_stock(val: str) -> str:
            colors = {
                "HIGH": "background-color: #ffe0e0; color: #c0392b; font-weight: bold",
                "MEDIUM": "background-color: #fff4cc; color: #e67e22; font-weight: bold",
                "LOW": "background-color: #e8f8e8; color: #27ae60",
            }
            return colors.get(val, "")

        def _color_expiry(val: str) -> str:
            colors = {
                "CRITICAL": "background-color: #f3e1fa; color: #8e44ad; font-weight: bold",
                "WARNING": "background-color: #fff4cc; color: #e67e22",
                "OK": "background-color: #e8f8e8; color: #27ae60",
            }
            return colors.get(val, "")

        display_df = df.drop(columns=["_item_id"])
        styled = (
            display_df.style
            .map(_color_stock, subset=["Stock Risk"])
            .map(_color_expiry, subset=["Expiry Risk"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()

        # Dependency map
        st.subheader("🔗 Item → Department Dependency Map")
        from core.db import get_affected_departments, get_all_items
        items_list = get_all_items()
        dep_records = []
        for item in items_list:
            depts = get_affected_departments(item["id"])
            if depts:
                dep_records.append({
                    "Item": item["name"],
                    "Affected Departments": " · ".join(d["name"] for d in depts),
                })
        if dep_records:
            st.dataframe(
                pd.DataFrame(dep_records),
                use_container_width=True,
                hide_index=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 – Analyse Email
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("📧 Analyse Supplier Email / PO")
    st.caption("Paste a supplier email or purchase order (English or Hindi). The AI will extract disruption info, forecast stock impact, and generate a recommendation.")

    # Sample email loader
    samples = {
        "— choose a sample —": "",
        "🚚 Insulin delay (English)":
            Path("data/sample_emails/insulin_delay_en.txt").read_text(encoding="utf-8"),
        "🔥 Paracetamol delay (Hindi)":
            Path("data/sample_emails/paracetamol_delay_hi.txt").read_text(encoding="utf-8"),
        "⚠️ Paracetamol expiry risk":
            Path("data/sample_emails/expiry_risk_paracetamol.txt").read_text(encoding="utf-8"),
    }
    chosen = st.selectbox("Load sample email", list(samples.keys()))
    email_text = st.text_area(
        "Email / PO text",
        value=samples[chosen],
        height=220,
        placeholder="Paste supplier email or PO here (English or Hindi) …",
        key="email_input",
    )

    analyse_btn = st.button("🔍 Analyse", type="primary", disabled=not email_text.strip())

    if analyse_btn and email_text.strip():
        # ── Step 1: Extract ────────────────────────────────────────────────
        with st.status("Step 1/4 – Extracting disruption info …", expanded=True) as status:
            try:
                from core.extract import extract_disruption
                extraction = extract_disruption(email_text)
                st.write("✅ Extraction complete")
                status.update(label="Step 1/4 – Extracted ✓", state="complete")
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                st.stop()

        # Display extraction
        st.subheader("📋 Extracted Disruption")
        ecols = st.columns(3)
        with ecols[0]:
            st.metric("Item", extraction.item)
            st.metric("Language", extraction.language_detected.upper())
        with ecols[1]:
            qty_val = f"{extraction.qty:.0f}" if extraction.qty else "Not specified"
            st.metric("Quantity Affected", qty_val)
            st.metric("ETA Delay", f"{extraction.new_eta_days} days")
        with ecols[2]:
            st.metric("Confidence", f"{extraction.confidence:.0%}")
        st.info(f"**Reason:** {extraction.reason}")

        col_json, _ = st.columns([1, 2])
        with col_json:
            with st.expander("Raw JSON"):
                st.json(extraction.model_dump())

        st.divider()

        # ── Step 2: Forecast ───────────────────────────────────────────────
        with st.status("Step 2/4 – Forecasting stock …", expanded=True) as status:
            from core.db import get_item_by_name, get_stock, get_consumption
            from core.forecast import forecast_days_of_stock
            item_rec = get_item_by_name(extraction.item)
            if not item_rec:
                st.warning(
                    f"Item '{extraction.item}' not in database. "
                    "Showing extraction only."
                )
                status.update(label="Step 2/4 – Item not in DB", state="error")
            else:
                stock_rows = get_stock(selected_facility_id, item_rec["id"])
                total_stock = sum(r["qty"] for r in stock_rows)
                consumption = get_consumption(selected_facility_id, item_rec["id"], days=30)
                fc = forecast_days_of_stock(
                    consumption_records=consumption,
                    current_stock=total_stock,
                    item_id=item_rec["id"],
                    facility_id=selected_facility_id,
                    unit=item_rec.get("unit", "units"),
                )
                status.update(label="Step 2/4 – Forecast ✓", state="complete")

        if item_rec:
            st.subheader("📈 Forecast")
            fc_cols = st.columns(4)
            with fc_cols[0]:
                st.metric("Current Stock",
                          f"{fc.current_stock:.0f} {fc.unit}")
            with fc_cols[1]:
                st.metric("Daily Consumption",
                          f"{fc.daily_consumption:.1f} {fc.unit}/day")
            with fc_cols[2]:
                days_str = ("∞" if fc.days_of_stock == float("inf")
                            else f"{fc.days_of_stock:.1f}")
                st.metric("Days of Stock Left", days_str)
            with fc_cols[3]:
                st.metric("Method", fc.method.replace("_", " ").title())

            # Risk colour
            from core.risk import assess_stock_risk, assess_expiry_risk, build_risk_assessment
            from core.db import find_surplus_facilities
            stock_risk = assess_stock_risk(fc.days_of_stock, extraction.new_eta_days)
            expiry_risk, nearest_expiry, days_until_expiry = assess_expiry_risk(stock_rows)
            surplus = find_surplus_facilities(item_rec["id"], selected_facility_id)

            risk_col = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(stock_risk, "⚪")
            st.markdown(
                f"**Stock-out Risk:** {risk_col} **{stock_risk}** "
                f"(stock runs out in ~{days_str} days; replacement arrives in {extraction.new_eta_days} days)"
            )

            if surplus:
                st.success(
                    f"**Surplus available at:** "
                    + ", ".join(f"{s['name']} ({s['total_qty']:.0f} {fc.unit})" for s in surplus)
                )

            st.divider()

            # ── Step 3: Affected departments ───────────────────────────────
            with st.status("Step 3/4 – Affected departments …") as status:
                from core.db import get_affected_departments
                depts = get_affected_departments(item_rec["id"])
                status.update(label="Step 3/4 – Departments ✓", state="complete")

            st.subheader("🏥 Affected Departments")
            if depts:
                dept_cols = st.columns(min(len(depts), 4))
                for i, d in enumerate(depts):
                    with dept_cols[i % 4]:
                        st.markdown(f"🏥 **{d['name']}**")
            else:
                st.info("No department dependencies found for this item.")

            st.divider()

            # ── Step 4: Agent recommendation ───────────────────────────────
            st.subheader("🤖 AI Agent Recommendation")
            st.caption(
                "The agent uses tool-calling to look up live stock, run forecasts, "
                "check surplus at other facilities, and search guidelines — then writes "
                "a recommendation grounded in real data. **Always subject to human review.**"
            )

            with st.spinner("Agent analysing … (this may take 30–60 seconds) …"):
                try:
                    from core.agent import run_agent
                    result = run_agent(extraction, selected_facility_id)
                    # Store trace in session state for Tab 3
                    st.session_state["last_trace"] = result.trace
                    st.session_state["last_citations"] = result.citations
                    st.session_state["last_item"] = extraction.item
                except Exception as exc:
                    st.error(f"Agent error: {exc}")
                    result = None

            if result:
                st.markdown(
                    f"""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-left: 4px solid #3498db; border-radius: 8px;
            padding: 16px 20px; color: #ecf0f1; margin: 8px 0;">
{result.recommendation.replace(chr(10), "<br>")}
</div>
""",
                    unsafe_allow_html=True,
                )

                if result.citations:
                    st.subheader("📖 Guideline Citations")
                    for cit in result.citations[:3]:
                        with st.expander(
                            f"📄 {cit['source']} (relevance: {cit['score']:.2f})"
                        ):
                            st.markdown(cit["text"])

                st.info(
                    "👉 View full tool-call trace in the **Agent Trace** tab."
                )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 – Agent Trace
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("🔍 Agent Trace")
    st.caption("Full tool-call reasoning trace from the last analysis. Provides transparency into how the AI reached its recommendation.")

    trace = st.session_state.get("last_trace", [])
    if not trace:
        st.info("Run an analysis in the **Analyse Email** tab to see the agent trace here.")
    else:
        item_label = st.session_state.get("last_item", "?")
        st.markdown(f"**Last analysed item:** `{item_label}`")
        st.divider()

        for i, step in enumerate(trace, 1):
            icon = {"thought": "💭", "tool_call": "🔧", "observation": "📥", "answer": "✅"}.get(step.kind, "•")
            kind_label = step.kind.replace("_", " ").title()

            css_class = {
                "thought": "trace-thought",
                "tool_call": "trace-tool",
                "observation": "trace-obs",
                "answer": "trace-answer",
            }.get(step.kind, "trace-thought")

            with st.expander(f"Step {i} — {icon} {kind_label}", expanded=(step.kind == "answer")):
                st.markdown(
                    f'<div class="{css_class}"><pre style="white-space:pre-wrap;font-size:0.85rem;">'
                    f'{step.content}</pre></div>',
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 – Guidelines
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("📚 Guidelines & SOPs")
    st.caption("Search fictional SOPs, essential medicines list, and transfer policies using semantic search.")

    query = st.text_input(
        "Search guidelines",
        placeholder="e.g. minimum buffer stock insulin, inter-facility transfer approval …",
        key="guideline_query",
    )
    n_results = st.slider("Max results", 1, 6, 3)

    search_btn = st.button("🔍 Search", type="primary", key="search_guidelines")

    if search_btn and query.strip():
        with st.spinner("Searching …"):
            try:
                from core.rag import retrieve_guidelines
                results = retrieve_guidelines(query.strip(), n=n_results)
            except Exception as exc:
                st.error(f"RAG error: {exc}. Ensure Ollama is running and index is built.")
                results = []

        if results:
            for r in results:
                with st.expander(
                    f"📄 {r['source']} — relevance: {r['score']:.2f}", expanded=True
                ):
                    st.markdown(r["text"])
        else:
            st.warning(
                "No results. Ensure the RAG index is built: run `python seed.py`"
            )

    st.divider()
    st.subheader("📁 Available Guideline Documents")
    from pathlib import Path as _Path
    guides = sorted(_Path("data/guidelines").glob("*.md"))
    for g in guides:
        with st.expander(f"📄 {g.name}"):
            st.markdown(g.read_text(encoding="utf-8"))
