"""
Streamlit UI for the Tariff Classification Agent.

Run: streamlit run src/ui/app.py
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

# Suppress torchvision optional-dependency warning from sentence_transformers
warnings.filterwarnings("ignore", message=".*torchvision.*")

# Fix libffi
os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.agent.orchestrator import TariffAgent, ClassificationResult
from src.data.sync import TariffSync, get_sync_status


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tariff Classification Agent",
    page_icon="https://img.icons8.com/color/48/customs-officer.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    .stDeployButton, [data-testid="stStatusWidget"] { display: none; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    /* Typography */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        opacity: 0.6;
        margin-bottom: 1.5rem;
    }
    /* Result display */
    .confidence-high { color: #2ecc71; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #e74c3c; font-weight: bold; }
    .hts-code {
        font-size: 2rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-family: monospace;
        border: 1px solid rgba(128,128,128,0.3);
    }
    .expert-alert {
        border-left: 4px solid #6c8ebf;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        background: rgba(108,142,191,0.15);
        color: inherit;
    }
    .tariff-warning {
        border-left: 4px solid #e74c3c;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        line-height: 1.4;
        background: rgba(231,76,60,0.12);
        color: inherit;
    }
    .tariff-warning strong {
        color: #e74c3c;
    }
    .ruling-card {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    div[data-testid="stTextInput"] input {
        font-size: 1.1rem;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        width: 260px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    .sidebar-brand {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .sidebar-tagline {
        font-size: 0.8rem;
        opacity: 0.6;
        margin-bottom: 1.5rem;
    }
    .sidebar-stat {
        font-size: 0.85rem;
        padding: 0.3rem 0;
    }
    .sidebar-stat strong {
        display: block;
        font-size: 1.1rem;
    }
    /* Quick-try chips */
    .example-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 1rem;
    }
    /* Reasoning steps */
    .reasoning-step {
        display: flex;
        gap: 0.8rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.15);
    }
    .reasoning-step:last-child { border-bottom: none; }
    .step-number {
        flex-shrink: 0;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: rgba(128,128,128,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .step-text {
        flex: 1;
        line-height: 1.5;
        padding-top: 3px;
    }
    .gri-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem;
        background: rgba(108,142,191,0.18);
        color: inherit;
        border: 1px solid rgba(108,142,191,0.3);
    }
    .clarification-box {
        border: 2px solid #f39c12;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        background: rgba(243,156,18,0.08);
    }
    .clarification-box h4 {
        margin: 0 0 0.8rem 0;
        color: #f39c12;
    }
    .clarification-box ul {
        margin: 0 0 0.8rem 0;
        padding-left: 1.2rem;
    }
    .clarification-box li {
        margin-bottom: 0.4rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Initialize agent (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    return TariffAgent()


# ---------------------------------------------------------------------------
# Sidebar — compact branding + data status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Agentica</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-tagline">AI Tariff Classification Agent</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Data status
    sync_status = get_sync_status()
    st.markdown(
        f'<div class="sidebar-stat"><strong>2,186</strong> indexed chunks</div>'
        f'<div class="sidebar-stat"><strong>283</strong> CROSS rulings</div>'
        f'<div class="sidebar-stat"><strong>15</strong> HTS chapters</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Last synced: {sync_status['last_sync']}")
    if st.button("Sync Now", use_container_width=True):
        with st.spinner("Syncing tariff data..."):
            syncer = TariffSync()
            result = syncer.run()
        if result.get("errors"):
            st.warning(f"Sync completed with errors: {result['errors']}")
        else:
            st.success(f"Synced. {result['new_chunks_indexed']} new chunks indexed.")
        st.rerun()

    st.markdown("---")

    # Pipeline info — compact
    st.caption(
        "**How it works**\n\n"
        "Hybrid BM25 + vector search over the HTS schedule "
        "and CBP CROSS rulings, re-ranked with a cross-encoder, "
        "then classified by LLM using GRI rules."
    )

    st.markdown("---")
    st.caption(
        "This is an AI-assisted decision support tool. "
        "Final classifications should be reviewed by a licensed customs broker."
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">Classify a product</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Describe any product to get its U.S. HTS tariff code, duty rate, and trade advisory</p>',
    unsafe_allow_html=True,
)

# Quick-try example buttons
EXAMPLES = [
    "Lithium-ion battery pack, 48V, for e-bike",
    "Apple iPhone 2025",
    "Disc brake rotor for commercial truck",
    "Cotton men's dress shirt",
    "Industrial robot arm, 6-axis",
    "Wireless noise-cancelling earbuds",
]

# Check if an example was clicked
selected_example = ""
example_cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    with example_cols[i]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            selected_example = ex

default_val = selected_example

# Input — wrapped in a form so Enter key submits

COUNTRIES = [
    "(not specified)",
    "China",
    "Vietnam",
    "India",
    "Taiwan",
    "South Korea",
    "Japan",
    "Thailand",
    "Indonesia",
    "Mexico",
    "Canada",
    "Germany",
    "United Kingdom",
    "Italy",
    "France",
    "Hong Kong",
    "Malaysia",
    "Bangladesh",
    "Turkey",
    "Singapore",
    "Australia",
    "Israel",
    "Other",
]

with st.form(key="classify_form"):
    col1, col2, col3 = st.columns([4, 2, 1])
    with col1:
        query = st.text_input(
            "Describe the product to classify",
            value=default_val,
            placeholder="e.g., Bluetooth wireless speaker with rechargeable battery",
            label_visibility="collapsed",
        )
    with col2:
        country = st.selectbox("Country of Origin", COUNTRIES, label_visibility="collapsed")
    with col3:
        classify_btn = st.form_submit_button("Classify", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
if classify_btn and query:
    # Clear previous result so stale data doesn't flash on screen
    st.session_state.pop("result", None)
    agent = get_agent()
    country_val = "" if country == "(not specified)" else country

    with st.status("Classifying...", expanded=True) as status:
        st.write("Stage 1: Parsing product description...")
        t0 = time.time()

        # Run classification
        result = agent.classify(query, country_of_origin=country_val)

        status.update(label=f"Classified in {result.elapsed_seconds:.1f}s", state="complete", expanded=False)

    # Store in session state for display
    st.session_state["result"] = result.to_dict()
    st.session_state["original_query"] = query
    st.session_state["country_val"] = country_val
    st.session_state.pop("refinement_submitted", None)

# Handle refinement re-classification
if st.session_state.get("refinement_submitted"):
    agent = get_agent()
    refined_query = st.session_state["refinement_submitted"]
    country_val = st.session_state.get("country_val", "")

    with st.status("Re-classifying with additional details...", expanded=True) as status:
        result = agent.classify(refined_query, country_of_origin=country_val)
        status.update(label=f"Re-classified in {result.elapsed_seconds:.1f}s", state="complete", expanded=False)

    st.session_state["result"] = result.to_dict()
    st.session_state["original_query"] = refined_query
    st.session_state.pop("refinement_submitted", None)


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    r = st.session_state["result"]

    st.markdown("---")

    # Main result row
    col_code, col_conf, col_duty = st.columns([2, 1, 1])

    with col_code:
        st.markdown("**HTS Code**")
        st.markdown(f'<span class="hts-code">{r["hts_code"]}</span>', unsafe_allow_html=True)
        st.caption(r["description"])

    with col_conf:
        conf = r["confidence"]
        if conf >= 0.8:
            conf_class = "confidence-high"
            conf_label = "HIGH"
        elif conf >= 0.6:
            conf_class = "confidence-medium"
            conf_label = "MEDIUM"
        else:
            conf_class = "confidence-low"
            conf_label = "LOW"
        st.markdown("**Confidence**")
        st.markdown(
            f'<span class="{conf_class}">{conf:.0%} ({conf_label})</span>',
            unsafe_allow_html=True,
        )

    with col_duty:
        st.markdown("**Duty Rate (MFN)**")
        duty_display = r['duty_rate']
        if not duty_display:
            st.markdown("*Not available*")
            st.caption("Rate not found in indexed HTS data")
        elif "unverified" in duty_display.lower():
            st.markdown(f"**{duty_display}**")
            st.caption("Could not verify against HTS source")
        else:
            st.markdown(f"**{duty_display}**")

    # Tariff guardrail warnings
    if r.get("tariff_warnings"):
        for w in r["tariff_warnings"]:
            st.markdown(
                f'<div class="tariff-warning"><strong>\u26A0 Tariff Advisory:</strong> {w}</div>',
                unsafe_allow_html=True,
            )

    # Expert review alert
    if r["needs_expert_review"]:
        st.markdown(
            f'<div class="expert-alert">'
            f'<strong>Expert Review Recommended</strong><br>'
            f'{r["expert_review_reason"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Human-in-the-loop: clarifying questions
    if r.get("clarifying_questions"):
        questions_html = "".join(f"<li>{q}</li>" for q in r["clarifying_questions"])
        st.markdown(
            f'<div class="clarification-box">'
            f'<h4>The agent needs more information</h4>'
            f'<ul>{questions_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.form(key="refine_form"):
            additional_info = st.text_input(
                "Provide additional details",
                placeholder="e.g., made of stainless steel, used in automotive braking, weighs 12kg",
                label_visibility="collapsed",
            )
            refine_btn = st.form_submit_button("Re-classify with details", type="primary")
        if refine_btn and additional_info:
            original = st.session_state.get("original_query", r["query"])
            st.session_state["refinement_submitted"] = f"{original}. Additional details: {additional_info}"
            st.rerun()

    # Tabs for details
    tab_reasoning, tab_rulings, tab_alternatives, tab_raw = st.tabs([
        "Reasoning", "Similar Rulings", "Alternatives", "Raw JSON",
    ])

    with tab_reasoning:
        # Parse reasoning into steps
        reasoning_text = r["reasoning"]
        import re as _re
        steps = _re.split(r'(?i)step\s*\d+\s*[:.]\s*', reasoning_text)
        steps = [s.strip() for s in steps if s.strip()]

        if len(steps) > 1:
            html_steps = ""
            for i, step in enumerate(steps, 1):
                html_steps += (
                    f'<div class="reasoning-step">'
                    f'<div class="step-number">{i}</div>'
                    f'<div class="step-text">{step}</div>'
                    f'</div>'
                )
            st.markdown(html_steps, unsafe_allow_html=True)
        else:
            st.markdown(reasoning_text)

        if r["gri_rules_applied"]:
            badges = " ".join(
                f'<span class="gri-badge">{rule}</span>'
                for rule in r["gri_rules_applied"]
            )
            st.markdown(
                f'<div style="margin-top: 0.8rem;"><strong>GRI Rules Applied</strong><br>{badges}</div>',
                unsafe_allow_html=True,
            )

    with tab_rulings:
        if r["similar_rulings"]:
            for ruling in r["similar_rulings"]:
                st.markdown(f'<div class="ruling-card">', unsafe_allow_html=True)
                st.markdown(f"**{ruling['ruling']}**")
                st.caption(ruling["text"])
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No similar CROSS rulings found.")

    with tab_alternatives:
        if r["alternative_codes"]:
            for alt in r["alternative_codes"]:
                st.markdown(f"- **{alt['code']}**: {alt['reason']}")
        else:
            st.info("No alternative classifications identified.")

    with tab_raw:
        st.json(r)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Agentica v0.1 | Data: USITC HTS 2026 + CBP CROSS rulings | "
    "AI-assisted — verify with a licensed customs broker before import decisions."
)
