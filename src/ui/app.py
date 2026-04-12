"""
Streamlit UI for the Tariff Classification Agent.

Run: streamlit run src/ui/app.py
"""

import os
import sys
import json
import time
from pathlib import Path

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
    /* Hide Streamlit deploy button */
    .stDeployButton, [data-testid="stStatusWidget"] { display: none; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .confidence-high { color: #2ecc71; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #e74c3c; font-weight: bold; }
    .hts-code {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        background: #ecf0f1;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-family: monospace;
    }
    .expert-alert {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .ruling-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    div[data-testid="stTextInput"] input {
        font-size: 1.1rem;
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
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")

    example_products = [
        "-- Select an example --",
        "Bluetooth wireless speaker with LED lights",
        "Smart home hub with voice assistant and Zigbee radio",
        "Lithium-ion battery pack for electric vehicles, 400V 75kWh",
        "Stainless steel wristwatch with analog display",
        "Carbon fiber drone frame with 4 motor mounts",
        "Industrial robot arm, 6-axis, 10kg payload",
        "Wireless earbuds with active noise cancellation",
        "Digital smart thermostat with WiFi and touchscreen",
    ]
    example = st.selectbox("Example products", example_products)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This agent classifies products into the U.S. Harmonized Tariff Schedule (HTS) "
        "using AI-powered analysis of the official HTS schedule and CBP CROSS rulings database."
    )
    st.markdown(
        "**Data sources:**\n"
        "- USITC HTS Schedule (2026 Rev. 5)\n"
        "- CBP CROSS Rulings (220,000+ rulings)\n"
    )
    st.markdown(
        "**Pipeline:**\n"
        "1. Query parsing & expansion\n"
        "2. Hybrid BM25 + vector search\n"
        "3. Cross-encoder re-ranking\n"
        "4. LLM classification with GRI rules\n"
    )
    st.markdown("---")
    st.markdown(
        "*Expert-in-the-loop: Low-confidence results are flagged for human review. "
        "This tool assists but does not replace licensed customs brokers.*"
    )

    # --- Data Sync ---
    st.markdown("---")
    st.markdown("### Data Sync")
    sync_status = get_sync_status()
    st.caption(f"Last synced: {sync_status['last_sync']}")
    st.caption(f"CROSS rulings: {sync_status['cross_ruling_count']:,}")
    st.caption(f"CROSS updated: {sync_status['cross_last_update']}")
    if st.button("Sync Now", use_container_width=True):
        with st.spinner("Syncing tariff data..."):
            syncer = TariffSync()
            result = syncer.run()
        if result.get("errors"):
            st.warning(f"Sync completed with errors: {result['errors']}")
        else:
            st.success(f"Synced. {result['new_chunks_indexed']} new chunks indexed.")
        st.rerun()


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">Tariff Classification Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-powered HTS code classification using RAG over official tariff schedules and CBP rulings</p>',
    unsafe_allow_html=True,
)

# Input
col1, col2 = st.columns([5, 1])
with col1:
    default_val = example if example != "-- Select an example --" else ""
    query = st.text_input(
        "Describe the product to classify",
        value=default_val,
        placeholder="e.g., Bluetooth wireless speaker with rechargeable battery",
        label_visibility="collapsed",
    )
with col2:
    classify_btn = st.button("Classify", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
if classify_btn and query:
    agent = get_agent()

    with st.status("Classifying...", expanded=True) as status:
        st.write("Stage 1: Parsing product description...")
        t0 = time.time()

        # Run classification
        result = agent.classify(query)

        status.update(label=f"Classified in {result.elapsed_seconds:.1f}s", state="complete", expanded=False)

    # Store in session state for display
    st.session_state["result"] = result.to_dict()


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
        st.markdown("**Duty Rate**")
        st.markdown(f"**{r['duty_rate'] or 'N/A'}**")

    # Expert review alert
    if r["needs_expert_review"]:
        st.markdown(
            f'<div class="expert-alert">'
            f'<strong>Expert Review Recommended</strong><br>'
            f'{r["expert_review_reason"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Tabs for details
    tab_reasoning, tab_rulings, tab_alternatives, tab_raw = st.tabs([
        "Reasoning", "Similar Rulings", "Alternatives", "Raw JSON",
    ])

    with tab_reasoning:
        st.markdown("**Classification Reasoning**")
        st.markdown(r["reasoning"])
        if r["gri_rules_applied"]:
            st.markdown(f"**GRI Rules Applied:** {', '.join(r['gri_rules_applied'])}")

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
    "Tariff Classification Agent v0.1 | Data: USITC HTS 2026 Rev. 5 + CBP CROSS | "
    "This is an AI-assisted tool. Final classification decisions should be reviewed by a licensed customs broker."
)
