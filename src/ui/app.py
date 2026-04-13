"""
Agentica - AI Tariff Classification Agent (Chat UI)

Run: streamlit run src/ui/app.py
"""

import os
import sys
import csv
import io
import json
import time
import warnings
import re as _re
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message=".*torchvision.*")

os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.agent.orchestrator import TariffAgent, ClassificationResult
from src.agent.tools import call_tool, list_tools, list_tools_by_category, ToolResult, TOOL_CATEGORIES
from src.data.sync import TariffSync, get_sync_status

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentica",
    page_icon="https://img.icons8.com/color/48/customs-officer.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stDeployButton, [data-testid="stStatusWidget"] { display: none; }
    #MainMenu { visibility: hidden; }
    /* Keep header visible so sidebar toggle (collapse/expand arrow) works */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }
    /* Sidebar: resizable via drag, default 320px, min 260px */
    section[data-testid="stSidebar"] {
        min-width: 260px;
        max-width: 520px;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    .sidebar-brand { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.1rem; }
    .sidebar-tagline { font-size: 0.8rem; opacity: 0.55; margin-bottom: 1rem; }
    .result-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px; padding: 1.2rem; margin: 0.6rem 0;
    }
    .hts-code {
        font-size: 1.8rem; font-weight: 700; font-family: monospace;
        padding: 0.3rem 0.8rem; border-radius: 6px;
        border: 1px solid rgba(128,128,128,0.3); display: inline-block;
    }
    .confidence-high { color: #2ecc71; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #e74c3c; font-weight: bold; }
    .tariff-warning {
        border-left: 4px solid #e74c3c; padding: 0.5rem 0.8rem; border-radius: 4px;
        margin: 0.3rem 0; font-size: 0.85rem; line-height: 1.4;
        background: rgba(231,76,60,0.10); color: inherit;
    }
    .tariff-warning strong { color: #e74c3c; }
    .expert-alert {
        border-left: 4px solid #6c8ebf; padding: 0.7rem; border-radius: 4px;
        margin: 0.4rem 0; background: rgba(108,142,191,0.12); color: inherit;
        font-size: 0.88rem;
    }
    .clarification-box {
        border: 2px solid #f39c12; border-radius: 8px; padding: 0.8rem;
        margin: 0.4rem 0; background: rgba(243,156,18,0.08);
    }
    .clarification-box h4 { margin: 0 0 0.4rem 0; color: #f39c12; font-size: 0.95rem; }
    .clarification-box ul { margin: 0; padding-left: 1.1rem; }
    .clarification-box li { margin-bottom: 0.2rem; font-size: 0.88rem; }
    .reasoning-step {
        display: flex; gap: 0.6rem; padding: 0.4rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.10);
    }
    .reasoning-step:last-child { border-bottom: none; }
    .step-number {
        flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%;
        background: rgba(128,128,128,0.15); display: flex;
        align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.78rem;
    }
    .step-text { flex: 1; line-height: 1.4; padding-top: 2px; font-size: 0.88rem; }
    .gri-badge {
        display: inline-block; padding: 0.12rem 0.45rem; border-radius: 10px;
        font-size: 0.73rem; font-weight: 600; margin: 0.1rem;
        background: rgba(108,142,191,0.18); border: 1px solid rgba(108,142,191,0.3);
        color: inherit;
    }
    .ruling-card {
        border: 1px solid rgba(128,128,128,0.22); border-radius: 6px;
        padding: 0.7rem; margin: 0.3rem 0; font-size: 0.88rem;
    }
    .history-item {
        padding: 0.4rem 0.5rem; border-radius: 6px; margin: 0.2rem 0;
        border: 1px solid rgba(128,128,128,0.18); font-size: 0.78rem;
    }
    .history-hts { font-weight: 700; font-family: monospace; }
    .history-query { opacity: 0.65; font-size: 0.72rem; }
    .tool-call-box {
        border: 1px solid rgba(108,142,191,0.35); border-radius: 8px;
        padding: 0rem; margin: 0.5rem 0; overflow: hidden;
    }
    .tool-call-header {
        background: rgba(108,142,191,0.12); padding: 0.5rem 0.8rem;
        font-size: 0.82rem; font-weight: 600; display: flex;
        align-items: center; gap: 0.5rem;
    }
    .tool-call-header .tool-icon { font-size: 1rem; }
    .tool-call-params {
        padding: 0.4rem 0.8rem; font-size: 0.76rem;
        font-family: monospace; border-bottom: 1px solid rgba(128,128,128,0.12);
        opacity: 0.7;
    }
    .tool-call-status {
        padding: 0.3rem 0.8rem; font-size: 0.74rem;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .tool-call-status.success { color: #2ecc71; }
    .tool-call-status.error { color: #e74c3c; }
    .tool-grid {
        display: grid; grid-template-columns: 1fr 1fr;
        gap: 0.35rem; margin: 0.3rem 0;
    }
    .tool-chip {
        display: flex; align-items: center; gap: 0.35rem;
        padding: 0.3rem 0.5rem; border-radius: 6px;
        border: 1px solid rgba(128,128,128,0.2);
        font-size: 0.68rem; line-height: 1.2;
        background: rgba(128,128,128,0.04);
    }
    .tool-chip .chip-icon { font-size: 0.85rem; flex-shrink: 0; }
    .tool-chip .chip-name { font-weight: 600; }
    .tool-chip .chip-int {
        font-size: 0.58rem; opacity: 0.5; text-transform: uppercase;
    }
    .cat-label {
        font-size: 0.7rem; font-weight: 700; opacity: 0.45;
        text-transform: uppercase; letter-spacing: 0.05em;
        margin: 0.5rem 0 0.15rem 0;
    }
    /* CRM Pipeline */
    .pipeline-bar {
        display: flex; gap: 0; margin: 0.5rem 0 0.8rem 0;
        border-radius: 8px; overflow: hidden;
        border: 1px solid rgba(128,128,128,0.2);
    }
    .pipeline-stage {
        flex: 1; padding: 0.5rem 0.3rem; text-align: center;
        font-size: 0.72rem; font-weight: 600; position: relative;
        transition: all 0.3s ease;
        border-right: 1px solid rgba(128,128,128,0.15);
    }
    .pipeline-stage:last-child { border-right: none; }
    .pipeline-stage .stage-icon { font-size: 1rem; display: block; margin-bottom: 0.15rem; }
    .pipeline-stage .stage-label { opacity: 0.5; }
    .pipeline-stage.completed {
        background: rgba(46,204,113,0.12);
    }
    .pipeline-stage.completed .stage-label { opacity: 1; color: #2ecc71; }
    .pipeline-stage.active {
        background: rgba(108,142,191,0.15);
    }
    .pipeline-stage.active .stage-label { opacity: 1; color: #6c8ebf; }
    .pipeline-stage.pending .stage-label { opacity: 0.3; }
    .deal-header {
        font-size: 0.78rem; opacity: 0.5; margin-bottom: 0.2rem;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .deal-id { font-family: monospace; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "uploaded_products" not in st.session_state:
    st.session_state.uploaded_products = []
if "deal" not in st.session_state:
    st.session_state.deal = None  # Active CRM deal


PIPELINE_STAGES = [
    {"key": "new", "label": "New Deal", "icon": "&#x1F4CB;"},
    {"key": "classified", "label": "Classified", "icon": "&#x1F50D;"},
    {"key": "costed", "label": "Costed", "icon": "&#x1F4B0;"},
    {"key": "reviewed", "label": "Reviewed", "icon": "&#x2696;"},
    {"key": "synced", "label": "CRM Synced", "icon": "&#x2705;"},
]


def create_deal(query: str, country: str = "") -> dict:
    """Create a new CRM deal from a classification request."""
    deal_id = f"DEAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return {
        "id": deal_id,
        "query": query,
        "country": country,
        "stage": "new",
        "classification": None,
        "landed_cost": None,
        "tools_used": [],
        "created": datetime.now().strftime("%H:%M:%S"),
    }


def advance_deal(stage: str, tool_name: str = ""):
    """Advance the active deal to a pipeline stage."""
    if st.session_state.deal:
        st.session_state.deal["stage"] = stage
        if tool_name:
            st.session_state.deal["tools_used"].append(tool_name)


def render_pipeline() -> str:
    """Render the CRM pipeline visualization."""
    deal = st.session_state.deal
    if not deal:
        return ""
    stage_order = [s["key"] for s in PIPELINE_STAGES]
    current_idx = stage_order.index(deal["stage"]) if deal["stage"] in stage_order else -1

    html = f'<div class="deal-header"><span class="deal-id">{deal["id"]}</span>'
    html += f' {deal["query"][:50]}'
    if deal["country"]:
        html += f' &middot; {deal["country"]}'
    html += '</div>'
    html += '<div class="pipeline-bar">'
    for i, s in enumerate(PIPELINE_STAGES):
        if i < current_idx:
            css = "completed"
        elif i == current_idx:
            css = "active"
        else:
            css = "pending"
        html += (f'<div class="pipeline-stage {css}">'
                 f'<span class="stage-icon">{s["icon"]}</span>'
                 f'<span class="stage-label">{s["label"]}</span></div>')
    html += '</div>'
    return html


@st.cache_resource
def get_agent():
    return TariffAgent()


def parse_csv_upload(file_bytes: bytes) -> list[dict]:
    """Parse uploaded CSV into list of product dicts."""
    text = file_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    products = []
    for row in reader:
        # Normalize column names (case-insensitive)
        norm = {k.strip().lower(): v.strip() for k, v in row.items() if k}
        desc = norm.get("product") or norm.get("description") or norm.get("item") or norm.get("name") or ""
        if not desc:
            continue
        products.append({
            "description": desc,
            "quantity": norm.get("quantity") or norm.get("qty") or "1",
            "unit_value": norm.get("unit_value") or norm.get("price") or norm.get("value") or "0",
            "country": norm.get("country") or norm.get("origin") or norm.get("country_of_origin") or "",
        })
    return products


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def _estimate_effective_rate(r):
    """Compute estimated effective duty rate including Section 301/IEEPA if present."""
    duty_str = r.get("duty_rate", "")
    base_pct = 0.0
    if duty_str:
        m = _re.search(r'(\d+\.?\d*)%', duty_str)
        if m:
            base_pct = float(m.group(1))
    # Check warnings for additional tariffs
    extra = 0.0
    for w in r.get("tariff_warnings", []):
        if "301" in w or "IEEPA" in w:
            extra = 25.0
            break
        elif "reciprocal" in w.lower():
            m2 = _re.search(r'(\d+)%', w)
            if m2:
                extra = float(m2.group(1))
                break
    return base_pct, extra


def render_result_card(r):
    conf = r["confidence"]
    if conf >= 0.8:
        cc, cl = "confidence-high", "HIGH"
    elif conf >= 0.6:
        cc, cl = "confidence-medium", "MEDIUM"
    else:
        cc, cl = "confidence-low", "LOW"
    duty = r.get("duty_rate", "")
    if not duty:
        dh = "<em>Not available</em>"
    elif "unverified" in duty.lower():
        dh = duty
    else:
        dh = f"<strong>{duty}</strong>"
    # Effective rate (MFN + Section 301/IEEPA/reciprocal)
    base_pct, extra_pct = _estimate_effective_rate(r)
    if extra_pct > 0:
        eff_total = base_pct + extra_pct
        eff_html = (
            f'<div><div style="font-size:0.78rem;opacity:0.55;">Est. Effective Rate</div>'
            f'<span class="confidence-low"><strong>{eff_total:.0f}%</strong></span>'
            f'<div style="font-size:0.72rem;opacity:0.5;">MFN {base_pct:.0f}% + additional {extra_pct:.0f}%</div></div>'
        )
    else:
        eff_html = ""
    return f"""<div class="result-card">
<div style="display:flex;gap:2rem;align-items:flex-start;flex-wrap:wrap;">
<div><div style="font-size:0.78rem;opacity:0.55;">HTS Code</div>
<span class="hts-code">{r['hts_code']}</span>
<div style="font-size:0.83rem;opacity:0.65;margin-top:0.2rem;">{r['description']}</div></div>
<div><div style="font-size:0.78rem;opacity:0.55;">Confidence</div>
<span class="{cc}">{conf:.0%} ({cl})</span></div>
<div><div style="font-size:0.78rem;opacity:0.55;">Duty Rate (MFN)</div>{dh}</div>
{eff_html}
</div></div>"""


def render_warnings(r):
    h = ""
    for w in r.get("tariff_warnings", []):
        h += f'<div class="tariff-warning"><strong>⚠ Tariff Advisory:</strong> {w}</div>'
    if r.get("needs_expert_review"):
        h += f'<div class="expert-alert"><strong>Expert Review Recommended</strong><br>{r["expert_review_reason"]}</div>'
    return h


def render_reasoning(r):
    text = r.get("reasoning", "")
    steps = _re.split(r'(?i)step\s*\d+\s*[:.]\s*', text)
    steps = [s.strip() for s in steps if s.strip()]
    if len(steps) > 1:
        h = ""
        for i, s in enumerate(steps, 1):
            h += f'<div class="reasoning-step"><div class="step-number">{i}</div><div class="step-text">{s}</div></div>'
    else:
        h = f"<div style='font-size:0.88rem;'>{text}</div>"
    if r.get("gri_rules_applied"):
        badges = " ".join(f'<span class="gri-badge">{rule}</span>' for rule in r["gri_rules_applied"])
        h += f'<div style="margin-top:0.5rem;"><strong>GRI Rules</strong> {badges}</div>'
    return h


def render_clarifications(r):
    qs = r.get("clarifying_questions", [])
    if not qs:
        return ""
    items = "".join(f"<li>{q}</li>" for q in qs)
    return f'<div class="clarification-box"><h4>The agent needs more information</h4><ul>{items}</ul></div>'


def render_rulings(r):
    if not r.get("similar_rulings"):
        return ""
    h = ""
    for ruling in r["similar_rulings"]:
        h += f'<div class="ruling-card"><strong>{ruling["ruling"]}</strong><br><span style="opacity:0.7;">{ruling["text"]}</span></div>'
    return h


# ---------------------------------------------------------------------------
# Agent actions
# ---------------------------------------------------------------------------
def add_msg(role, content, html=False):
    st.session_state.messages.append({"role": role, "content": content, "html": html})


def do_classify(query, country_str):
    agent = get_agent()
    cval = country_str.strip() if country_str else ""

    # Create CRM deal
    st.session_state.deal = create_deal(query, cval)

    result = agent.classify(query, country_of_origin=cval)
    r = result.to_dict()

    # Advance deal
    advance_deal("classified", "get_hts_code")
    st.session_state.deal["classification"] = r

    # Build agent response
    parts = [render_result_card(r)]
    w = render_warnings(r)
    if w:
        parts.append(w)
    cl = render_clarifications(r)
    if cl:
        parts.append(cl)
    add_msg("assistant", "\n".join(parts), html=True)

    # Reasoning (collapsible)
    reasoning = render_reasoning(r)
    add_msg("assistant", f"<details><summary><strong>Classification reasoning</strong></summary>{reasoning}</details>", html=True)

    # Rulings
    rulings = render_rulings(r)
    if rulings:
        add_msg("assistant", f"<details><summary><strong>Similar CROSS rulings ({len(r['similar_rulings'])})</strong></summary>{rulings}</details>", html=True)

    # Save to history
    st.session_state.history.append({
        "query": query, "country": country_str, "result": r,
        "timestamp": datetime.now().strftime("%H:%M"),
    })

    # Suggest next actions
    actions = ["Run full pipeline", "Generate invoice", "Calculate landed cost"]
    if r.get("tariff_warnings"):
        actions.extend(["Draft supplier letter", "Draft surcharge notice",
                        "File exemption request", "Send report via email"])
    actions.extend(["Compare countries", "Slack alert", "Push to SAP",
                    "Export controls check", "Impact report"])
    add_msg("assistant", "**What next?** " + " · ".join(f"*{a}*" for a in actions))

    return r


def do_full_pipeline(classification: dict):
    """Run the full CRM pipeline: Classify (done) -> Cost -> Review -> Sync."""
    country_val = ""
    if st.session_state.deal:
        country_val = st.session_state.deal.get("country", "") or "China"

    # Stage 2: Costed — landed cost calculation
    add_msg("assistant", "**Stage 2/4 — Calculating landed cost...**")
    result = call_tool("landed_cost_calculator", classification, country=country_val)
    add_msg("assistant", render_tool_call(
        result.tool_name, result.params_used, result.status, result.elapsed_ms
    ), html=True)
    add_msg("assistant", result.output)
    advance_deal("costed", "landed_cost_calculator")
    time.sleep(1)

    # Stage 3: Reviewed — generate invoice + compliance check
    add_msg("assistant", "**Stage 3/4 — Generating invoice and compliance review...**")
    inv_result = call_tool("generate_invoice", classification, country=country_val)
    add_msg("assistant", render_tool_call(
        inv_result.tool_name, inv_result.params_used, inv_result.status, inv_result.elapsed_ms
    ), html=True)
    add_msg("assistant", inv_result.output)
    advance_deal("reviewed", "generate_invoice")
    time.sleep(1)

    # Stage 4: Synced — push to CRM
    add_msg("assistant", "**Stage 4/4 — Syncing to CRM...**")
    crm_result = call_tool("zoho_create_deal", classification)
    add_msg("assistant", render_tool_call(
        crm_result.tool_name, crm_result.params_used, crm_result.status, crm_result.elapsed_ms
    ), html=True)
    add_msg("assistant", crm_result.output)
    advance_deal("synced", "zoho_create_deal")

    # Summary
    deal = st.session_state.deal
    n_tools = len(deal["tools_used"]) if deal else 0
    add_msg("assistant",
        f"**Pipeline complete.** Deal `{deal['id']}` processed through {n_tools} tool calls: "
        f"`{'` → `'.join(deal['tools_used'])}`\n\n"
        "The deal is now in the CRM with classification, landed cost, and invoice attached. "
        "A compliance officer can review and approve from the CRM dashboard."
    )


def do_compare(base_query):
    agent = get_agent()
    countries = ["China", "Vietnam", "Mexico", "India", "Japan"]
    add_msg("assistant", f"Comparing **{base_query}** across {len(countries)} origin countries...")

    rows = []
    for c in countries:
        r = agent.classify(base_query, country_of_origin=c)
        rd = r.to_dict()
        nw = len(rd.get("tariff_warnings", []))
        exp = "Yes" if rd.get("needs_expert_review") else "No"
        rows.append(f"| {c} | `{rd['hts_code']}` | {rd.get('duty_rate','N/A')} | {nw} | {exp} |")
        time.sleep(1)

    table = "| Country | HTS Code | Base Duty | Advisories | Expert Review |\n|---|---|---|---|---|\n" + "\n".join(rows)
    add_msg("assistant", table)
    add_msg("assistant", "Base MFN rate is identical across origins. The difference is in **additional tariffs** (Section 301, IEEPA, reciprocal) that vary by country. China typically has the highest effective rate.")


def do_checklist(r):
    query = r.get("query", "")
    code = r.get("hts_code", "")
    has_warnings = bool(r.get("tariff_warnings"))
    expert = r.get("needs_expert_review", False)

    cl = f"**Import Compliance Checklist — {code}**\n\n"
    cl += f"Product: {query}\n\n"
    cl += "- [ ] Verify HTS code against official USITC HTS schedule\n"
    cl += "- [ ] Confirm product description matches tariff line item\n"
    cl += "- [ ] Obtain country of origin certificate from supplier\n"
    if has_warnings:
        cl += "- [ ] Check Section 301 duty applicability\n"
        cl += "- [ ] Check IEEPA / reciprocal tariff status\n"
        cl += "- [ ] Verify no AD/CVD orders apply\n"
    cl += "- [ ] Check FTA preferential rate eligibility\n"
    cl += "- [ ] Verify import licenses/permits (FDA/FCC/CPSC if applicable)\n"
    if expert:
        cl += "- [ ] **Submit to licensed customs broker for review**\n"
    cl += "- [ ] File CBP entry (CF 3461 / CF 7501)\n"
    cl += "- [ ] Retain classification records for 5 years\n"
    add_msg("assistant", cl)


def do_explain_alternatives(r):
    alts = r.get("alternative_codes", [])
    if not alts:
        add_msg("assistant", "No alternative classifications were identified for this product.")
        return
    msg = f"**Why {r['hts_code']} was chosen over alternatives:**\n\n"
    for alt in alts:
        msg += f"- **{alt['code']}** — {alt.get('reason', 'No detail')}\n"
    add_msg("assistant", msg)


def render_tool_call(tool_name: str, params: dict, status: str, elapsed_ms: int = 0) -> str:
    """Render a tool-call status card (HTML)."""
    desc_map = {t["name"]: t["description"] for t in list_tools()}
    desc = desc_map.get(tool_name, "")
    params_str = ", ".join(f"{k}={json.dumps(v) if isinstance(v, str) else v}" for k, v in params.items() if k != "hts_code")
    status_class = "success" if status == "success" else "error"
    status_icon = "&#x2713;" if status == "success" else "&#x2717;"
    elapsed = f" ({elapsed_ms}ms)" if elapsed_ms else ""
    return (
        f'<div class="tool-call-box">'
        f'<div class="tool-call-header"><span class="tool-icon">&#x1F527;</span> '
        f'<strong>{tool_name}</strong>'
        f'<span style="opacity:0.5;font-weight:400;"> — {desc}</span></div>'
        f'<div class="tool-call-params">{params_str}</div>'
        f'<div class="tool-call-status {status_class}">{status_icon} {status}{elapsed}</div>'
        f'</div>'
    )


def do_tool_call(tool_name: str, classification: dict, **kwargs):
    """Execute a tool and render the call + output in chat."""
    result = call_tool(tool_name, classification, **kwargs)
    # Show the tool-call card
    add_msg("assistant", render_tool_call(
        result.tool_name, result.params_used, result.status, result.elapsed_ms
    ), html=True)
    # Show the tool output
    add_msg("assistant", result.output)
    # Advance deal if applicable
    if st.session_state.deal:
        costing_tools = {"landed_cost_calculator", "generate_invoice"}
        review_tools = {"draft_exemption_request", "export_controls_check", "draft_supplier_letter",
                        "draft_surcharge_notice"}
        sync_tools = {"zoho_create_deal", "zoho_create_invoice", "wave_create_invoice",
                      "quickbooks_invoice", "sap_update_material", "erp_update_po",
                      "send_email", "slack_notify", "teams_notify"}
        if tool_name in costing_tools:
            advance_deal("costed", tool_name)
        elif tool_name in review_tools:
            advance_deal("reviewed", tool_name)
        elif tool_name in sync_tools:
            advance_deal("synced", tool_name)
        else:
            advance_deal(st.session_state.deal["stage"], tool_name)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Agentica</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-tagline">Agentic CRM for Global Trade</p>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Tool showcase ---
    st.markdown("**MCP Tools**")
    grouped = list_tools_by_category()
    for cat_key, cat_meta in TOOL_CATEGORIES.items():
        tools_in_cat = grouped.get(cat_key, [])
        if not tools_in_cat:
            continue
        st.markdown(f'<div class="cat-label">{cat_meta["icon"]} {cat_meta["label"]}</div>', unsafe_allow_html=True)
        chips_html = '<div class="tool-grid">'
        for t in tools_in_cat:
            int_label = f'<div class="chip-int">{t["integration"]}</div>' if t.get("integration") else ""
            chips_html += (
                f'<div class="tool-chip">'
                f'<span class="chip-icon">{t.get("icon", "")}</span>'
                f'<div><div class="chip-name">{t["name"].replace("_", " ").title()}</div>'
                f'{int_label}</div></div>'
            )
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("---")

    # --- CSV Upload ---
    st.markdown("**Bulk Upload**")
    uploaded = st.file_uploader("Product list (CSV)", type=["csv"],
                                label_visibility="collapsed",
                                help="CSV with columns: product/description, quantity, unit_value, country")
    if uploaded is not None:
        products = parse_csv_upload(uploaded.read())
        if products:
            st.session_state.uploaded_products = products
            st.success(f"{len(products)} products loaded")
            if st.button("Classify all", use_container_width=True):
                add_msg("user", f"Batch classify {len(products)} products from uploaded CSV")
                agent = get_agent()
                with st.spinner(f"Classifying {len(products)} products..."):
                    result = call_tool("batch_classify", {}, products=products, agent=agent)
                add_msg("assistant", render_tool_call(
                    result.tool_name, result.params_used, result.status, result.elapsed_ms
                ), html=True)
                add_msg("assistant", result.output)
                st.rerun()
        else:
            st.warning("No valid rows found. Need column: product or description")

    st.markdown("---")

    # --- Data metrics ---
    sync_status = get_sync_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", "2,186")
    c2.metric("Rulings", "283")
    c3.metric("Chapters", "15")
    st.caption(f"Synced: {sync_status['last_sync']}")

    # --- History ---
    if st.session_state.history:
        st.markdown("---")
        st.markdown("**History**")
        for i, h in enumerate(reversed(st.session_state.history)):
            code = h["result"].get("hts_code", "???")
            conf = h["result"].get("confidence", 0)
            st.markdown(
                f'<div class="history-item">'
                f'<span class="history-hts">{code}</span> '
                f'<span style="opacity:0.45;">({conf:.0%})</span><br>'
                f'<span class="history-query">{h["query"][:45]}</span>'
                f'<span style="float:right;opacity:0.35;font-size:0.68rem;">{h["timestamp"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if i >= 9:
                break

    st.markdown("---")
    st.caption("AI-assisted. Verify with a licensed customs broker.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.uploaded_products = []
        st.session_state.deal = None
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
# CRM Pipeline visualization
pipeline_html = render_pipeline()
if pipeline_html:
    st.markdown(pipeline_html, unsafe_allow_html=True)

# Welcome message
if not st.session_state.messages:
    tools_count = len(list_tools())
    add_msg("assistant",
        f"I am **Agentica** — an agentic CRM for global trade compliance, with **{tools_count} MCP tools** connected.\n\n"
        "**How it works:** Describe a product and I create a *deal* that flows through a CRM pipeline:\n\n"
        "1. **Classify** — AI classifies the product into an HTS code using RAG over tariff schedules and 283 CBP rulings\n"
        "2. **Cost** — Calculate landed cost with duties, freight, Section 301/IEEPA surcharges\n"
        "3. **Review** — Generate invoice, flag tariff advisories, human-in-the-loop governance\n"
        "4. **Sync** — Push to CRM (Zoho/Salesforce), ERP (SAP/Oracle), accounting (QuickBooks/Wave)\n\n"
        "Say **\"run full pipeline\"** after classifying to execute all stages automatically, or run each tool individually.\n\n"
        "**Quick actions:** *Draft supplier cost-sharing letter* · *Announce surcharge to customers* · "
        "*File tariff exemption with USTR* · *Send report via Gmail* · *Post alert to Slack/Teams*\n\n"
        "Try: *Lithium-ion battery pack for EV from China*"
    )

# Render all messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("html"):
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Country extraction helper
# ---------------------------------------------------------------------------
_KNOWN_COUNTRIES = [
    "china", "vietnam", "india", "taiwan", "south korea", "japan", "thailand",
    "indonesia", "mexico", "canada", "germany", "united kingdom", "italy",
    "france", "hong kong", "malaysia", "bangladesh", "turkey", "singapore",
    "australia", "israel", "brazil", "philippines", "cambodia", "pakistan",
    "sri lanka", "egypt", "south africa", "russia", "iran", "north korea",
    "syria", "cuba",
]

def _extract_country(text: str) -> str:
    """Extract a country name from user text if present."""
    tl = text.lower()
    for c in sorted(_KNOWN_COUNTRIES, key=len, reverse=True):
        if c in tl:
            return c.title()
    if "from " in tl:
        m = _re.search(r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
        if m:
            return m.group(1)
    return ""


# Chat input
if prompt := st.chat_input("Describe a product, or ask a follow-up..."):
    add_msg("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    pl = prompt.lower().strip()
    last_r = st.session_state.history[-1]["result"] if st.session_state.history else None

    with st.chat_message("assistant"):
        # === Full CRM pipeline ===
        if last_r and any(kw in pl for kw in ["full pipeline", "run pipeline", "run all", "full workflow", "automate"]):
            with st.spinner("Running full CRM pipeline..."):
                do_full_pipeline(last_r)

        # === Compare countries ===
        elif last_r and any(kw in pl for kw in ["compare", "countries", "across", "other country"]):
            with st.spinner("Comparing across countries..."):
                do_compare(last_r["query"])

        # === Compliance checklist ===
        elif last_r and any(kw in pl for kw in ["checklist", "compliance", "what do i need", "steps to import"]):
            do_checklist(last_r)

        # === Explain alternatives ===
        elif last_r and any(kw in pl for kw in ["alternative", "why not", "other code", "ruled out"]):
            do_explain_alternatives(last_r)

        # === Document tools ===
        elif last_r and any(kw in pl for kw in ["invoice", "customs invoice"]) and "zoho" not in pl and "wave" not in pl and "quickbooks" not in pl:
            with st.spinner("Generating invoice..."):
                country_val = st.session_state.history[-1].get("country", "") or "China"
                do_tool_call("generate_invoice", last_r, country=country_val)

        elif last_r and any(kw in pl for kw in ["landed cost", "landed", "total cost", "freight"]):
            with st.spinner("Calculating landed cost..."):
                country_val = st.session_state.history[-1].get("country", "") or "China"
                do_tool_call("landed_cost_calculator", last_r, country=country_val)

        # === Communication tools ===
        elif last_r and any(kw in pl for kw in ["supplier", "cost-sharing", "cost sharing", "negotiate", "vendor"]):
            with st.spinner("Drafting supplier letter..."):
                do_tool_call("draft_supplier_letter", last_r)

        elif last_r and any(kw in pl for kw in ["surcharge", "price hike", "price increase", "customer notice", "announce"]):
            with st.spinner("Drafting surcharge notice..."):
                do_tool_call("draft_surcharge_notice", last_r)

        elif last_r and any(kw in pl for kw in ["email", "gmail", "outlook", "send report", "send email"]):
            with st.spinner("Sending email..."):
                provider = "outlook" if "outlook" in pl else "gmail"
                do_tool_call("send_email", last_r, provider=provider)

        elif last_r and any(kw in pl for kw in ["slack", "slack alert", "slack notify"]):
            with st.spinner("Posting to Slack..."):
                do_tool_call("slack_notify", last_r)

        elif last_r and any(kw in pl for kw in ["teams", "microsoft teams", "teams notify"]):
            with st.spinner("Posting to Teams..."):
                do_tool_call("teams_notify", last_r)

        # === Regulatory tools ===
        elif last_r and any(kw in pl for kw in ["exemption", "exclusion", "exempt", "waiver", "ustr"]):
            with st.spinner("Drafting exemption request..."):
                do_tool_call("draft_exemption_request", last_r)

        elif last_r and any(kw in pl for kw in ["export control", "ear", "bis", "screen", "denied party"]):
            with st.spinner("Running export controls screening..."):
                do_tool_call("export_controls_check", last_r)

        elif any(kw in pl for kw in ["hts lookup", "duty rate", "usitc", "lookup rate", "live rate", "check rate", "tariff rate"]):
            query = classification.get("hts_code", prompt) if (classification := (last_r or {})) else prompt
            with st.spinner("Querying USITC HTS API..."):
                do_tool_call("hts_lookup", last_r or {"query": prompt}, query=prompt)

        # === Integration tools ===
        elif last_r and any(kw in pl for kw in ["zoho invoice", "zoho books"]):
            with st.spinner("Creating Zoho Books invoice..."):
                do_tool_call("zoho_create_invoice", last_r)

        elif last_r and any(kw in pl for kw in ["zoho crm", "zoho deal", "create deal"]):
            with st.spinner("Creating Zoho CRM deal..."):
                do_tool_call("zoho_create_deal", last_r)

        elif last_r and any(kw in pl for kw in ["wave", "wave invoice"]):
            with st.spinner("Creating Wave invoice..."):
                do_tool_call("wave_create_invoice", last_r)

        elif last_r and any(kw in pl for kw in ["quickbooks", "qb invoice", "qbo"]):
            with st.spinner("Creating QuickBooks invoice..."):
                do_tool_call("quickbooks_invoice", last_r)

        elif last_r and any(kw in pl for kw in ["sap", "material master", "s4hana"]):
            with st.spinner("Updating SAP material master..."):
                do_tool_call("sap_update_material", last_r)

        elif last_r and any(kw in pl for kw in ["erp", "purchase order", "update po", "netsuite", "oracle"]):
            with st.spinner("Updating ERP purchase order..."):
                erp = "Oracle" if "oracle" in pl else ("NetSuite" if "netsuite" in pl else "SAP")
                do_tool_call("erp_update_po", last_r, erp_system=erp)

        # === Analytics tools ===
        elif last_r and any(kw in pl for kw in ["impact", "impact report", "executive", "analysis", "c-suite"]):
            with st.spinner("Generating tariff impact report..."):
                do_tool_call("tariff_impact_report", last_r)

        # === Refinement ===
        elif last_r and any(kw in pl for kw in ["refine", "actually it", "more detail", "it is also", "made of"]):
            refined = f"{last_r['query']}. Additional details: {prompt}"
            with st.spinner("Re-classifying..."):
                do_classify(refined, "")

        # === New classification (default) ===
        else:
            country = _extract_country(prompt)
            with st.spinner("Classifying..."):
                do_classify(prompt, country)

    st.rerun()
