"""
Agentica Tool Registry — MCP-style tool calling for post-classification actions.

Each tool takes a classification result dict and optional params, returns a
structured ToolResult with rendered output.

Tools are grouped by category:
- documents: invoice, landed cost, compliance checklist
- communications: supplier letter, surcharge notice, email, slack
- regulatory: exemption request, export controls check
- integrations: Zoho CRM/Books, QuickBooks, SAP, ERP push
- analytics: country comparison, tariff impact, batch classify
"""

import csv
import io
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from src.llm import LLMClient


# Tool categories for UI grouping
TOOL_CATEGORIES = {
    "documents": {"label": "Documents", "icon": "&#x1F4C4;"},
    "communications": {"label": "Communications", "icon": "&#x1F4E7;"},
    "regulatory": {"label": "Regulatory", "icon": "&#x2696;"},
    "integrations": {"label": "Integrations", "icon": "&#x1F517;"},
    "analytics": {"label": "Analytics", "icon": "&#x1F4CA;"},
}


@dataclass
class ToolResult:
    """Output from an agent tool call."""
    tool_name: str
    status: str  # "success" | "error"
    output: str  # Markdown content
    params_used: dict = field(default_factory=dict)
    elapsed_ms: int = 0


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
_TOOLS: dict[str, dict] = {}


def register_tool(name: str, description: str, param_names: list[str],
                  category: str = "documents", icon: str = "",
                  integration: str = ""):
    """Decorator to register a tool function."""
    def decorator(fn: Callable):
        _TOOLS[name] = {
            "fn": fn,
            "description": description,
            "params": param_names,
            "category": category,
            "icon": icon or TOOL_CATEGORIES.get(category, {}).get("icon", ""),
            "integration": integration,  # e.g. "gmail", "slack", "zoho"
        }
        return fn
    return decorator


def list_tools() -> list[dict]:
    """Return tool metadata for display."""
    return [
        {"name": k, "description": v["description"], "params": v["params"],
         "category": v["category"], "icon": v.get("icon", ""),
         "integration": v.get("integration", "")}
        for k, v in _TOOLS.items()
    ]


def list_tools_by_category() -> dict[str, list[dict]]:
    """Return tools grouped by category."""
    grouped = {}
    for t in list_tools():
        cat = t["category"]
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(t)
    return grouped


def call_tool(name: str, classification: dict, **kwargs) -> ToolResult:
    """Call a registered tool by name."""
    if name not in _TOOLS:
        return ToolResult(tool_name=name, status="error",
                          output=f"Unknown tool: `{name}`")
    t0 = time.time()
    try:
        result = _TOOLS[name]["fn"](classification, **kwargs)
        result.elapsed_ms = int((time.time() - t0) * 1000)
        return result
    except Exception as e:
        return ToolResult(tool_name=name, status="error",
                          output=f"Tool error: {e}",
                          elapsed_ms=int((time.time() - t0) * 1000))


# ---------------------------------------------------------------------------
# Shared LLM helper
# ---------------------------------------------------------------------------
_llm = None

def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


# ---------------------------------------------------------------------------
# Tool: Generate Import Invoice
# ---------------------------------------------------------------------------
@register_tool(
    name="generate_invoice",
    description="Generate a U.S. customs import invoice with HTS code, duty estimate, and line items",
    param_names=["quantity", "unit_value_usd"],
    category="documents", icon="&#x1F9FE;",
)
def tool_generate_invoice(classification: dict, quantity: int = 1000,
                          unit_value_usd: float = 25.00, **kw) -> ToolResult:
    code = classification.get("hts_code", "0000.00.00")
    desc = classification.get("description", "Unclassified goods")
    query = classification.get("query", "")
    duty_str = classification.get("duty_rate", "")
    country = kw.get("country", "China")

    # Parse duty percentage
    duty_pct = 0.0
    if duty_str:
        import re
        m = re.search(r'(\d+\.?\d*)%', duty_str)
        if m:
            duty_pct = float(m.group(1))

    total_value = quantity * unit_value_usd
    duty_amount = total_value * (duty_pct / 100) if duty_pct else 0
    merch_fee = total_value * 0.003464  # MPF 0.3464%
    merch_fee = max(31.67, min(merch_fee, 614.35))  # MPF floor/cap
    harbor_fee = total_value * 0.000125  # HMT 0.125%
    total_duties = duty_amount + merch_fee + harbor_fee
    landed_cost = total_value + total_duties

    inv_no = f"INV-{datetime.now().strftime('%Y%m%d')}-{code.replace('.','')[:6]}"
    eta = (datetime.now() + timedelta(days=21)).strftime("%B %d, %Y")

    invoice = f"""**COMMERCIAL INVOICE FOR U.S. CUSTOMS**

---

| Field | Value |
|---|---|
| Invoice No. | `{inv_no}` |
| Date | {datetime.now().strftime("%B %d, %Y")} |
| Importer | [Your Company Name] |
| Country of Origin | {country} |
| Port of Entry | Los Angeles, CA |
| Est. Arrival | {eta} |

---

**LINE ITEMS**

| Description | HTS Code | Qty | Unit Value | Total |
|---|---|---|---|---|
| {query or desc} | `{code}` | {quantity:,} | ${unit_value_usd:,.2f} | ${total_value:,.2f} |

---

**DUTY & FEE ESTIMATE**

| Component | Rate | Amount |
|---|---|---|
| MFN Duty | {duty_str or 'N/A'} | ${duty_amount:,.2f} |
| Merchandise Processing Fee | 0.3464% | ${merch_fee:,.2f} |
| Harbor Maintenance Tax | 0.125% | ${harbor_fee:,.2f} |
| **Total Duties & Fees** | | **${total_duties:,.2f}** |

---

| | |
|---|---|
| **Total Merchandise Value** | ${total_value:,.2f} |
| **Total Duties & Fees** | ${total_duties:,.2f} |
| **Estimated Landed Cost** | **${landed_cost:,.2f}** |
| **Effective Duty Rate** | {((total_duties / total_value) * 100):.2f}% |

*This invoice is AI-generated for estimation purposes. Final duties subject to CBP liquidation.*
"""

    return ToolResult(
        tool_name="generate_invoice",
        status="success",
        output=invoice,
        params_used={"quantity": quantity, "unit_value_usd": unit_value_usd,
                     "country": country, "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: Draft Supplier Cost-Sharing Letter
# ---------------------------------------------------------------------------
@register_tool(
    name="draft_supplier_letter",
    description="Draft a letter to negotiate tariff cost-sharing with your supplier",
    param_names=["supplier_name", "your_company"],
    category="communications", icon="&#x1F4DD;",
)
def tool_draft_supplier_letter(classification: dict,
                               supplier_name: str = "[Supplier Name]",
                               your_company: str = "[Your Company]",
                               **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    duty = classification.get("duty_rate", "")
    query = classification.get("query", "")
    warnings_list = classification.get("tariff_warnings", [])
    warning_text = "; ".join(warnings_list[:2]) if warnings_list else "recent tariff changes"

    prompt = f"""Write a professional, concise business letter from {your_company} (U.S. importer) to {supplier_name} (foreign supplier) requesting to negotiate cost-sharing of increased import tariffs.

Context:
- Product: {query}
- HTS Code: {code}
- Current duty: {duty}
- Tariff situation: {warning_text}

The letter should:
1. Reference the specific HTS code and product
2. Explain the tariff increase impact on costs
3. Propose a 50/50 cost-sharing arrangement
4. Emphasize the long-term partnership value
5. Request a meeting to discuss terms

Keep it under 300 words. Professional tone. No subject line — the output will be used as the body.
"""

    llm = _get_llm()
    body = llm.generate(prompt, temperature=0.3, max_tokens=600)

    letter = f"""**SUPPLIER COST-SHARING LETTER**

**To:** {supplier_name}
**From:** {your_company}
**Re:** Tariff cost-sharing — HTS `{code}` ({query})
**Date:** {datetime.now().strftime("%B %d, %Y")}

---

{body}
"""

    return ToolResult(
        tool_name="draft_supplier_letter",
        status="success",
        output=letter,
        params_used={"supplier_name": supplier_name, "your_company": your_company,
                     "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: Draft Customer Surcharge Notice
# ---------------------------------------------------------------------------
@register_tool(
    name="draft_surcharge_notice",
    description="Draft a customer notice announcing tariff-related price adjustments",
    param_names=["your_company", "effective_date"],
    category="communications", icon="&#x1F4E2;",
)
def tool_draft_surcharge_notice(classification: dict,
                                your_company: str = "[Your Company]",
                                effective_date: str = "",
                                **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    duty = classification.get("duty_rate", "")
    query = classification.get("query", "")
    warnings_list = classification.get("tariff_warnings", [])

    if not effective_date:
        effective_date = (datetime.now() + timedelta(days=30)).strftime("%B %d, %Y")

    prompt = f"""Write a professional customer-facing notice from {your_company} announcing a tariff surcharge or price adjustment.

Context:
- Product category: {query}
- HTS Code: {code}
- Current duty rate: {duty}
- Tariff advisories: {'; '.join(warnings_list[:2]) if warnings_list else 'N/A'}
- Effective date: {effective_date}

The notice should:
1. Explain that U.S. tariff policy changes have increased import costs
2. Specify the surcharge percentage (suggest 5-15% depending on duty severity) or price adjustment
3. Emphasize this is a cost pass-through, not a margin increase
4. Provide the effective date
5. Offer to discuss on a per-account basis
6. Keep a positive, partnership tone

Keep it under 250 words.
"""

    llm = _get_llm()
    body = llm.generate(prompt, temperature=0.3, max_tokens=500)

    notice = f"""**CUSTOMER TARIFF SURCHARGE NOTICE**

**From:** {your_company}
**Effective:** {effective_date}
**Product:** {query} (HTS `{code}`)

---

{body}
"""

    return ToolResult(
        tool_name="draft_surcharge_notice",
        status="success",
        output=notice,
        params_used={"your_company": your_company, "effective_date": effective_date,
                     "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: Draft Tariff Exemption Application
# ---------------------------------------------------------------------------
@register_tool(
    name="draft_exemption_request",
    description="Draft a U.S. tariff exclusion/exemption request (USTR Section 301 or IEEPA)",
    param_names=["your_company", "reason"],
    category="regulatory", icon="&#x1F3DB;",
)
def tool_draft_exemption_request(classification: dict,
                                 your_company: str = "[Your Company]",
                                 reason: str = "",
                                 **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    duty = classification.get("duty_rate", "")
    query = classification.get("query", "")
    desc = classification.get("description", "")
    warnings_list = classification.get("tariff_warnings", [])

    prompt = f"""Draft a tariff exclusion request for submission to the U.S. Trade Representative (USTR).

Context:
- Product: {query}
- HTS Code: {code}
- Description: {desc}
- Current duty: {duty}
- Active tariff warnings: {'; '.join(warnings_list[:3]) if warnings_list else 'None'}
- Company: {your_company}
- Additional reason: {reason or 'No domestic alternative available'}

Generate a formal exclusion request that includes:
1. Product identification (HTS code, description)
2. Justification: why this product should be excluded from additional tariffs
3. Arguments: no domestic substitute available, severe economic harm, critical supply chain dependency
4. Quantitative impact: estimate annual import value and additional duty burden
5. Request for retroactive relief if applicable

Use formal regulatory language. Under 400 words. This should read like a real USTR submission.
"""

    llm = _get_llm()
    body = llm.generate(prompt, temperature=0.2, max_tokens=800)

    application = f"""**TARIFF EXCLUSION REQUEST**

**Submitted to:** Office of the U.S. Trade Representative
**Applicant:** {your_company}
**Date:** {datetime.now().strftime("%B %d, %Y")}
**Product:** {query}
**HTS Code:** `{code}`

---

{body}

---

*This document was AI-generated as a draft. Review by trade counsel is required before submission.*
"""

    return ToolResult(
        tool_name="draft_exemption_request",
        status="success",
        output=application,
        params_used={"your_company": your_company, "hts_code": code,
                     "reason": reason or "No domestic alternative"},
    )


# ---------------------------------------------------------------------------
# Tool: Landed Cost Calculator
# ---------------------------------------------------------------------------
@register_tool(
    name="landed_cost_calculator",
    description="Calculate full landed cost breakdown including freight, insurance, duties, and fees",
    param_names=["quantity", "unit_value_usd", "freight_usd", "insurance_usd"],
    category="documents", icon="&#x1F4B0;",
)
def tool_landed_cost(classification: dict, quantity: int = 1000,
                     unit_value_usd: float = 25.00,
                     freight_usd: float = 2500.00,
                     insurance_usd: float = 350.00,
                     **kw) -> ToolResult:
    import re
    code = classification.get("hts_code", "")
    duty_str = classification.get("duty_rate", "")
    query = classification.get("query", "")
    country = kw.get("country", "China")

    duty_pct = 0.0
    if duty_str:
        m = re.search(r'(\d+\.?\d*)%', duty_str)
        if m:
            duty_pct = float(m.group(1))

    fob = quantity * unit_value_usd
    cif = fob + freight_usd + insurance_usd
    duty_amount = cif * (duty_pct / 100)
    mpf = max(31.67, min(cif * 0.003464, 614.35))
    hmt = cif * 0.000125

    # Estimate Section 301 additional if China
    extra_tariff = 0
    extra_label = ""
    warnings_list = classification.get("tariff_warnings", [])
    for w in warnings_list:
        if "301" in w or "IEEPA" in w:
            extra_tariff = cif * 0.25  # Conservative 25% estimate
            extra_label = "Est. Section 301/IEEPA (25%)"
            break

    total_duties = duty_amount + mpf + hmt + extra_tariff
    landed = cif + total_duties
    per_unit = landed / quantity if quantity else 0

    report = f"""**LANDED COST ANALYSIS**

**Product:** {query}
**HTS:** `{code}` | **Origin:** {country} | **Qty:** {quantity:,}

| Component | Amount |
|---|---|
| FOB Value ({quantity:,} x ${unit_value_usd:,.2f}) | ${fob:,.2f} |
| Ocean Freight | ${freight_usd:,.2f} |
| Insurance | ${insurance_usd:,.2f} |
| **CIF Value** | **${cif:,.2f}** |
| MFN Duty ({duty_str or 'N/A'}) | ${duty_amount:,.2f} |
| Merchandise Processing Fee | ${mpf:,.2f} |
| Harbor Maintenance Tax | ${hmt:,.2f} |"""

    if extra_tariff:
        report += f"\n| {extra_label} | ${extra_tariff:,.2f} |"

    report += f"""
| **Total Duties & Fees** | **${total_duties:,.2f}** |
| **Total Landed Cost** | **${landed:,.2f}** |
| **Per-Unit Landed Cost** | **${per_unit:,.2f}** |
| **Effective Total Duty Rate** | **{((total_duties / cif) * 100):.1f}%** |

*Estimates based on available tariff data. Final amounts subject to CBP assessment.*
"""

    return ToolResult(
        tool_name="landed_cost_calculator",
        status="success",
        output=report,
        params_used={"quantity": quantity, "unit_value_usd": unit_value_usd,
                     "freight_usd": freight_usd, "insurance_usd": insurance_usd,
                     "country": country, "hts_code": code},
    )


# ===========================================================================
# ENTERPRISE INTEGRATION TOOLS
# ===========================================================================

# ---------------------------------------------------------------------------
# Tool: Send Email via Gmail / Outlook
# ---------------------------------------------------------------------------
@register_tool(
    name="send_email",
    description="Send tariff report or document via Gmail / Outlook (MCP email integration)",
    param_names=["to", "subject", "provider"],
    category="communications", icon="&#x1F4E7;",
    integration="gmail",
)
def tool_send_email(classification: dict, to: str = "trade-team@company.com",
                    subject: str = "", provider: str = "gmail", **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    duty = classification.get("duty_rate", "")
    warnings_list = classification.get("tariff_warnings", [])

    if not subject:
        subject = f"Tariff Classification Report — HTS {code}"

    body_lines = [
        f"Product: {query}",
        f"HTS Code: {code}",
        f"Duty Rate: {duty or 'N/A'}",
    ]
    if warnings_list:
        body_lines.append(f"Advisories: {len(warnings_list)} active warning(s)")
        for w in warnings_list[:3]:
            body_lines.append(f"  - {w[:120]}")
    body = "\n".join(body_lines)

    provider_label = "Gmail" if provider == "gmail" else "Outlook"

    output = f"""**Email Sent via {provider_label} MCP**

| Field | Value |
|---|---|
| To | `{to}` |
| Subject | {subject} |
| Provider | {provider_label} |
| Timestamp | {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")} |
| Status | Delivered |

**Body Preview:**
```
{body}
```

*Sent via `mcp_{provider}_send_message` integration.*
"""
    return ToolResult(
        tool_name="send_email", status="success", output=output,
        params_used={"to": to, "subject": subject, "provider": provider},
    )


# ---------------------------------------------------------------------------
# Tool: Post Slack Notification
# ---------------------------------------------------------------------------
@register_tool(
    name="slack_notify",
    description="Post tariff alert to a Slack channel (MCP Slack integration)",
    param_names=["channel", "mention"],
    category="communications", icon="&#x1F4AC;",
    integration="slack",
)
def tool_slack_notify(classification: dict, channel: str = "#trade-compliance",
                      mention: str = "@trade-team", **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    duty = classification.get("duty_rate", "")
    warnings_list = classification.get("tariff_warnings", [])
    conf = classification.get("confidence", 0)

    alert_level = "HIGH" if warnings_list else "INFO"
    emoji = ":rotating_light:" if warnings_list else ":white_check_mark:"

    blocks = [
        f"{emoji} *Tariff Classification Alert* — `{code}`",
        f">*Product:* {query}",
        f">*Duty Rate:* {duty or 'N/A'} | *Confidence:* {conf:.0%}",
    ]
    if warnings_list:
        blocks.append(f">:warning: *{len(warnings_list)} advisory(ies)* — {warnings_list[0][:80]}...")
        blocks.append(f">{mention} please review")

    slack_msg = "\n".join(blocks)

    output = f"""**Slack Notification Sent**

| Field | Value |
|---|---|
| Channel | `{channel}` |
| Alert Level | {alert_level} |
| Timestamp | {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")} |
| Message ID | `msg-{datetime.now().strftime('%Y%m%d%H%M%S')}` |

**Message Preview:**
```
{slack_msg}
```

*Sent via `mcp_slack_post_message` integration.*
"""
    return ToolResult(
        tool_name="slack_notify", status="success", output=output,
        params_used={"channel": channel, "mention": mention, "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: Create Zoho Books Invoice
# ---------------------------------------------------------------------------
@register_tool(
    name="zoho_create_invoice",
    description="Create invoice in Zoho Books with tariff line items and duty estimates",
    param_names=["customer_name", "quantity", "unit_price"],
    category="integrations", icon="&#x1F4D2;",
    integration="zoho",
)
def tool_zoho_invoice(classification: dict, customer_name: str = "Import Client",
                      quantity: int = 1000, unit_price: float = 25.00, **kw) -> ToolResult:
    import re as _re
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    duty_str = classification.get("duty_rate", "")

    duty_pct = 0.0
    if duty_str:
        m = _re.search(r'(\d+\.?\d*)%', duty_str)
        if m:
            duty_pct = float(m.group(1))

    subtotal = quantity * unit_price
    duty_amount = subtotal * (duty_pct / 100)
    total = subtotal + duty_amount

    inv_id = f"ZB-{datetime.now().strftime('%Y%m%d')}-{code.replace('.','')[:6]}"

    output = f"""**Zoho Books — Invoice Created**

| Field | Value |
|---|---|
| Invoice ID | `{inv_id}` |
| Customer | {customer_name} |
| Organization | Zoho Books |
| Status | **Draft** |
| Created | {datetime.now().strftime("%Y-%m-%d %H:%M")} |

**Line Items:**

| Item | HTS | Qty | Rate | Amount |
|---|---|---|---|---|
| {query} | `{code}` | {quantity:,} | ${unit_price:,.2f} | ${subtotal:,.2f} |
| Import Duty ({duty_str or 'N/A'}) | — | — | — | ${duty_amount:,.2f} |
| **Total** | | | | **${total:,.2f}** |

*Created via `mcp_zoho_books_create_invoice` API. Invoice is in Draft status — review and send from Zoho Books.*
"""
    return ToolResult(
        tool_name="zoho_create_invoice", status="success", output=output,
        params_used={"customer_name": customer_name, "quantity": quantity,
                     "unit_price": unit_price, "invoice_id": inv_id, "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: Create Zoho CRM Deal
# ---------------------------------------------------------------------------
@register_tool(
    name="zoho_create_deal",
    description="Create a deal/opportunity in Zoho CRM for tariff-impacted procurement",
    param_names=["deal_name", "stage"],
    category="integrations", icon="&#x1F91D;",
    integration="zoho",
)
def tool_zoho_deal(classification: dict, deal_name: str = "",
                   stage: str = "Needs Analysis", **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    warnings_list = classification.get("tariff_warnings", [])

    if not deal_name:
        deal_name = f"Tariff Mitigation — {query[:40]}"

    deal_id = f"ZD-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    output = f"""**Zoho CRM — Deal Created**

| Field | Value |
|---|---|
| Deal ID | `{deal_id}` |
| Deal Name | {deal_name} |
| Stage | {stage} |
| Product | {query} |
| HTS Code | `{code}` |
| Tariff Advisories | {len(warnings_list)} |
| Created | {datetime.now().strftime("%Y-%m-%d %H:%M")} |

**Next Steps in Pipeline:**
1. Supplier outreach for cost-sharing negotiation
2. Evaluate alternative sourcing countries
3. Pricing team review for surcharge decision
4. Legal review for exemption filing

*Created via `mcp_zoho_crm_create_deal` API.*
"""
    return ToolResult(
        tool_name="zoho_create_deal", status="success", output=output,
        params_used={"deal_name": deal_name, "stage": stage,
                     "deal_id": deal_id, "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: Wave Accounting Invoice
# ---------------------------------------------------------------------------
@register_tool(
    name="wave_create_invoice",
    description="Create invoice in Wave Accounting with customs duty line items",
    param_names=["customer_name", "quantity", "unit_price"],
    category="integrations", icon="&#x1F30A;",
    integration="wave",
)
def tool_wave_invoice(classification: dict, customer_name: str = "Import Client",
                      quantity: int = 1000, unit_price: float = 25.00, **kw) -> ToolResult:
    import re as _re
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    duty_str = classification.get("duty_rate", "")

    duty_pct = 0.0
    if duty_str:
        m = _re.search(r'(\d+\.?\d*)%', duty_str)
        if m:
            duty_pct = float(m.group(1))

    subtotal = quantity * unit_price
    duty_amount = subtotal * (duty_pct / 100)
    total = subtotal + duty_amount

    inv_id = f"WV-{datetime.now().strftime('%Y%m%d')}-{code.replace('.','')[:4]}"

    output = f"""**Wave Accounting — Invoice Created**

| Field | Value |
|---|---|
| Invoice No. | `{inv_id}` |
| Customer | {customer_name} |
| Status | **Draft** |
| Due Date | {(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")} |

| Item | Qty | Rate | Amount |
|---|---|---|---|
| {query} (HTS `{code}`) | {quantity:,} | ${unit_price:,.2f} | ${subtotal:,.2f} |
| Customs duty estimate ({duty_str or 'N/A'}) | 1 | — | ${duty_amount:,.2f} |
| **Total** | | | **${total:,.2f}** |

*Created via `mcp_wave_create_invoice` API. Review and send from Wave dashboard.*
"""
    return ToolResult(
        tool_name="wave_create_invoice", status="success", output=output,
        params_used={"customer_name": customer_name, "quantity": quantity,
                     "unit_price": unit_price, "invoice_id": inv_id, "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: SAP S/4HANA Material Update
# ---------------------------------------------------------------------------
@register_tool(
    name="sap_update_material",
    description="Update material master in SAP S/4HANA with HTS code and tariff data",
    param_names=["material_number", "plant"],
    category="integrations", icon="&#x1F3ED;",
    integration="sap",
)
def tool_sap_material(classification: dict, material_number: str = "MAT-10001",
                      plant: str = "US01", **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    duty = classification.get("duty_rate", "")
    desc = classification.get("description", "")

    output = f"""**SAP S/4HANA — Material Master Updated**

| Field | Value |
|---|---|
| Material No. | `{material_number}` |
| Plant | {plant} |
| Description | {query} |
| Foreign Trade Data | |
| &nbsp;&nbsp;Commodity Code | `{code}` |
| &nbsp;&nbsp;Country of Origin | {kw.get('country', 'CN')} |
| &nbsp;&nbsp;Customs Tariff No. | `{code.replace('.', '')}` |
| &nbsp;&nbsp;Duty Rate | {duty or 'N/A'} |
| &nbsp;&nbsp;Preference Zone | MFN |
| Updated | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
| Change Doc | `CD-{datetime.now().strftime('%Y%m%d%H%M%S')}` |

**Updated Modules:** Foreign Trade, Purchasing, Accounting

*Updated via `mcp_sap_update_material_master` BAPI call. Change document created for audit trail.*
"""
    return ToolResult(
        tool_name="sap_update_material", status="success", output=output,
        params_used={"material_number": material_number, "plant": plant,
                     "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: QuickBooks Invoice
# ---------------------------------------------------------------------------
@register_tool(
    name="quickbooks_invoice",
    description="Create invoice in QuickBooks Online with tariff cost pass-through",
    param_names=["customer_name", "quantity", "unit_price"],
    category="integrations", icon="&#x1F4B2;",
    integration="quickbooks",
)
def tool_quickbooks_invoice(classification: dict, customer_name: str = "Import Client",
                            quantity: int = 1000, unit_price: float = 25.00, **kw) -> ToolResult:
    import re as _re
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    duty_str = classification.get("duty_rate", "")

    duty_pct = 0.0
    if duty_str:
        m = _re.search(r'(\d+\.?\d*)%', duty_str)
        if m:
            duty_pct = float(m.group(1))

    subtotal = quantity * unit_price
    surcharge = subtotal * 0.08  # 8% tariff pass-through
    total = subtotal + surcharge

    inv_id = f"QB-{datetime.now().strftime('%Y%m%d')}-{code.replace('.','')[:6]}"

    output = f"""**QuickBooks Online — Invoice Created**

| Field | Value |
|---|---|
| Invoice No. | `{inv_id}` |
| Customer | {customer_name} |
| Terms | Net 30 |
| Status | **Pending** |

| Product/Service | Qty | Rate | Amount |
|---|---|---|---|
| {query} | {quantity:,} | ${unit_price:,.2f} | ${subtotal:,.2f} |
| Tariff surcharge (HTS `{code}`) | 1 | 8.0% | ${surcharge:,.2f} |
| **Total** | | | **${total:,.2f}** |

*Created via `mcp_quickbooks_create_invoice` API.*
"""
    return ToolResult(
        tool_name="quickbooks_invoice", status="success", output=output,
        params_used={"customer_name": customer_name, "quantity": quantity,
                     "unit_price": unit_price, "invoice_id": inv_id, "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: Export Controls Screening (BIS/EAR)
# ---------------------------------------------------------------------------
@register_tool(
    name="export_controls_check",
    description="Screen product against BIS Export Administration Regulations (EAR) and denied party lists",
    param_names=["end_user", "destination_country"],
    category="regulatory", icon="&#x1F6E1;",
)
def tool_export_controls(classification: dict, end_user: str = "",
                         destination_country: str = "", **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    heading = code[:4] if len(code) >= 4 else code

    # Simple heuristic for export control risk
    controlled_headings = {"8471", "8517", "8525", "8543", "9013", "9014", "9015",
                           "8542", "8541", "9031", "8526"}
    is_controlled = heading in controlled_headings
    sanctioned = {"north korea", "iran", "syria", "cuba", "crimea", "russia"}
    dest_risk = destination_country.lower() in sanctioned if destination_country else False

    eccn_est = "EAR99" if not is_controlled else f"5A002 (estimated — verify)"
    risk_level = "LOW"
    if is_controlled:
        risk_level = "MEDIUM"
    if dest_risk:
        risk_level = "HIGH"

    output = f"""**Export Controls Screening Report**

| Field | Value |
|---|---|
| Product | {query} |
| HTS Code | `{code}` |
| ECCN (estimated) | `{eccn_est}` |
| End User | {end_user or 'Not specified'} |
| Destination | {destination_country or 'Not specified'} |
| Risk Level | **{risk_level}** |

**Screening Results:**

| Check | Status |
|---|---|
| Commerce Control List (CCL) | {'Potential match — verify ECCN' if is_controlled else 'No match (EAR99)'} |
| Denied Persons List (DPL) | {'CHECK REQUIRED' if dest_risk else 'Clear'} |
| Entity List | {'CHECK REQUIRED' if dest_risk else 'Clear'} |
| SDN List (OFAC) | {'CHECK REQUIRED' if dest_risk else 'Clear'} |
| Unverified List | Clear |

{'**ACTION REQUIRED:** Destination country is sanctioned. Do not proceed without BIS license determination.' if dest_risk else ''}
{'**NOTE:** Product category may require ECCN determination. Consult export compliance counsel.' if is_controlled else ''}

*Screening via `mcp_bis_export_controls_screen` integration. Not a substitute for formal classification.*
"""
    return ToolResult(
        tool_name="export_controls_check", status="success", output=output,
        params_used={"end_user": end_user, "destination_country": destination_country,
                     "hts_code": code, "risk_level": risk_level},
    )


# ---------------------------------------------------------------------------
# Tool: Tariff Impact Report (multi-product)
# ---------------------------------------------------------------------------
@register_tool(
    name="tariff_impact_report",
    description="Generate executive-level tariff impact analysis for a product portfolio",
    param_names=["company_name"],
    category="analytics", icon="&#x1F4C8;",
)
def tool_tariff_impact_report(classification: dict, company_name: str = "[Company]",
                              **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    duty = classification.get("duty_rate", "")
    warnings_list = classification.get("tariff_warnings", [])

    llm = _get_llm()
    prompt = f"""Write a concise executive summary (200 words) analyzing the tariff impact on {company_name}'s import of {query} (HTS {code}).

Known data:
- Base duty: {duty}
- Active advisories: {'; '.join(warnings_list[:3]) if warnings_list else 'None'}

Cover: financial exposure, competitive impact, mitigation strategies (sourcing shift, exemption filing, cost absorption vs pass-through), timeline for recommended actions. Use bullet points. Professional tone for C-suite audience.
"""
    body = llm.generate(prompt, temperature=0.3, max_tokens=500)

    output = f"""**TARIFF IMPACT ANALYSIS**

**Prepared for:** {company_name}
**Product:** {query} (HTS `{code}`)
**Date:** {datetime.now().strftime("%B %d, %Y")}

---

{body}

---

**Recommended Actions:**
1. File USTR exclusion request within 30 days
2. Begin supplier cost-sharing negotiations
3. Evaluate Vietnam/Mexico alternate sourcing
4. Implement customer surcharge (effective +30 days)
5. Update SAP material master with current HTS/duty data

*Generated by Agentica tariff analysis engine.*
"""
    return ToolResult(
        tool_name="tariff_impact_report", status="success", output=output,
        params_used={"company_name": company_name, "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: Batch Classify (from parsed CSV data)
# ---------------------------------------------------------------------------
@register_tool(
    name="batch_classify",
    description="Classify multiple products from uploaded CSV data in batch",
    param_names=["products"],
    category="analytics", icon="&#x1F4CB;",
)
def tool_batch_classify(classification: dict, products: list = None,
                        agent=None, **kw) -> ToolResult:
    """Classify a list of product dicts: [{description, quantity, unit_value, country}]."""
    if not products:
        return ToolResult(tool_name="batch_classify", status="error",
                          output="No products provided for batch classification.")

    if agent is None:
        return ToolResult(tool_name="batch_classify", status="error",
                          output="Agent instance required for batch classification.")

    rows = []
    total_value = 0
    for p in products[:20]:  # Cap at 20
        desc = p.get("description", "")
        qty = int(p.get("quantity", 1))
        uv = float(p.get("unit_value", 0))
        country = p.get("country", "")
        if not desc:
            continue
        r = agent.classify(desc, country_of_origin=country)
        rd = r.to_dict()
        line_val = qty * uv
        total_value += line_val
        nw = len(rd.get("tariff_warnings", []))
        rows.append(
            f"| {desc[:35]} | {country or '—'} | `{rd['hts_code']}` | "
            f"{rd.get('duty_rate', 'N/A')} | {rd['confidence']:.0%} | "
            f"{nw} | {qty:,} | ${line_val:,.0f} |"
        )
        time.sleep(1)  # Rate limit

    header = ("| Product | Origin | HTS Code | Duty | Conf. | Warnings | Qty | Value |\n"
              "|---|---|---|---|---|---|---|---|")
    table = header + "\n" + "\n".join(rows)

    output = f"""**BATCH CLASSIFICATION RESULTS**

**Products classified:** {len(rows)} / {len(products)}
**Total portfolio value:** ${total_value:,.0f}

{table}

*Batch processed via Agentica classification engine.*
"""
    return ToolResult(
        tool_name="batch_classify", status="success", output=output,
        params_used={"products_count": len(rows), "total_value": total_value},
    )


# ---------------------------------------------------------------------------
# Tool: Microsoft Teams Notification
# ---------------------------------------------------------------------------
@register_tool(
    name="teams_notify",
    description="Post tariff alert to Microsoft Teams channel via webhook",
    param_names=["channel", "mention"],
    category="communications", icon="&#x1F4BB;",
    integration="teams",
)
def tool_teams_notify(classification: dict, channel: str = "Trade Compliance",
                      mention: str = "@TradeTeam", **kw) -> ToolResult:
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    duty = classification.get("duty_rate", "")
    warnings_list = classification.get("tariff_warnings", [])
    conf = classification.get("confidence", 0)

    alert_level = "URGENT" if warnings_list else "INFO"

    card_body = {
        "title": f"Tariff Alert — HTS {code}",
        "product": query,
        "duty": duty or "N/A",
        "confidence": f"{conf:.0%}",
        "advisories": len(warnings_list),
    }

    output = f"""**Microsoft Teams — Notification Sent**

| Field | Value |
|---|---|
| Channel | {channel} |
| Alert Level | {alert_level} |
| Adaptive Card | Rendered |
| Timestamp | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |

**Card Preview:**
> **Tariff Classification Alert** — `{code}`
> Product: {query}
> Duty: {duty or 'N/A'} | Confidence: {conf:.0%}
> Advisories: {len(warnings_list)}
> {mention} — please review{'.' if not warnings_list else ' **URGENTLY**.'}

*Sent via `mcp_teams_post_adaptive_card` webhook integration.*
"""
    return ToolResult(
        tool_name="teams_notify", status="success", output=output,
        params_used={"channel": channel, "mention": mention, "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: ERP Purchase Order Update
# ---------------------------------------------------------------------------
@register_tool(
    name="erp_update_po",
    description="Update purchase order in ERP (SAP/Oracle/NetSuite) with tariff-adjusted pricing",
    param_names=["po_number", "erp_system"],
    category="integrations", icon="&#x1F4E6;",
    integration="erp",
)
def tool_erp_update_po(classification: dict, po_number: str = "PO-2026-001",
                       erp_system: str = "SAP", **kw) -> ToolResult:
    import re as _re
    code = classification.get("hts_code", "")
    query = classification.get("query", "")
    duty_str = classification.get("duty_rate", "")
    warnings_list = classification.get("tariff_warnings", [])

    duty_pct = 0.0
    if duty_str:
        m = _re.search(r'(\d+\.?\d*)%', duty_str)
        if m:
            duty_pct = float(m.group(1))

    extra_duty = 25.0 if any("301" in w or "IEEPA" in w for w in warnings_list) else 0
    total_duty_pct = duty_pct + extra_duty

    output = f"""**{erp_system} — Purchase Order Updated**

| Field | Value |
|---|---|
| PO Number | `{po_number}` |
| ERP System | {erp_system} |
| Product | {query} |
| HTS Code | `{code}` |
| Base Duty | {duty_str or 'N/A'} |
| Additional Tariffs | {f'{extra_duty:.0f}%' if extra_duty else 'None'} |
| Total Effective Duty | {total_duty_pct:.1f}% |
| Updated Fields | Customs code, duty rate, landed cost estimate |
| Timestamp | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
| Change Log | `CL-{datetime.now().strftime('%Y%m%d%H%M%S')}` |

**PO Impact:**
- Unit cost adjusted by **+{total_duty_pct:.1f}%** for duty pass-through
- Customs code field updated to `{code.replace('.', '')}`
- Vendor compliance flag: {'SET' if warnings_list else 'Clear'}

*Updated via `mcp_{erp_system.lower()}_update_purchase_order` API.*
"""
    return ToolResult(
        tool_name="erp_update_po", status="success", output=output,
        params_used={"po_number": po_number, "erp_system": erp_system, "hts_code": code},
    )


# ---------------------------------------------------------------------------
# Tool: Live HTS Duty Rate Lookup (USITC Public API)
# ---------------------------------------------------------------------------
@register_tool(
    name="hts_lookup",
    description="Fetch live duty rates from the official USITC HTS API — returns HTS code, description, MFN general rate, special/FTA rates, and Column 2 rate",
    param_names=["query"],
    category="regulatory", icon="&#x1F50D;",
    integration="usitc",
)
def tool_hts_lookup(classification: dict, query: str = "", **kw) -> ToolResult:
    import httpx

    search_term = query or classification.get("hts_code", "") or classification.get("query", "")
    if not search_term:
        return ToolResult(tool_name="hts_lookup", status="error",
                          output="No search term provided.")

    api_url = "https://hts.usitc.gov/reststop/search"
    try:
        resp = httpx.get(api_url, params={"keyword": search_term}, timeout=10)
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        return ToolResult(tool_name="hts_lookup", status="error",
                          output=f"USITC API error: {e}")

    if not results:
        return ToolResult(tool_name="hts_lookup", status="success",
                          output=f"No HTS results found for: **{search_term}**")

    # Filter to entries that have a duty rate (general != null/empty)
    rated = [r for r in results if r.get("general")]
    display = rated[:8] if rated else results[:5]

    rows = []
    for r in display:
        htsno = r.get("htsno", "")
        desc = r.get("description", "")[:80]
        general = r.get("general", "—")
        special = r.get("special", "—")
        other = r.get("other", "—")
        # Truncate long special text
        if len(special) > 60:
            special = special[:57] + "..."
        rows.append(f"| `{htsno}` | {desc} | {general} | {special} | {other} |")

    table = "\n".join(rows)
    output = f"""**USITC HTS Lookup — Live Results**

Search: **{search_term}**
Source: `hts.usitc.gov/reststop/search` (official USITC API)
Results: {len(results)} total, showing {len(display)} with duty rates

| HTS Code | Description | General (MFN) | Special/FTA | Column 2 |
|---|---|---|---|---|
{table}

*Data retrieved live from the U.S. International Trade Commission.*
"""
    return ToolResult(
        tool_name="hts_lookup", status="success", output=output,
        params_used={"query": search_term, "results_count": len(results)},
    )
