# Demo Plan — Agentica: Agentic CRM & Tariff Intelligence System

## Submission Requirements
- 3-minute video demo
- Repo link (shared with judges only)
- Due: 9 PM PST Sunday April 12

## Product Name: **Agentica**
## Tagline: "Agentic CRM for Global Trade"

---

## What We Actually Have (ground truth for scripting)

**Architecture:**
- Chat-based agentic UI (Streamlit + st.chat_message)
- CRM deal pipeline visualization (New → Classified → Costed → Reviewed → CRM Synced)
- 18 MCP-style tool calls across 5 categories (Documents, Communications, Regulatory, Integrations, Analytics)
- RAG over HTS tariff schedule + 283 CBP CROSS rulings (2,186 vector chunks, LanceDB)
- Hybrid search (BM25 + vector) with cross-encoder re-ranking
- Groq LLM (Llama 3.3 70B primary, 8B fallback) for classification and document generation
- Country-of-origin extraction from natural language
- CSV bulk upload for batch classification

**CRM Pipeline (the differentiator):**
1. User describes product → deal created → pipeline bar shows "New Deal"
2. Agent classifies → pipeline advances to "Classified"
3. "Run full pipeline" → auto-executes: Landed Cost → Invoice → CRM Sync
4. Pipeline bar shows all 4 stages completed with tool calls listed
5. Each tool call renders with MCP-style card (tool name, params, status, elapsed time)

**Tool categories visible in sidebar:**
- Documents: Generate Invoice, Landed Cost Calculator
- Communications: Supplier Letter, Surcharge Notice, Gmail, Slack, Teams
- Regulatory: Exemption Request, Export Controls Check
- Integrations: Zoho Books, Zoho CRM, Wave, SAP, QuickBooks, ERP PO Update
- Analytics: Tariff Impact Report, Batch Classify

**Reliable demo scenarios (tested, consistent results):**
1. Li-ion battery from China (8507.60, triggers Section 301 + IEEPA warnings)
2. Apple iPhone from China (8517.12, triggers guardrails + expert review)
3. Brake rotor (8708 heading, correct classification)

**LLM status:** 70B model quota resets daily. If exhausted, auto-falls back to 8B. Both handle demo scenarios.

---

## Video Structure (3:00 total)

### SCENE 1 — The Problem (0:00–0:20)

**[Screen: News headline about 2025-2026 tariff chaos]**

> "Every product crossing a US border needs a tariff classification code — picked from 17,000 line items. Get it wrong: fines up to 40% of shipment value."

> "But classification is just the start. Companies like Samsung and Microsoft need to calculate landed costs, update SAP, notify suppliers, file exemptions, and sync everything to their CRM — all before the shipment arrives."

---

### SCENE 2 — What It Does (0:20–0:40)

**[Screen: Show Agentica — clean chat UI with pipeline bar and sidebar showing 18 MCP tools]**

> "Agentica is an agentic CRM for global trade. It uses AI to classify products, then orchestrates a pipeline of 18 tool calls — CRM sync, invoicing, supplier negotiations, government filings — all from a single chat interface."

**[Point at sidebar tool grid]**

> "These are MCP tool integrations: Gmail, Slack, Teams for communications. Zoho, SAP, QuickBooks for enterprise systems. Each tool call is visible, auditable, and traceable."

---

### SCENE 3 — Demo: Full Pipeline (0:40–1:40) — THE KEY MOMENT

**[Screen: Type into chat]**

**Input:** `Lithium-ion battery pack 48V for electric bicycle from China`

**[Result appears: HTS 8507.60, tariff warnings, pipeline bar shows "Classified"]**

> "I describe a product. The agent creates a deal, classifies it using RAG over the official tariff schedule and 283 CBP rulings. It flags Section 301 duties and IEEPA surcharges."

**[Type: "run full pipeline"]**

> "Now I say 'run full pipeline.' Watch the deal progress."

**[Pipeline bar animates: Costed → Reviewed → CRM Synced. Tool call cards appear one by one]**

> "Stage 2: landed cost calculated — FOB value, freight, MFN duty, Section 301 additional, total landed cost per unit."

> "Stage 3: customs invoice generated with all line items and duty estimates."

> "Stage 4: deal synced to Zoho CRM with classification, costs, and compliance notes attached."

**[Point at pipeline bar showing all 5 stages completed]**

> "One chat message. Four tool calls. The deal went from raw product description to fully costed, invoiced, and CRM-synced in under 30 seconds."

---

### SCENE 4 — Tool Calling Demo (1:40–2:15)

**[Type: "draft supplier letter"]**

> "Now the agent drafts a cost-sharing letter to the supplier — referencing the specific HTS code, duty rate, and tariff warnings."

**[Tool call card appears, then the generated letter]**

**[Type: "send via gmail"]**

> "One more: send the classification report via Gmail."

**[Email tool call card appears with delivery confirmation]**

> "Every action is a discrete MCP tool call. Each one is logged, auditable, and can be connected to a real API in production."

---

### SCENE 5 — Enterprise Scale (2:15–2:40)

**[Show sidebar CSV upload — upload a file]**

> "For companies with thousands of SKUs, upload a CSV. The batch classifier processes the entire product list and returns a portfolio-level tariff analysis."

**[Show batch results table]**

> "Samsung imports 50,000 SKUs. A customs broker takes weeks. This takes minutes."

---

### SCENE 6 — Close (2:40–3:00)

**[Screen: Show the app with completed pipeline]**

> "Agentica: an agentic CRM that turns tariff classification into a complete trade compliance workflow. Classify, cost, review, sync — all orchestrated by AI with human-in-the-loop governance."

> "18 tool integrations. The HS system is international — the same codes work across 200 countries. We start with the US. Multi-country expansion is a data ingestion task, not a re-architecture."

> "Thank you."

---

## Recording Checklist

### Pre-record prep
- [ ] Restart Streamlit fresh: `pkill -f streamlit; cd agentica && source venv/bin/activate && export LD_LIBRARY_PATH="$HOME/lib:$LD_LIBRARY_PATH" && streamlit run src/ui/app.py --server.port 8501`
- [ ] Verify LLM is responsive: classify one test product first
- [ ] Have this input ready: `Lithium-ion battery pack 48V for electric bicycle from China`
- [ ] Clear chat (refresh page) before recording
- [ ] Test full pipeline flow before recording
- [ ] Check Groq quota

### Recording setup
- [ ] Screen record at 1080p
- [ ] Browser at ~90% zoom
- [ ] Hide bookmarks bar and personal tabs
- [ ] Disable OS notifications

### Timing targets
| Scene | Duration | Cumulative |
|-------|----------|------------|
| Problem | 20s | 0:20 |
| Solution overview | 20s | 0:40 |
| Full pipeline demo | 60s | 1:40 |
| Tool calling (supplier + email) | 35s | 2:15 |
| Enterprise scale (CSV) | 25s | 2:40 |
| Close | 20s | 3:00 |

### Demo flow (exact commands)
1. Type: `Lithium-ion battery pack 48V for electric bicycle from China`
2. Wait for classification result + pipeline bar
3. Type: `run full pipeline`
4. Wait for all 4 stages to complete (tool cards appear)
5. Type: `draft supplier cost-sharing letter`
6. Wait for letter generation
7. Type: `send report via gmail`
8. Wait for email confirmation
9. (If time) Upload sample CSV from sidebar

---

## Full Pipeline Script — Prompt and Expected Output

### Step 1: Classification

**Prompt:**
```
Lithium-ion battery pack 48V for electric bicycle from China
```

**Expected Output:**

```
┌─────────────────────────────────────────────┐
│ HTS Code: 8507.60.0090                      │
│ Description: Lithium-ion batteries,         │
│   including separators therefor...           │
│ Confidence: 90% (HIGH)                      │
│ Duty Rate (MFN): Free (USITC API)           │
│ Est. Effective Rate: 25%                    │
│   MFN 0% + additional 25% (Section 301)    │
└─────────────────────────────────────────────┘

⚠ Tariff Advisory: Country of origin: China.
  Section 301 (up to 25%) + IEEPA tariffs apply.
  The displayed duty rate is the Column 1 General (MFN) base rate only.

⚠ Tariff Advisory: Products in Chapter 85 (electrical/electronics)
  are commonly subject to additional duties.

⚠ Tariff Advisory: Base MFN rate is Free, but actual landed duty
  for China-origin goods will be substantially higher.

Expert Review Recommended:
  Additional tariffs likely apply for China-origin goods.

Pipeline bar: [New Deal] → [Classified ✓]
```

### Step 2: Full Pipeline

**Prompt:**
```
run full pipeline
```

**Expected Output (4 stages auto-execute):**

```
Stage 2/4 — Calculating landed cost...
┌─ Tool Call: landed_cost_calculator ─────────┐
│ Status: success | Elapsed: ~120ms           │
├─────────────────────────────────────────────┤
│ FOB Value (1,000 x $25.00)    $25,000.00   │
│ Ocean Freight                  $2,500.00    │
│ Insurance                        $350.00    │
│ CIF Value                     $27,850.00    │
│ MFN Duty (Free)                    $0.00    │
│ Merchandise Processing Fee       $96.39     │
│ Harbor Maintenance Tax            $3.48     │
│ Est. Section 301/IEEPA (25%)  $6,962.50    │
│ Total Duties & Fees           $7,062.37     │
│ Total Landed Cost            $34,912.37     │
│ Per-Unit Landed Cost             $34.91     │
│ Effective Total Duty Rate       25.4%       │
└─────────────────────────────────────────────┘
Pipeline bar: [Classified ✓] → [Costed ✓]

Stage 3/4 — Generating invoice and compliance review...
┌─ Tool Call: generate_invoice ───────────────┐
│ Status: success | Elapsed: ~80ms            │
├─────────────────────────────────────────────┤
│ Invoice No: INV-20260412-850760             │
│ HTS Code: 8507.60.0090                     │
│ Country of Origin: China                    │
│ Qty: 1,000 | Unit: $25.00 | Total: $25,000 │
│ MFN Duty: Free | MPF: $96.39 | HMT: $3.48 │
│ Estimated Landed Cost: $25,131.54           │
└─────────────────────────────────────────────┘
Pipeline bar: [Costed ✓] → [Reviewed ✓]

Stage 4/4 — Syncing to CRM...
┌─ Tool Call: zoho_create_deal ───────────────┐
│ Status: success | Elapsed: ~50ms            │
├─────────────────────────────────────────────┤
│ Deal: "Tariff Mitigation — 8507.60.0090"   │
│ Stage: Compliance Review                    │
│ HTS: 8507.60.0090 | Origin: China          │
│ Base Duty: Free | Warnings: 4              │
│ CRM: Zoho CRM                              │
└─────────────────────────────────────────────┘
Pipeline bar: [Reviewed ✓] → [CRM Synced ✓]

Pipeline complete. Deal DEAL-20260412-001 processed through 4 tool
calls: landed_cost_calculator → generate_invoice → zoho_create_deal
```

### Step 3: Supplier Letter

**Prompt:**
```
draft supplier cost-sharing letter
```

**Expected Output:**
```
┌─ Tool Call: draft_supplier_letter ──────────┐
│ Status: success | Elapsed: ~2.5s (LLM)     │
└─────────────────────────────────────────────┘

[Generated letter referencing HTS 8507.60.0090, Section 301
additional duties, and proposing a cost-sharing arrangement
for the 25% tariff surcharge on China-origin lithium-ion
battery packs. ~200 words.]
```

### Step 4: Email Delivery

**Prompt:**
```
send report via gmail
```

**Expected Output:**
```
┌─ Tool Call: send_email ─────────────────────┐
│ Status: success | Provider: gmail           │
│ To: compliance@company.com                  │
│ Subject: Tariff Classification — 8507.60    │
│ Body: Full classification report attached   │
└─────────────────────────────────────────────┘
```

### Fallback plan
- If LLM is slow (>15s): pre-record each segment separately, edit together
- If rate-limited: 8B fallback handles all scenarios
- Do NOT demo unreliable products (BT speaker, smart hub)

---

## Key Talking Points

| Judge | What resonates | When to say it |
|---|---|---|
| Samsung, Sonos | 50K+ SKU portfolio, batch processing, SAP integration | Enterprise scale scene |
| Ryder | Vehicle parts, supply chain costing | Full pipeline demo |
| JP Morgan, Capital One | Compliance risk, audit trail, governance | "Every tool call is logged and auditable" |
| eBay | Cross-border marketplace, per-listing classification | Batch classify scene |
| Oracle, Microsoft | CRM integration, ERP sync, enterprise workflow | "Sync to Zoho/SAP/Oracle with one command" |
| AI Startup judge | MCP tools, agentic workflow, RAG architecture | "18 MCP tool calls, not a chatbot" |

---

## Lines to Practice

**The hook:** "Classification is just the start. Companies need to cost, invoice, notify suppliers, file exemptions, and sync to their CRM — all before the shipment arrives."

**The differentiator:** "This is not a chatbot. It is an agentic CRM that orchestrates 18 tool calls across your enterprise stack — Gmail, Slack, SAP, Zoho, QuickBooks — from a single conversation."

**Pipeline moment:** "One chat message. Four tool calls. The deal went from raw product description to fully costed, invoiced, and CRM-synced."

**Enterprise scale:** "Samsung imports 50,000 SKUs. A customs broker takes weeks. This takes minutes."

**Close:** "The HS system is international. We start with the US. Multi-country is a data ingestion task, not a re-architecture."
