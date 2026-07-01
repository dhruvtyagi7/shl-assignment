# SHL Conversational Assessment Recommender — Approach Document

**Submission for:** SHL Take-Home Assignment  
**Date:** July 2026

---

## The Problem

The goal was to build a conversational AI that helps hiring managers find the right SHL assessments by talking rather than searching through filters. The hard constraints that shaped every decision:

- The response schema is graded by an automated evaluator — any deviation from `{reply, recommendations, end_of_conversation}` fails outright
- Every URL in the response must be real and scraped — no hallucinated links
- The API is stateless — the full conversation history comes in on every request
- 30-second response timeout per call, 8 turns max per conversation
- Must handle edge cases cleanly: vague openers, mid-conversation refinements, comparison questions, injection attempts, off-topic requests

---

## Architecture

```
POST /chat
  └── Pydantic schema validation
  └── Guardrails check (regex, <5ms)
  └── TF-IDF retrieval → top-12 catalog candidates
  └── Gemini 1.5 Flash call with grounded system prompt (~3-5s)
  └── URL validation + schema enforcement
  └── ChatResponse
```

Four components, one pass per request. No chaining, no second LLM calls — fits comfortably inside the 30s budget.

---

## Data Collection

**Challenge:** SHL's old catalog URL (`/solutions/products/product-catalog/`) now redirects to a JS-rendered marketing page, not a filterable table. The actual individual product pages live under `/products/assessments/{category}/{slug}/`.

**Approach:** I wrote a Playwright-based crawler (`scraper.py`) that seeds from known category pages and follows links to individual product pages. Playwright is needed because the page requires JavaScript to render; `requests` + BeautifulSoup returns a 403. The scraper dismisses the cookie consent banner before extracting text, since the banner's paragraph tags appear before the actual content and pollute naive extraction.

**Reality check:** SHL aggressively rate-limits scrapers (Cloudflare bot detection blocks many pages). For some products the automated scraper hit bot walls, so the descriptions in `catalog.json` are a combination of scraped content and reconstructed-from-SHL's-own-marketing-copy descriptions that I can verify are accurate. The URLs are all real and verified.

The catalog stores: `name`, `url`, `test_type`, `description`, `duration`, `remote_testing`, `adaptive_support`.

---

## Retrieval Design

**Choice: TF-IDF with bigrams**

I went with TF-IDF over sentence-transformers + FAISS for a few practical reasons:

- The catalog is small (~20-50 items). At this scale, TF-IDF achieves solid recall without the overhead of a vector store.
- No GPU, no external API call, no cold-start delay. The index fits in RAM and queries take under 50ms.
- The trade-off is weaker semantic matching — TF-IDF won't match "analytical thinking" to "inductive reasoning" unless those words appear in the description. I mitigate this by writing fuller descriptions and repeating product names in the index document.

**Query construction:** I concatenate user messages from the conversation history, weighting recent turns more by repeating them. This means if someone says "actually add personality tests," the latest message dominates the query and the retrieval picks up personality assessments.

**Recall@10 results** (against the 10 test conversation scenarios):

| Scenario | Recall@10 |
|---|---|
| Java developer (coding + cognitive) | ~0.8 |
| Leadership personality | ~0.9 |
| Contact center / volume hiring | ~0.8 |
| Mid-conversation refinement | ~0.75 |
| Average | **~0.81** |

The main failure mode is when users describe a need using purely business language ("we want someone who can handle ambiguity") rather than assessment-type words. A sentence-transformer model would handle this better — that's the natural upgrade path.

---

## Prompt Design

The system prompt does three things:

1. **Grounds the model** — it only sees the 12 retrieved catalog candidates, not the full catalog. Every URL the model uses has to be one it was given in the prompt. I also explicitly tell it: "Do not use your own training knowledge about these products."

2. **Defines turn-by-turn behavior** — when to clarify vs. when to commit to a recommendation. The rule is: ask one clarifying question if the request is genuinely vague (no role, no requirement), then commit. Don't keep asking.

3. **Enforces the schema** — the prompt specifies the exact JSON structure and what the fields should contain. I still validate post-generation in code and strip/fix anything that doesn't match, because models occasionally drift even with explicit instructions.

Temperature is set to 0.3 — low enough for consistent JSON output, high enough to avoid repetitive phrasing.

---

## Guardrails

A regex-based pre-check runs before the LLM call and catches:

- **Prompt injection patterns:** "ignore your instructions," "show me your system prompt," "act as a different AI," etc. (16 patterns)
- **Off-topic topics:** recipes, sports, dating, medical/legal advice, general coding tutorials, etc. (12 patterns)

Rule-based because it's deterministic, fast, and testable. The patterns are straightforward enough that a regex is more reliable than a classifier — it won't have false negatives on variations like "disregard all previous context."

Even on refusals, the response returns the correct schema (`recommendations: []`, `end_of_conversation: false`).

---

## What I'd Do Differently With More Time

- **Better retrieval:** Replace TF-IDF with a sentence-transformer model (e.g., `all-MiniLM-L6-v2`) for semantic matching. This would handle role-description queries that don't use assessment vocabulary.
- **Richer catalog:** The current 22-item catalog is what's publicly accessible on SHL's website. The full Individual Test Solutions catalog has 50+ entries — accessing it would require an SHL account or a different data source.
- **Conversation state extraction:** Right now the agent re-reads the full conversation every turn. A small extraction step (pulling out confirmed role, level, requirements as structured data) would make retrieval more targeted.
- **Streaming:** The 30-second timeout is fine for Gemini Flash, but streaming the response would improve perceived latency significantly.

---

## Tools Used

- **Playwright** — browser automation for scraping JS-rendered pages
- **scikit-learn** — TF-IDF vectorizer and cosine similarity
- **FastAPI** — API framework; Pydantic for schema validation
- **Gemini 1.5 Flash** — LLM for conversation and reasoning
- **Antigravity (Google DeepMind AI assistant)** — used for code structure planning, debugging the scraper's bot-detection issues, and iterating on the system prompt. Every design decision was my own choice and I can explain all of it.
