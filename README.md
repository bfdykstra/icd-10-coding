# ICD-10 Coding Assistant

A web app that lets you upload a discharge summary and get a list of ICD-10 codes that are likely missing from it. The current approach uses an LLM to analyze the summary and propose additional codes. Results stream live to the UI.

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS 4, shadcn/ui (Radix primitives)
- **Backend**: FastAPI, Uvicorn, Pydantic v2, Server‑Sent Events (SSE)
- **ML/DS**: OpenAI (`openai` + `instructor` for structured outputs), pandas, numpy, scikit‑learn, PyArrow, (optional) transformers/xgboost, ChromaDB
- **Data**: MIMIC‑IV snapshot (+ synthetic discharge summaries generated for this project)

## What the tool does

- Accepts a discharge summary plus any ICD‑10 codes already assigned
- Uses an LLM to infer additional, potentially missing ICD‑10 codes
- Streams results back to the client via SSE for a responsive UX

## Background, data, and approach

- The MIMIC‑IV snapshot available to this project did not include discharge summaries.
- Synthetic discharge summaries were generated with GPT‑5‑mini using `server/notebooks/generate_discharge_summaries.ipynb`, conditioned on structured patient/hospitalization data and the assigned ICD‑10 codes.

I explored several approaches:

1. Medical NER → LLM
   - Extracted diseases/medications/anatomy with medical NER, then fed the entities + summary to an LLM to propose missing codes.
   - This underperformed. See `server/notebooks/evaluate_retrieval_strategies.ipynb` for experiments.
2. RAG over similar summaries
   - Retrieved nearest discharge summaries and their codes, then used the model to propose missing codes.
   - This also underperformed due to subtle clinical differences causing false positives/negatives.
   - See `server/scripts/validate_api.py` and results in `server/results/validation/api_validation_results_with_rag.json`.
3. Final approach (current)
   - Pure LLM prompting against the provided summary (and existing codes), streaming results.
   - Best observed results: **mean precision 36.0%**, **recall 22.3%**, **F1 26.1%**. See `server/results/validation/api_validation_results_gpt-4.1-mini_new_prompt.json`.

## Frontend details and performance

- **Problem**: Typing in the discharge summary textarea was extremely laggy when `value={dischargeSummary}` was set. Root cause wasn’t debounce; it was `ICD10MultiSelect` re‑rendering on every keystroke and processing ~2,467 codes. The Command search was also filtering all items per keystroke.
- **Fixes in `ICD10MultiSelect`**: `React.memo`, `useMemo` for a `Set` of selected codes, `useCallback` handlers, manual search with cap of 100 rendered items, lazy filtering only when open, disabled built‑in filtering (`shouldFilter={false}`), controlled search state, and a small “Showing X of Y” indicator.
- **Fixes in `ICD10Checker`**: Simplified to a single `dischargeSummary` state (removed debouncing) and re‑enabled the optimized multi‑select.
- **Result**: ~96% fewer rendered items (2,467 → 100 max), snappy typing, fast dropdown search, and file uploads correctly populate the textarea.
- See `client/components/icd10-multi-select.tsx` and `client/components/icd10-checker.tsx`.

## Key APIs

- Backend base URL: `http://localhost:8000` (configurable)
- Streaming endpoint (SSE): `POST /check-icd-codes/streaming`
  - Request body contains `discharge_summary` and `existing_codes` (code + description)
  - Frontend integration: `client/lib/api.ts`

## Getting started (local dev)

### Prerequisites

- Node.js 20+
- Python 3.11+
- OpenAI API key

### Backend

```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here  # or put it in a .env file
python run_server.py  # runs on 0.0.0.0:8000
```

Notes:

- CORS allows `http://localhost:3000` and `https://icd-10-coding.vercel.app` out of the box (see `server/main.py`).

### Frontend

```bash
cd client
npm install
# Configure the API URL for the client
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev  # http://localhost:3000
```

### Using the app

1. Open `http://localhost:3000`
2. Paste or upload a discharge summary
3. Add any existing codes via the multi‑select
4. Run the check and watch suggested codes stream in

## Repository structure (high‑level)

- `client/` — Next.js app, UI components, streaming client (`client/lib/api.ts`)
- `server/` — FastAPI app, utilities, notebooks, evaluation and validation scripts
  - `server/main.py` — API (SSE endpoint at `/check-icd-codes/streaming`)
  - `server/run_server.py` — Local dev server (Uvicorn, port 8000)
  - `server/utils/` — LLM, embeddings, vector store, NER helpers
  - `server/notebooks/` — Data prep, embeddings, experiments (see descriptions below)
  - `server/results/` — Evaluation/validation outputs

## Notebooks and scripts of interest

- `server/notebooks/generate_discharge_summaries.ipynb` — Creates synthetic discharge summaries used here
- `server/notebooks/evaluate_retrieval_strategies.ipynb` — NER/RAG exploration and retrieval experiments
- `server/notebooks/compile_validation_results.ipynb` — Collates validation/eval outputs for reporting
- `server/scripts/validate_api.py` — Runs API‑level validations; see JSON/CSV outputs in `server/results/validation/`

## Validation results

- Final (LLM‑only) prompt: `server/results/validation/api_validation_results_gpt-4.1-mini_new_prompt.json`
- RAG variant: `server/results/validation/api_validation_results_with_rag.json`
- Additional comparative plots can be found under `server/results/evaluation/` (e.g., F1@k charts)

## Configuration

- Backend model and OpenAI client config: `server/utils/llm_utils.py`, `server/utils/config.py`
- Client API base URL: `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`)

## Notes and caveats

- This is a research prototype. Reported metrics are dataset‑ and prompt‑dependent.
- RAG/NER components, embeddings, and vector stores are present but not used in the current best approach.

## License

TBD
