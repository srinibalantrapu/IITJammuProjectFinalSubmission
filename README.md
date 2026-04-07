# AI Travel Planner — CrewAI multi-agent assignment

Four collaborating agents implement the **AI Travel Planner** scenario: **Destination Researcher** → **Budget Planner** → **Hotel Finder** → **Itinerary Generator**, using **CrewAI** with **structured (Pydantic) outputs** per stage.

**Default runtime:** an **application-parallel** pipeline (destination, then **budget ∥ hotel** in parallel, then itinerary) — not a single sequential `Crew` for the whole flow. See **trade-off** below.

## Parallel vs sequential — explicit trade-off (speed vs coupling)

In the **default parallel** pipeline, **Budget Planner** and **Hotel Finder** start at the **same time** after destination completes. That means the **Hotel Finder never receives the Budget Planner’s JSON** (no `overall_band`, `stated_budget_interpretation`, or category breakdown from that agent). Hotels are chosen from the **user’s `budget_hint` / message**, **destination research JSON**, and session context only.

- **Why we accept this:** It removes one serial LLM wait in phase 2 and **reduces end-to-end latency**, which matters for interactive demos.
- **Downside:** Nightly “hints” and property choices may **drift slightly** from the formal budget band the Budget agent later outputs — they usually align if `budget_hint` is specific, but it is **not guaranteed** the way a fully sequential chain would be.
- **Mitigation:** The **Itinerary Generator** runs **after** both agents finish and sees **budget + hotel + destination** JSON, so the written plan can still be narratively consistent; for submissions that must show **hotel ↔ budget agent coupling**, run **`TRAVEL_PLANNER_PARALLEL=0`** or **`python src/main.py … --sequential`** (one Crew, task `context` chain — slower).

**Demo video tip:** State this trade-off in **one sentence** (e.g. *“We parallelize budget and hotel for speed; hotels use the form budget text, not the budget agent’s output, until the itinerary merges both.”*) so graders hear the design choice on purpose, not as a bug.

## What gets remembered

- **Session memory (Gradio):** Previous user/assistant turns are summarized into a `conversation_context` string injected into every task. On follow-ups, destination/duration are also **merged from the last JSON plan** when the form is left blank, and short messages like “make it low-cost” are copied into **budget hint** when relevant.
- **CrewAI unified memory:** Optional and **off in the Gradio app by default** (it adds embedding I/O latency). Enable with `TRAVEL_PLANNER_UNIFIED_MEMORY=true` in `.env`, or use `python src/main.py ... --unified-memory` for CLI demos.

**Clear memory:** In the Gradio UI, use **Clear memory (reset chat & outputs)** to wipe session state for demos.

## Speed tips

- **Parallel pipeline (default):** see **Parallel vs sequential — explicit trade-off** above (hotel does not see budget agent JSON in phase 2). Disable with `TRAVEL_PLANNER_PARALLEL=0` or CLI `--sequential` for full task-context ordering.
- Set **`TRAVEL_PLANNER_MODEL=gpt-4o-mini`** (see `.env.example`) so all four agents use a fast model.
- Keep **unified memory off** unless you need it for the assignment write-up (session memory still handles follow-ups).
- Expect on the order of **~1–2 minutes** with parallel mode (network and model dependent), longer with `--sequential`.

## Setup

```bash
cd "/path/to/IITJammu Project"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY (required). Optional: SERPER_API_KEY — used for destination **and** hotel name checks when set.
```

## Run

**CLI** (defaults build a minimal message if you only pass structured flags):

```bash
PYTHONPATH=src python src/main.py "5 days in Kyoto, temples and food" \
  --budget mid-range --style relaxed
```

Simulate prior context for testing:

```bash
PYTHONPATH=src python src/main.py "Now make it more budget-friendly" \
  --context "User previously asked for 5 days in Kyoto with mid-range budget."
```

**Gradio UI:**

```bash
python app_gradio.py
```

## Deliverables

| Deliverable | Where |
|-------------|--------|
| GitHub repo | This repository (initialize/push as needed) |
| Demo video (2–5 min) | Record CLI and/or Gradio; show follow-up memory; **briefly mention** parallel phase-2 trade-off (hotel vs budget agent context — see section above) |
| Architecture diagram | [docs/architecture.md](docs/architecture.md) (Mermaid) · [docs/architecture_diagram.html](docs/architecture_diagram.html) (visual) |
| Sample outputs | [outputs/samples/](outputs/samples/) — **live LLM runs** (see `_provenance` in each JSON) |
| Optional UI | [app_gradio.py](app_gradio.py) |

## Data disclaimer

Hotel and price suggestions are **illustrative** unless you add a live booking/pricing API. The **Hotel Finder** agent is instructed to always surface a disclaimer; verify everything before booking.

## Layout

```
src/agents.py             # Agent roles (env-var config cached)
src/tasks.py              # Sequential four-task chain + TRIP_CONTEXT
src/tasks_parallel.py     # Single-task prompts + JSON injection for parallel mode
src/schemas.py            # Pydantic output models (DestinationResearchOutput, etc.)
src/crew.py               # Sequential Crew assembly
src/parallel_pipeline.py  # Threaded phase-2 orchestration (budget || hotel)
src/memory.py             # Session conversation -> context string
src/followup_context.py   # Merge form slots from prior JSON; tag refinements
src/utils.py              # Shared helpers (dump_model, normalize_chat_history, TripInputs, TurnResult)
src/main.py               # CLI + run_pipeline router
app_gradio.py             # Gradio UI (JSON tab + readable tab + clear + status label)
tests/                    # Pytest suite (schemas, followup_context, memory, utils)
outputs/samples/          # Saved example runs
```

## License

Educational / coursework use.
