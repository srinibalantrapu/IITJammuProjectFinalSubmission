# AI Travel Planner

A multi-agent travel planning system built with **CrewAI**, featuring four collaborating AI agents and a **Gradio** web interface.

## Architecture

```
User Input → Destination Researcher → Budget Planner ─┐
                                      Hotel Finder   ─┤ (parallel)
                                                       └→ Itinerary Generator → Final Plan
```

Each agent produces **structured Pydantic output**, and the pipeline runs in parallel by default (budget and hotel execute simultaneously after destination research completes).

> To run sequentially instead: `TRAVEL_PLANNER_PARALLEL=0` or `--sequential` on CLI.

## Quick Start

```bash
# 1. Clone and set up
cd "IITJammu Project"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add OPENAI_API_KEY (required), SERPER_API_KEY (optional)
```

### Gradio UI

```bash
python app_gradio.py
```

### CLI

```bash
PYTHONPATH=src python src/main.py "5 days in Kyoto, temples and food" \
  --budget mid-range --style relaxed
```

Follow-up with prior context:

```bash
PYTHONPATH=src python src/main.py "Now make it more budget-friendly" \
  --context "User previously asked for 5 days in Kyoto with mid-range budget."
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `SERPER_API_KEY` | — | Optional. Enables web search for destinations and hotels |
| `TRAVEL_PLANNER_MODEL` | `gpt-4o-mini` | LLM model for all agents |
| `TRAVEL_PLANNER_PARALLEL` | `1` | Set to `0` for sequential pipeline |
| `TRAVEL_PLANNER_UNIFIED_MEMORY` | `false` | Enable CrewAI's built-in vector memory |
| `TRAVEL_PLANNER_DISABLE_SERPER` | `false` | Disable web search even if key is set |

## Session Memory

The Gradio UI maintains conversational context across turns:

- Chat history is summarized and injected into every agent task
- Follow-up messages reuse prior destination/duration when the form is left blank
- Short refinements like "make it cheaper" are routed to the appropriate agent
- Use **Start over** in the UI to reset all session state

This is separate from CrewAI's unified memory, which is off by default.

## Project Layout

```
app_gradio.py                 Gradio web interface
src/
  agents.py                   Agent definitions (cached config)
  schemas.py                  Pydantic output models
  tasks.py                    Sequential task chain
  tasks_parallel.py           Parallel-mode task prompts
  crew.py                     Sequential Crew assembly
  parallel_pipeline.py        Threaded parallel orchestration
  memory.py                   Session context builder
  followup_context.py         Follow-up slot merging and tagging
  utils.py                    Shared helpers and type definitions
  main.py                     CLI entry point
tests/                        Pytest suite
outputs/samples/              Example LLM-generated plans
docs/architecture.md          Architecture diagram (Mermaid)
```

## Sample Outputs

Pre-generated plans from live LLM runs are in [`outputs/samples/`](outputs/samples/):

- `sample_01_kyoto.json` — 5-day Kyoto temple trip
- `sample_02_lisbon_low_cost.json` — Budget Lisbon getaway
- `sample_03_family_barcelona.json` — Family trip to Barcelona

## License

Educational / coursework use — IIT Jammu.
