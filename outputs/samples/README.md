# Sample outputs (live pipeline runs)

These files were produced by **actual** `python src/main.py` executions (CrewAI + OpenAI), not hand-written placeholders. Each JSON includes a `_provenance` object (UTC timestamp, scenario label, pipeline mode).

**Capture settings:** `TRAVEL_PLANNER_DISABLE_SERPER=true` so samples are reproducible without Serper API calls; hotel names still require independent verification.

## Files

| File | Scenario |
|------|-----------|
| `sample_01_kyoto.json` | 5 days Kyoto, temples & food, ~USD 100/day mid-range |
| `sample_02_lisbon_low_cost.json` | 7 days Lisbon, backpacker ~EUR 45/day |
| `sample_03_family_barcelona.json` | 7 days Barcelona, family mid-range |
| `sample_03_family_barcelona.md` | Same run as `sample_03`…json, rendered for readers |

## Regenerate (requires `OPENAI_API_KEY` in `.env`)

```bash
cd "/path/to/IITJammu Project"
export PYTHONPATH=src
export TRAVEL_PLANNER_DISABLE_SERPER=true

python3 src/main.py "Plan 5 days in Kyoto focused on temples and casual food." \
  --destination "Kyoto, Japan" --dates "5 days" \
  --budget "Approximately USD 100 per day per person, mid-range" \
  --interests "temples, Gion, Nishiki Market" --style "relaxed pacing" \
  > outputs/samples/sample_01_kyoto.json
```

(Adjust prompts and flags for other destinations.)
