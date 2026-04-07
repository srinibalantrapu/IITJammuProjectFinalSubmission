"""Run destination -> (budget || hotel) -> itinerary to cut wall-clock vs pure sequential."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from crewai import Crew, Process
from crewai.crews.crew_output import CrewOutput

from agents import create_agents
from tasks_parallel import create_parallel_tasks
from utils import dump_model

log = logging.getLogger(__name__)

PHASE2_TIMEOUT_SECS = int(os.getenv("TRAVEL_PLANNER_PHASE_TIMEOUT", "300"))


class PipelineError(RuntimeError):
    """Raised when a pipeline phase fails so callers can distinguish agent errors."""


def _pydantic_to_json_str(task_output: object) -> str:
    p = getattr(task_output, "pydantic", None)
    if p is None:
        return "(none)"
    if hasattr(p, "model_dump"):
        return json.dumps(p.model_dump(), ensure_ascii=False, indent=2)
    return json.dumps(str(p))


def _crew_memory_flag(unified_memory: bool | None) -> bool:
    if unified_memory is None:
        return os.getenv("TRAVEL_PLANNER_UNIFIED_MEMORY", "").lower() in ("1", "true", "yes")
    return unified_memory


def _one_task_crew(agent: object, task: object, *, verbose: bool, unified_memory: bool) -> Crew:
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        memory=unified_memory,
        verbose=verbose,
    )


def _parse_stashed_plan(raw: str) -> dict[str, Any] | None:
    if not raw or raw.strip() in ("(none)", "{}", ""):
        return None
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or not d.get("destination"):
        return None
    return d


def _skip_destination_phase(merged: dict[str, str]) -> bool:
    if os.getenv("TRAVEL_PLANNER_SKIP_DEST_ON_REFINE", "1").lower() in ("0", "false", "no"):
        return False
    conv = (merged.get("conversation_context") or "").strip()
    if conv in ("(none)", ""):
        return False
    return _parse_stashed_plan(merged.get("stashed_prior_plan_json") or "") is not None


def run_parallel_pipeline(
    merged_inputs: dict[str, str],
    *,
    verbose: bool = False,
    unified_memory: bool | None = None,
) -> dict[str, Any]:
    """
    Phase 1: destination (skipped on refine when stashed plan + chat context exist).
    Phase 2: budget and hotel concurrently (two threads, separate agent instances).
    Phase 3: itinerary with all prior JSON injected.
    """
    use_mem = _crew_memory_flag(unified_memory)

    base = dict(merged_inputs)
    base.setdefault("prior_destination_json", "(none)")
    base.setdefault("prior_budget_json", "(none)")
    base.setdefault("prior_hotels_json", "(none)")
    base.setdefault("stashed_prior_plan_json", "(none)")

    dest_dict_snapshot: dict[str, Any] = {}
    memory_mode = "full"

    # --- Phase 1: destination (optional skip for faster follow-ups) ---
    if _skip_destination_phase(base):
        stashed = _parse_stashed_plan(base.get("stashed_prior_plan_json") or "")
        if stashed:
            dest_dict_snapshot = dump_model(stashed.get("destination"))
            base["prior_destination_json"] = json.dumps(dest_dict_snapshot, ensure_ascii=False, indent=2)
            memory_mode = "refine_reuse_destination"
    else:
        agents_1 = create_agents(verbose=verbose)
        tasks_1 = create_parallel_tasks(agents_1)
        c1 = _one_task_crew(
            agents_1["destination_researcher"],
            tasks_1["destination"],
            verbose=verbose,
            unified_memory=use_mem,
        )
        r1 = c1.kickoff(inputs=base)
        if not isinstance(r1, CrewOutput) or not r1.tasks_output:
            raise PipelineError("Destination phase produced no output")
        dest_json = _pydantic_to_json_str(r1.tasks_output[0])
        base["prior_destination_json"] = dest_json
        dest_dict_snapshot = dump_model(getattr(r1.tasks_output[0], "pydantic", None))

    # --- Phase 2: budget || hotel ---
    def _run_budget() -> CrewOutput:
        ag = create_agents(verbose=verbose)
        t = create_parallel_tasks(ag)
        cr = _one_task_crew(
            ag["budget_planner"],
            t["budget"],
            verbose=verbose,
            unified_memory=use_mem,
        )
        out = cr.kickoff(inputs=dict(base))
        if not isinstance(out, CrewOutput):
            raise TypeError("Budget crew: expected CrewOutput")
        return out

    def _run_hotel() -> CrewOutput:
        ag = create_agents(verbose=verbose)
        t = create_parallel_tasks(ag)
        cr = _one_task_crew(
            ag["hotel_finder"],
            t["hotel"],
            verbose=verbose,
            unified_memory=use_mem,
        )
        out = cr.kickoff(inputs=dict(base))
        if not isinstance(out, CrewOutput):
            raise TypeError("Hotel crew: expected CrewOutput")
        return out

    errors: list[str] = []
    r_budget: CrewOutput | None = None
    r_hotel: CrewOutput | None = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_budget = pool.submit(_run_budget)
        f_hotel = pool.submit(_run_hotel)

        try:
            r_budget = f_budget.result(timeout=PHASE2_TIMEOUT_SECS)
        except Exception:
            log.exception("Budget agent failed in phase 2")
            errors.append("Budget agent failed")

        try:
            r_hotel = f_hotel.result(timeout=PHASE2_TIMEOUT_SECS)
        except Exception:
            log.exception("Hotel agent failed in phase 2")
            errors.append("Hotel agent failed")

    if r_budget is None or r_hotel is None or errors:
        raise PipelineError(
            f"Phase 2 (budget || hotel) errors: {'; '.join(errors) or 'missing output'}"
        )
    if not r_budget.tasks_output or not r_hotel.tasks_output:
        raise PipelineError("Budget or hotel phase produced no task output")

    base["prior_budget_json"] = _pydantic_to_json_str(r_budget.tasks_output[0])
    base["prior_hotels_json"] = _pydantic_to_json_str(r_hotel.tasks_output[0])

    # --- Phase 3: itinerary ---
    agents_3 = create_agents(verbose=verbose)
    tasks_3 = create_parallel_tasks(agents_3)
    c3 = _one_task_crew(
        agents_3["itinerary_generator"],
        tasks_3["itinerary"],
        verbose=verbose,
        unified_memory=use_mem,
    )
    r3 = c3.kickoff(inputs=base)
    if not isinstance(r3, CrewOutput) or not r3.tasks_output:
        raise PipelineError("Itinerary phase produced no output")

    return {
        "destination": dest_dict_snapshot,
        "budget": dump_model(getattr(r_budget.tasks_output[0], "pydantic", None)),
        "hotels": dump_model(getattr(r_hotel.tasks_output[0], "pydantic", None)),
        "itinerary": dump_model(getattr(r3.tasks_output[0], "pydantic", None)),
        "raw_final": r3.raw,
        "pipeline_mode": "parallel",
        "session_memory_mode": memory_mode,
    }
