"""Assemble Crew: sequential process + optional CrewAI unified memory."""

from __future__ import annotations

import os

from crewai import Crew, Process

from agents import create_agents
from tasks import create_tasks


def _default_unified_memory() -> bool:
    """Unified memory adds embeddings + storage latency; opt-in via env."""
    return os.getenv("TRAVEL_PLANNER_UNIFIED_MEMORY", "").lower() in ("1", "true", "yes")


def build_crew(*, verbose: bool = False, unified_memory: bool | None = None) -> Crew:
    agents = create_agents(verbose=verbose)
    tasks = create_tasks(agents)

    ordered_agents = [
        agents["destination_researcher"],
        agents["budget_planner"],
        agents["hotel_finder"],
        agents["itinerary_generator"],
    ]

    use_mem = _default_unified_memory() if unified_memory is None else unified_memory

    return Crew(
        agents=ordered_agents,
        tasks=tasks,
        process=Process.sequential,
        memory=use_mem,
        verbose=verbose,
    )
