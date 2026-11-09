"""Shared policy + drive execution pipeline for the Runner and the GUI worker.

Both the headless Runner and the GUI worker must derive actions from the exact
same sequence: sense every step, step the DriveCore, then query the Policy. This
module holds that pipeline so the two execution paths cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from simlab.core.types import Action, Observation


def needs_observations(policy: Any, drive_core: Any) -> bool:
    """Whether any active component requires fresh observations each step.

    Components that keep state (a recurrent DriveCore, a Policy that reads obs
    and drive_state) set ``needs_observations = True`` (the default). Fast
    paths may only skip per-step sensing, drive updates and snapshots when no
    active component needs observations; stateful components must be stepped
    every step.
    """
    needs_policy = policy is not None and getattr(
        policy, "needs_observations", True
    )
    needs_drive = drive_core is not None and getattr(
        drive_core, "needs_observations", True
    )
    return needs_policy or needs_drive


def compute_actions(
    world: Any,
    policy: Any,
    drive_core: Any,
    step: int,
    *,
    batched: bool = False,
) -> tuple[dict[str, Action], dict[str, tuple[float, ...]], dict[str, Observation]]:
    """Compute per-agent actions for the current world state.

    Uses only the world's observations (rays + proprioception) plus the drive
    core and policy, never direct world internals for policy decisions. World
    state is consulted only by policies that explicitly opt into it via a
    ``select_action`` method (e.g. OracleHeuristicPolicy).

    Args:
        world: The World instance to sense.
        policy: A registered Policy instance (or None).
        drive_core: A registered DriveCore instance (or None).
        step: The current simulation step index.
        batched: If True and the policy exposes ``act_batch``, run one
            batched forward across all live agents instead of one per-agent
            call. The per-agent path (default) is the reference implementation
            and is used for evaluation/training reproducibility; the batched
            path is a latency optimization for GUI world-view workers.

    Returns:
        A tuple of ``(actions, drives, observations)``.
    """
    observations = world._observation_batch()
    live_ids = [
        agent_id
        for agent_id, state in world._agents.items()
        if state["is_alive"]
    ]
    values = []
    drives: dict[str, tuple[float, ...]] = {}
    if drive_core is not None:
        live_observations = [observations[agent_id] for agent_id in live_ids]
        values = drive_core.step(live_observations, {"rng": world._rng})
        drives = {
            agent_id: tuple(float(value) for value in values[index])
            for index, agent_id in enumerate(live_ids)
        }
    actions: dict[str, Action] = {}
    if policy is None:
        for agent_id in live_ids:
            actions[agent_id] = Action()
    elif hasattr(policy, "select_action"):
        for agent_id in live_ids:
            state = world._agents[agent_id]
            world_state = {
                "food_locations": [
                    patch["position"] for patch in world._food_patches
                ],
                "food_patches": [
                    (tuple(patch["position"]), patch["amount"])
                    for patch in world._food_patches
                ],
                "hazard_locations": [
                    tuple(patch["position"]) for patch in world._hazard_patches
                ],
            }
            actions[agent_id] = policy.select_action(
                state, world_state, [], step
            )
    elif batched and hasattr(policy, "act_batch"):
        # One batched forward for all live agents (single inference call).
        live_obs = [observations[agent_id] for agent_id in live_ids]
        live_drives = [drives.get(agent_id, ()) for agent_id in live_ids]
        batch = policy.act_batch(live_obs, live_drives)
        actions = dict(zip(live_ids, batch))
    else:
        for agent_id in live_ids:
            actions[agent_id] = policy.act(
                observations[agent_id], drives.get(agent_id, ())
            )
    return actions, drives, observations
