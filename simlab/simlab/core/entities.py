"""SimLab entities module - Agent, FoodPatch, Hazard."""

from __future__ import annotations

from dataclasses import dataclass, field

from simlab.core.types import Action, AgentState


@dataclass
class FoodPatch:
    """A patch of food in the world."""

    position: tuple[int, int]  # (x, y) position
    energy_value: float = 10.0  # Energy restored when eaten
    consumed: bool = False  # Whether the food has been consumed


@dataclass
class Hazard:
    """A hazard in the world that damages agents."""

    position: tuple[int, int]  # (x, y) position
    damage_per_contact: float = 1.0  # Damage per contact
    active: bool = True  # Whether the hazard is active


@dataclass
class Agent:
    """An agent in the simulation."""

    agent_id: str  # Unique identifier
    position: tuple[int, int]  # (x, y) position
    energy: float  # Current energy level
    max_energy: float  # Maximum energy capacity
    is_alive: bool = True  # Whether the agent is alive
    actions_this_step: list[Action] = field(
        default_factory=list
    )  # Actions taken this step

    def update_position(
        self, new_position: tuple[int, int], world_width: int, world_height: int
    ) -> None:
        """Update agent position with boundary checking.

        Args:
            new_position: New (x, y) position.
            world_width: Width of the world for boundary checking.
            world_height: Height of the world for boundary checking.
        """
        x, y = new_position
        x = max(0, min(x, world_width - 1))
        y = max(0, min(y, world_height - 1))
        self.position = (x, y)

    def deduct_energy(self, amount: float) -> None:
        """Deduct energy from the agent.

        Args:
            amount: Amount of energy to deduct.
        """
        self.energy = max(0.0, self.energy - amount)
        if self.energy <= 0:
            self.is_alive = False

    def gain_energy(self, amount: float, max_cap: float | None = None) -> None:
        """Gain energy for the agent.

        Args:
            amount: Amount of energy to add.
            max_cap: Maximum energy cap. If None, uses max_energy.
        """
        cap = max_cap if max_cap is not None else self.max_energy
        self.energy = min(cap, self.energy + amount)

    def to_agent_state(self) -> AgentState:
        """Convert to AgentState type.

        Returns:
            AgentState representation.
        """
        return AgentState(
            position=self.position,
            energy=self.energy,
            max_energy=self.max_energy,
            is_alive=self.is_alive,
            agent_id=self.agent_id,
        )

    @classmethod
    def from_agent_state(cls, state: AgentState, agent_id: str) -> Agent:
        """Create Agent from AgentState.

        Args:
            state: AgentState to convert from.
            agent_id: Agent ID.

        Returns:
            New Agent instance.
        """
        return cls(
            agent_id=agent_id,
            position=state.position,
            energy=state.energy,
            max_energy=state.max_energy,
            is_alive=state.is_alive,
        )


class EntityRegistry:
    """Registry for managing entities in the world."""

    def __init__(self) -> None:
        """Initialize an empty entity registry."""
        self._agents: dict[str, Agent] = {}
        self._food_patches: dict[tuple[int, int], FoodPatch] = {}
        self._hazards: dict[tuple[int, int], Hazard] = {}

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the registry.

        Args:
            agent: Agent to add.
        """
        self._agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry.

        Args:
            agent_id: ID of the agent to remove.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID.

        Args:
            agent_id: ID of the agent to get.

        Returns:
            Agent or None if not found.
        """
        return self._agents.get(agent_id)

    def add_food(self, food_patch: FoodPatch) -> None:
        """Add a food patch to the registry.

        Args:
            food_patch: Food patch to add.
        """
        pos = food_patch.position
        self._food_patches[pos] = food_patch

    def remove_food(self, position: tuple[int, int]) -> None:
        """Remove a food patch at the given position.

        Args:
            position: (x, y) position of the food to remove.
        """
        if position in self._food_patches:
            del self._food_patches[position]

    def get_food(self, position: tuple[int, int]) -> FoodPatch | None:
        """Get a food patch at the given position.

        Args:
            position: (x, y) position to check.

        Returns:
            FoodPatch or None if not found.
        """
        return self._food_patches.get(position)

    def add_hazard(self, hazard: Hazard) -> None:
        """Add a hazard to the registry.

        Args:
            hazard: Hazard to add.
        """
        pos = hazard.position
        self._hazards[pos] = hazard

    def remove_hazard(self, position: tuple[int, int]) -> None:
        """Remove a hazard at the given position.

        Args:
            position: (x, y) position of the hazard to remove.
        """
        if position in self._hazards:
            del self._hazards[position]

    def get_hazard(self, position: tuple[int, int]) -> Hazard | None:
        """Get a hazard at the given position.

        Args:
            position: (x, y) position to check.

        Returns:
            Hazard or None if not found.
        """
        return self._hazards.get(position)

    def get_all_agents(self) -> list[Agent]:
        """Get all agents in the registry.

        Returns:
            List of all agents.
        """
        return list(self._agents.values())

    def get_all_food_patches(self) -> list[FoodPatch]:
        """Get all food patches in the registry.

        Returns:
            List of all food patches.
        """
        return list(self._food_patches.values())

    def get_all_hazards(self) -> list[Hazard]:
        """Get all hazards in the registry.

        Returns:
            List of all hazards.
        """
        return list(self._hazards.values())

    def agent_ids(self) -> list[str]:
        """Get all agent IDs.

        Returns:
            List of agent IDs.
        """
        return list(self._agents.keys())

    def food_positions(self) -> list[tuple[int, int]]:
        """Get all food patch positions.

        Returns:
            List of (x, y) positions.
        """
        return [fp.position for fp in self._food_patches.values()]

    def hazard_positions(self) -> list[tuple[int, int]]:
        """Get all hazard positions.

        Returns:
            List of (x, y) positions.
        """
        return [h.position for h in self._hazards.values()]
