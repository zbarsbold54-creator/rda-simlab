"""SimLab config module - load TOML into dataclasses (tomllib)."""

import os
from dataclasses import dataclass, field
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class WorldConfig:
    """Configuration for the simulation world."""

    width: int = 50
    height: int = 50
    seed: int = 42


@dataclass
class AgentConfig:
    """Configuration for agents."""

    initial_energy: float = 100.0
    move_cost: float = 1.0
    food_eat: float = 10.0
    count: int = 1
    # Motor limits used to derive near-food thrust (rad/s and world units/step
    # per thrust unit). Defaults match the historical hardcoded constants.
    max_turn_rate: float = 3.0
    speed_factor: float = 8.0


@dataclass
class HazardConfig:
    """Configuration for hazards."""

    damage_per_step: float = 1.0
    count: int = 4
    detection_radius: float = 30.0


@dataclass
class FoodConfig:
    """Configuration for food."""

    spawn_rate: float = 0.1
    max_food: int = 20
    regrow_rate: float = 0.0
    supply_ratio: float | None = None
    # Radius within which an agent can eat a patch (world units).
    eat_dist: float = 1.5


@dataclass
class SimLabConfig:
    """Main configuration container for SimLab."""

    world: WorldConfig = field(default_factory=WorldConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    food: FoodConfig = field(default_factory=FoodConfig)
    drive: str = "simple"
    retention: str = "fixed_decay"
    policy: str = "heuristic"
    # Runtime-injected path to a checkpoint for a learned policy. Never enters
    # to_dict() so it does not affect resolved config hashes or baselines.
    policy_checkpoint: str | None = None
    # Runtime-injected kwargs passed to the selected drive core (GUI
    # architecture panel). Never serialized; world behavior is unchanged when
    # left as None.
    drive_config: dict[str, Any] | None = None
    # Runtime-injected policy config dict (used when no checkpoint is loaded).
    # Never serialized.
    policy_cfg: dict[str, Any] | None = None

    @classmethod
    def from_toml(cls, path: str) -> "SimLabConfig":
        """Load configuration from a TOML file.

        Args:
            path: Path to the TOML configuration file.

        Returns:
            SimLabConfig instance populated with values from the TOML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            tomllib.TOMLDecodeError: If the TOML file is malformed.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimLabConfig":
        """Create SimLabConfig from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            SimLabConfig instance.
        """
        world_data = data.get("world", {})
        agent_data = data.get("agent", {})
        hazard_data = data.get("hazard", {})
        food_data = data.get("food", {})
        components = data.get("components", {})

        return cls(
            world=WorldConfig(
                width=world_data.get("width", 50),
                height=world_data.get("height", 50),
                seed=world_data.get("seed", 42),
            ),
            agent=AgentConfig(
                initial_energy=agent_data.get(
                    "initial_energy", agent_data.get("energy", 100.0)
                ),
                move_cost=agent_data.get("move_cost", 1.0),
                food_eat=agent_data.get("food_eat", 10.0),
                count=agent_data.get("count", 1),
                max_turn_rate=agent_data.get("max_turn_rate", 3.0),
                speed_factor=agent_data.get("speed_factor", 8.0),
            ),
            hazard=HazardConfig(
                damage_per_step=hazard_data.get("damage_per_step", 1.0),
                count=hazard_data.get("count", 4),
                detection_radius=hazard_data.get("detection_radius", 30.0),
            ),
            food=FoodConfig(
                spawn_rate=food_data.get("spawn_rate", 0.1),
                max_food=food_data.get("max_food", 20),
                regrow_rate=food_data.get("regrow_rate", 0.0),
                supply_ratio=food_data.get("supply_ratio"),
                eat_dist=food_data.get("eat_dist", 1.5),
            ),
            drive=components.get("drive", "simple"),
            retention=components.get("retention", "fixed_decay"),
            policy=components.get("policy", "heuristic"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "world": {
                "width": self.world.width,
                "height": self.world.height,
                "seed": self.world.seed,
            },
            "agent": {
                "initial_energy": self.agent.initial_energy,
                "move_cost": self.agent.move_cost,
                "food_eat": self.agent.food_eat,
                "count": self.agent.count,
                "max_turn_rate": self.agent.max_turn_rate,
                "speed_factor": self.agent.speed_factor,
            },
            "hazard": {
                "damage_per_step": self.hazard.damage_per_step,
                "count": self.hazard.count,
                "detection_radius": self.hazard.detection_radius,
            },
            "food": {
                "spawn_rate": self.food.spawn_rate,
                "max_food": self.food.max_food,
                "regrow_rate": self.food.regrow_rate,
                "eat_dist": self.food.eat_dist,
                **(
                    {"supply_ratio": self.food.supply_ratio}
                    if self.food.supply_ratio is not None
                    else {}
                ),
            },
            "components": {
                "drive": self.drive,
                "retention": self.retention,
                "policy": self.policy,
            },
        }

    def save(self, path: str) -> None:
        """Save configuration to a TOML file.

        Args:
            path: Path where the TOML configuration file will be saved.
        """
        import tomllib

        data = self.to_dict()
        with open(path, "wb") as f:
            tomllib.dump(data, f)


# Default configuration loader
def load_config(path: str = "configs/default.toml") -> SimLabConfig:
    """Load SimLab configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file. Defaults to "configs/default.toml".

    Returns:
        SimLabConfig instance populated with values from the TOML file.
    """
    return SimLabConfig.from_toml(path)
