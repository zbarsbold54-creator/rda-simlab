"""Min-turn-radius invariant for principled heuristic policies."""

from simlab.core.registry import get, load_plugins

load_plugins()

# Principled heuristics must be able to turn tighter than the eat range.
PRIJNCIPLED = ["heuristic", "heuristic_noflee", "oracle", "oracle_noflee"]
# Frozen world-v1 versions intentionally violate the invariant (thrust 0.6).
FROZEN = ["heuristic_v2", "oracle_v2"]

_EAT_DIST = 1.5
_MAX_TURN_RATE = 3.0
_SPEED_FACTOR = 8.0


def _min_turn_radius(policy) -> float:
    return policy.eat_thrust * policy.speed_factor / policy.max_turn_rate


def test_principled_heuristics_turn_inside_eat_range():
    for name in PRIJNCIPLED:
        policy_type = get("policy", name)
        assert policy_type is not None, name
        policy = policy_type()
        assert _min_turn_radius(policy) < policy.eat_dist
        # The invariant is derived to be comfortably inside the eat range.
        assert _min_turn_radius(policy) <= 0.7 * policy.eat_dist


def test_frozen_v2_heuristics_violate_invariant():
    # The world-v1 frozen policies use thrust 0.6 -> min radius 1.6 > 1.5.
    for name in FROZEN:
        policy_type = get("policy", name)
        assert policy_type is not None, name
        policy = policy_type()
        assert _min_turn_radius(policy) >= policy.eat_dist


def test_reverting_thrust_to_0_6_fails_invariant():
    # Simulate the old hardcoded thrust on a principled heuristic: the
    # invariant predicate must report False for thrust 0.6.
    policy_type = get("policy", "heuristic")
    policy = policy_type()
    policy.eat_thrust = 0.6
    assert _min_turn_radius(policy) > _EAT_DIST
    assert not (_min_turn_radius(policy) < policy.eat_dist)
