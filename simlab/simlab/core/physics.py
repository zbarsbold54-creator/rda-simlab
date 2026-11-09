"""SimLab physics module - movement, collisions, arena bounds."""

from __future__ import annotations


def apply_movement(
    position: tuple[int, int],
    direction: tuple[int, int],
    world_width: int,
    world_height: int,
    wrap_around: bool = False,
) -> tuple[int, int]:
    """Apply movement direction to position with boundary checking.

    Args:
        position: Current (x, y) position.
        direction: Movement direction (dx, dy).
        world_width: Width of the world for boundary checking.
        world_height: Height of the world for boundary checking.
        wrap_around: Whether to wrap around world edges (toroidal).

    Returns:
        New (x, y) position after movement.
    """
    x, y = position
    dx, dy = direction

    new_x = x + dx
    new_y = y + dy

    if wrap_around:
        # Toroidal wrapping - wrap around edges
        new_x = new_x % world_width
        new_y = new_y % world_height
    else:
        # Boundary clamping - stay within bounds
        new_x = max(0, min(new_x, world_width - 1))
        new_y = max(0, min(new_y, world_height - 1))

    return (new_x, new_y)


def check_collision(
    position1: tuple[int, int],
    position2: tuple[int, int],
    tolerance: int = 1,
) -> bool:
    """Check if two positions collide within a tolerance.

    Args:
        position1: First (x, y) position.
        position2: Second (x, y) position.
        tolerance: Manhattan distance tolerance for collision.

    Returns:
        True if positions are within tolerance of each other.
    """
    x1, y1 = position1
    x2, y2 = position2
    distance = abs(x1 - x2) + abs(y1 - y2)
    return distance <= tolerance


def compute_reflection(
    position: tuple[int, int],
    direction: tuple[int, int],
    world_width: int,
    world_height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute reflection when hitting world boundaries.

    Args:
        position: Current (x, y) position.
        direction: Current (dx, dy) direction.
        world_width: Width of the world.
        world_height: Height of the world.

    Returns:
        Tuple of (new_position, new_direction) after reflection.
    """
    x, y = position
    dx, dy = direction

    new_dx = dx
    new_dy = dy

    # Check horizontal boundaries
    if x <= 0 or x >= world_width - 1:
        new_dx = -dx

    # Check vertical boundaries
    if y <= 0 or y >= world_height - 1:
        new_dy = -dy

    # Calculate new position after reflection
    new_x = x + new_dx
    new_y = y + new_dy

    # Clamp to bounds
    new_x = max(0, min(new_x, world_width - 1))
    new_y = max(0, min(new_y, world_height - 1))

    return ((new_x, new_y), (new_dx, new_dy))


def compute_arena_bounds(
    position: tuple[int, int],
    world_width: int,
    world_height: int,
    margin: int = 0,
) -> dict[str, int]:
    """Compute arena bounds around a position.

    Args:
        position: (x, y) position to compute bounds around.
        world_width: Width of the world.
        world_height: Height of the world.
        margin: Margin to add around the position.

    Returns:
        Dictionary with min_x, max_x, min_y, max_y bounds.
    """
    x, y = position

    return {
        "min_x": max(0, x - margin),
        "max_x": min(world_width - 1, x + margin),
        "min_y": max(0, y - margin),
        "max_y": min(world_height - 1, y + margin),
    }
