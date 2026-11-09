# EXAMPLE, not project code.
# This tiny 1D world shows the NumPy pattern we will reuse in the real sim.

import numpy as np

# 1D world: 5 cells. 0 = empty, 1 = occupied.
world = np.array([0, 1, 0, 1, 0], dtype=np.int8)  # shape (5,)
print("Initial world:", world)  # shape (5,)
print("Cell 2:", world[2])  # scalar

# A simple update rule:
# - If a cell is already alive, it stays alive.
# - If a cell is empty and it has at least one live neighbor, it becomes alive.
left = np.roll(world, 1)  # shape (5,)
right = np.roll(world, -1)  # shape (5,)

birth_mask = (world == 0) & ((left == 1) | (right == 1))  # shape (5,)
next_world = (world | birth_mask.astype(np.int8)).astype(np.int8)  # shape (5,)

print("Birth mask:", birth_mask)  # shape (5,)
print("Next world:", next_world)  # shape (5,)

# Slow reference: same rule written in plain Python loops.
slow_next_world = np.zeros_like(world)  # shape (5,)
for i in range(world.shape[0]):
    alive_now = world[i] == 1
    left_alive = world[(i - 1) % world.shape[0]] == 1
    right_alive = world[(i + 1) % world.shape[0]] == 1
    if alive_now or left_alive or right_alive:
        slow_next_world[i] = 1
print("Slow reference:", slow_next_world)  # shape (5,)
