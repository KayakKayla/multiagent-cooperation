"""Station coordinates transcribed from the local Story 1-3 bridge snapshot.

Grid column = native X + 6; row = 3 - native Z. Two extra solid obstacles
were identified in collider captures. Some boundary/passage clearance, the
serving cell, and rounded spawns remain approximate at this resolution.
Unobserved geometry is declared here, not inferred as floor at runtime.
"""
from enum import IntEnum


class Tile(IntEnum):
    FLOOR = 0
    WALL = 1
    COUNTER = 2
    BOARD = 3
    HOB = 4
    CUCUMBER = 5
    RICE = 6
    NORI = 7
    SERVE = 8
    RETURN = 9
    SINK = 10
    DRYING = 11
    BIN = 12
    PRAWN = 13  # Present in the map; outside this prototype's one recipe.


STATIONS = {
    (0, 1): Tile.WALL, (1, 1): Tile.BOARD, (2, 1): Tile.COUNTER,
    (3, 1): Tile.BOARD, (4, 1): Tile.COUNTER, (5, 1): Tile.BOARD,
    (6, 1): Tile.COUNTER, (0, 2): Tile.COUNTER,
    (0, 3): Tile.RICE, (0, 4): Tile.PRAWN, (0, 5): Tile.NORI,
    (0, 6): Tile.SERVE,
    (3, 4): Tile.COUNTER, (3, 5): Tile.COUNTER, (3, 6): Tile.COUNTER,
    (6, 2): Tile.COUNTER, (6, 3): Tile.COUNTER,
    (6, 4): Tile.COUNTER, (6, 5): Tile.COUNTER, (6, 6): Tile.COUNTER,
    (6, 7): Tile.BIN,
    (9, 0): Tile.COUNTER, (10, 0): Tile.HOB, (11, 0): Tile.HOB,
    (12, 0): Tile.HOB, (13, 0): Tile.WALL,
    (13, 1): Tile.COUNTER, (13, 2): Tile.COUNTER,
    (13, 3): Tile.CUCUMBER, (13, 4): Tile.COUNTER,
    (13, 5): Tile.COUNTER, (13, 6): Tile.COUNTER,
    (9, 7): Tile.DRYING, (10, 7): Tile.SINK, (11, 7): Tile.RETURN,
    # Centers inside real wall colliders Col_Objects (11) and (10).
    # Cell (3, 8) stays floor: a chef traversed the off-center lower passage.
    (3, 7): Tile.WALL, (9, 8): Tile.WALL,
}


def grid():
    width, height = 14, 10
    cells = [[int(Tile.FLOOR) for _ in range(width)] for _ in range(height)]
    for x in range(width):
        cells[height - 1][x] = int(Tile.WALL)
        if x < 9:
            cells[0][x] = int(Tile.WALL)
    for y in range(height):
        cells[y][0] = cells[y][-1] = int(Tile.WALL)
    for (x, y), tile in STATIONS.items():
        cells[y][x] = int(tile)
    return cells


SPAWNS = ((4, 3), (11, 3))
POTS = ((10, 0), (11, 0), (12, 0))
PLATES = ((3, 4), (3, 6), (13, 2), (13, 5))
RETURN_CELL, DRYING_CELL = (11, 7), (9, 7)
