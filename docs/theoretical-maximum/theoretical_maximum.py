"""Audit model for the pinned Coordination Ring rules.

This is a small standard-library model for auditing the saved action schedule.
It permits every reachable counter, including the central handoff counter.
"""
import json
from pathlib import Path


GRID = ['WWWPW', 'W   P', 'B W W', 'O   W', 'WOXWW']
FLOOR = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)]
VEC = [(0, -1), (0, 1), (1, 0), (-1, 0)]
DIR = [2, 1, 3, 0, -1, -1]
NAMES = ['right', 'down', 'left', 'up', 'stay', 'interact']
CELLS = sorted({(x+dx, y+dy) for x, y in FLOOR for dx, dy in VEC} - set(FLOOR))
TYPES = [GRID[y][x] for x, y in CELLS] + ['N']
# Inventory: empty=0, onion=1, plate=2, soup=3.
# Pot codes: 10=empty, 11/12=one/two onions, 13=cooked,
# 14..33=countdown 1..20. Tick occurs after both interactions.
DOMAINS = {'W': list(range(4)), 'P': list(range(10, 34)),
           'O': [101], 'B': [102], 'X': [103], 'N': [100]}


def interact(code, inv):
    out, obj, delivery = inv, code, 0
    if code == 101 and inv == 0:
        out = 1
    elif code == 102 and inv == 0:
        out = 2
    elif code == 103 and inv == 3:
        out, delivery = 0, 1
    elif code < 4:
        if inv == 0 and code:
            out, obj = code, 0
        elif inv and code == 0:
            out, obj = 0, inv
    elif code in (10, 11, 12) and inv == 1:
        out, obj = 0, code + 1 if code < 12 else 33
    elif code == 13 and inv == 2:
        out, obj = 3, 10
    return out, obj, delivery


def tick(code):
    return code-1 if 14 <= code <= 33 else code


def movement(pos, direction, action):
    x, y = FLOOR[pos]
    nd = DIR[action] if DIR[action] >= 0 else direction
    np = pos
    target = len(CELLS)
    if action < 4:
        dx, dy = VEC[nd]
        if (x+dx, y+dy) in FLOOR:
            np = FLOOR.index((x+dx, y+dy))
    elif action == 5:
        dx, dy = VEC[direction]
        if (x+dx, y+dy) in CELLS:
            target = CELLS.index((x+dx, y+dy))
    return np, nd, target


def replay(actions):
    positions, directions, inventory = [1, 7], [0, 0], [0, 0]
    cells = [DOMAINS[k][0] for k in TYPES]
    rows, score = [], 0
    for t, pair in enumerate(actions, 1):
        desired, nd, targets = zip(*(movement(positions[a], directions[a], pair[a]) for a in range(2)))
        if desired[0] != desired[1] and not (desired[0] == positions[1] and desired[1] == positions[0]):
            positions = list(desired)
        directions = list(nd)
        assert positions[0] != positions[1]
        events = []
        for a in range(2):
            old, oldinv = cells[targets[a]], inventory[a]
            inventory[a], cells[targets[a]], delivered = interact(old, oldinv)
            score += delivered
            if delivered:
                events.append(f'chef {a} delivers')
            elif old != cells[targets[a]] or oldinv != inventory[a]:
                events.append(f'chef {a}: inventory {oldinv}->{inventory[a]}, cell {targets[a]} {old}->{cells[targets[a]]}')
        cells = list(map(tick, cells))
        rows.append([t, *[NAMES[a] for a in pair], *[str(FLOOR[p]) for p in positions],
                     *directions, *inventory, cells[CELLS.index((3,0))], cells[CELLS.index((4,1))],
                     cells[CELLS.index((2,2))], score, '; '.join(events)])
    return score, rows



if __name__ == '__main__':
    actions = json.loads((Path(__file__).parent/'theoretical-maximum-actions.json').read_text())
    count, _ = replay(actions)
    assert count == 18
    print(f'{count} deliveries, {20*count} reward; replay passed')
