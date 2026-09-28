"""
sudoku.py (engine) — pure Sudoku logic for the /sudoku Lab: candidate computation, the two
fundamental human solving techniques (naked singles, hidden singles), a unique-solution puzzle
generator, and the text rendering fed to a ChoiceBackend for cells neither technique can place.

Deliberately has no dependency on ChoiceBackend/EngineRegistry -- this module only answers "what
are the rules", never "what should we guess". app/controllers/sudoku.py owns the guess step,
calling a real classifier via choose() when this module reports no certain move.

Puzzle generation mirrors the algorithm in the attached sudoku_playable_.html's makePuzzle():
MRV-heuristic backtracking to fill a full grid, then round-robin clue removal across the nine
boxes, each removal checked against countSolutions()<=2 to guarantee the puzzle stays uniquely
solvable throughout.
"""
from __future__ import annotations

import random

N = 9
ALL_DIGITS = frozenset(range(1, 10))


def _build_units() -> list[list[int]]:
    units = [[r * 9 + c for c in range(9)] for r in range(9)]              # 9 rows
    units += [[r * 9 + c for r in range(9)] for c in range(9)]             # 9 columns
    for br in range(3):                                                    # 9 boxes
        for bc in range(3):
            units.append([(br * 3 + r) * 9 + (bc * 3 + c) for r in range(3) for c in range(3)])
    return units


UNITS: list[list[int]] = _build_units()                                    # 27 units of 9 cells each
CELL_UNITS: list[list[int]] = [[u for u, cells in enumerate(UNITS) if i in cells] for i in range(81)]
PEERS: list[list[int]] = [sorted({j for u in CELL_UNITS[i] for j in UNITS[u]} - {i}) for i in range(81)]

DIFFICULTIES = {
    "easy": {"clues": 44, "min_box": 4, "min_line": 3, "label": "Easy"},
    "medium": {"clues": 34, "min_box": 3, "min_line": 2, "label": "Medium"},
    "hard": {"clues": 23, "min_box": 2, "min_line": 0, "label": "Hard"},
}


def candidates(grid: list[int], i: int) -> set[int]:
    """Legal digits for cell i given what's already placed among its peers. Empty set for a
    filled cell (nothing to place) OR a genuinely stuck cell (a contradiction upstream)."""
    if grid[i]:
        return set()
    used = {grid[j] for j in PEERS[i] if grid[j]}
    return set(ALL_DIGITS) - used


def _most_constrained(grid: list[int]) -> tuple[int, set[int]] | None:
    """Empty cell with the fewest candidates (MRV heuristic) -- both for backtracking (prunes
    the search fastest) and for deciding which cell most deserves a real classifier's attention
    once no free technique applies anywhere. None once the grid is full."""
    best, best_cands, best_n = -1, None, 10
    for i in range(81):
        if grid[i]:
            continue
        c = candidates(grid, i)
        n = len(c)
        if n == 0:
            return i, c                          # contradiction: caller must handle immediately
        if n < best_n:
            best, best_cands, best_n = i, c, n
            if n == 1:
                break
    return (best, best_cands) if best >= 0 else None


def count_solutions(grid: list[int], limit: int = 2) -> int:
    """Counts solutions up to `limit` (never more work than needed to know "unique" vs "not")."""
    g = list(grid)
    count = 0

    def rec() -> None:
        nonlocal count
        if count >= limit:
            return
        nxt = _most_constrained(g)
        if nxt is None:
            count += 1
            return
        i, cands = nxt
        if not cands:
            return
        for v in cands:
            g[i] = v
            rec()
            g[i] = 0
            if count >= limit:
                return

    rec()
    return count


def generate_full(rng: random.Random) -> list[int]:
    """A complete, valid, randomised 81-cell grid via MRV-heuristic backtracking."""
    g = [0] * 81

    def rec() -> bool:
        nxt = _most_constrained(g)
        if nxt is None:
            return True
        i, cands = nxt
        if not cands:
            return False
        vals = list(cands)
        rng.shuffle(vals)
        for v in vals:
            g[i] = v
            if rec():
                return True
            g[i] = 0
        return False

    rec()
    return g


def make_puzzle(difficulty: str, seed: int | None = None) -> tuple[list[int], list[int]]:
    """Returns (puzzle, solution). Clue removal is round-robin across the nine boxes, each
    removal gated on count_solutions()==1 staying true, so the puzzle is uniquely solvable
    throughout -- never just "probably" unique."""
    cfg = DIFFICULTIES[difficulty]
    rng = random.Random(seed)
    solution = generate_full(rng)
    puzzle = list(solution)

    per_box: list[list[int]] = []
    for b in range(9):
        br, bc = (b // 3) * 3, (b % 3) * 3
        cells = [(br + r) * 9 + bc + c for r in range(3) for c in range(3)]
        rng.shuffle(cells)
        per_box.append(cells)

    order: list[int] = []
    for k in range(9):
        boxes = list(range(9))
        rng.shuffle(boxes)
        order += [per_box[b][k] for b in boxes]

    box_n, row_n, col_n = [9] * 9, [9] * 9, [9] * 9
    clues = 81
    for idx in order:
        if clues <= cfg["clues"]:
            break
        r, c = idx // 9, idx % 9
        b = (r // 3) * 3 + (c // 3)
        if box_n[b] <= cfg["min_box"] or row_n[r] <= cfg["min_line"] or col_n[c] <= cfg["min_line"]:
            continue
        keep = puzzle[idx]
        puzzle[idx] = 0
        if count_solutions(puzzle, 2) != 1:
            puzzle[idx] = keep
            continue
        box_n[b] -= 1
        row_n[r] -= 1
        col_n[c] -= 1
        clues -= 1
    return puzzle, solution

def make_puzzle__NEW(difficulty: str, seed: int | None = None) -> tuple[list[int], list[int]]:
    """Returns (puzzle, solution). Clue removal is round-robin across the nine boxes, each
    removal gated on count_solutions()==1 staying true, so the puzzle is uniquely solvable
    throughout -- never just "probably" unique."""
    cfg = DIFFICULTIES[difficulty]
    rng = random.Random(seed)
    solution = generate_full(rng)
    puzzle = list(solution)

    per_box: list[list[int]] = []
    for b in range(9):
        br, bc = (b // 3) * 3, (b % 3) * 3
        cells = [(br + r) * 9 + bc + c for r in range(3) for c in range(3)]
        rng.shuffle(cells)
        per_box.append(cells)

    order: list[int] = []
    for k in range(9):
        boxes = list(range(9))
        rng.shuffle(boxes)
        order += [per_box[b][k] for b in boxes]

    box_n, row_n, col_n = [9] * 9, [9] * 9, [9] * 9
    clues = 81

    # Multi-pass loop: keep attempting clue removal as long as previous passes made progress
    changed = True
    while changed and clues > cfg["clues"]:
        changed = False
        for idx in order:
            if clues <= cfg["clues"]:
                break
            if puzzle[idx] == 0:
                continue

            r, c = idx // 9, idx % 9
            b = (r // 3) * 3 + (c // 3)
            if box_n[b] <= cfg["min_box"] or row_n[r] <= cfg["min_line"] or col_n[c] <= cfg["min_line"]:
                continue

            keep = puzzle[idx]
            puzzle[idx] = 0
            if count_solutions(puzzle, 2) != 1:
                puzzle[idx] = keep
                continue

            box_n[b] -= 1
            row_n[r] -= 1
            col_n[c] -= 1
            clues -= 1
            changed = True

        # Re-shuffle the candidate order for the next sweep so it doesn't test cells in the exact same sequence
        rng.shuffle(order)

    return puzzle, solution

def find_naked_single(grid: list[int]) -> tuple[int, int, set[int]] | None:
    """A cell with exactly one legal candidate -- the most basic, always-certain technique."""
    for i in range(81):
        if grid[i]:
            continue
        c = candidates(grid, i)
        if len(c) == 1:
            return i, next(iter(c)), c
    return None


def find_hidden_single(grid: list[int]) -> tuple[int, int, set[int], str] | None:
    """A digit that has only one legal cell left within some row/column/box, even if that cell
    has other candidates too -- still logically certain, just a different kind of certain."""
    for u in range(27):
        kind = "row" if u < 9 else "column" if u < 18 else "box"
        for d in range(1, 10):
            cells = [i for i in UNITS[u] if grid[i] == 0 and d in candidates(grid, i)]
            if len(cells) == 1:
                i = cells[0]
                return i, d, candidates(grid, i), kind
    return None


def next_guess_cell(grid: list[int]) -> tuple[int, set[int]] | None:
    """The most-constrained empty cell once neither free technique applies anywhere -- this is
    the one a real classifier gets asked about. None if the grid is already full."""
    return _most_constrained(grid)


def render_context(grid: list[int], i: int) -> str:
    """Text description of cell i's row/column/box state, fed to a ChoiceBackend as `context`."""
    r, c = i // 9, i % 9
    b = (r // 3) * 3 + (c // 3)
    row_vals = sorted(grid[j] for j in UNITS[r] if grid[j])
    col_vals = sorted(grid[j] for j in UNITS[9 + c] if grid[j])
    box_vals = sorted(grid[j] for j in UNITS[18 + b] if grid[j])
    cands = sorted(candidates(grid, i))
    fmt = lambda vs: ", ".join(map(str, vs)) if vs else "nothing yet"
    return (
        f"Sudoku cell at row {r + 1}, column {c + 1}. "
        f"Row {r + 1} already has: {fmt(row_vals)}. "
        f"Column {c + 1} already has: {fmt(col_vals)}. "
        f"Its 3x3 box already has: {fmt(box_vals)}. "
        f"After eliminating digits already used in this row, column and box, the digits that "
        f"could still legally go here are: {fmt(cands)}."
    )
