"""
sudoku.py (controller) — /sudoku: watch the OpenJEV classification approach solve a Sudoku
puzzle one cell at a time.

Same two-speed split as live.py and decide.py, applied to a puzzle instead of text or a game:
  - FREE, always certain: naked singles and hidden singles (app/engine/sudoku.py, pure logic,
    no model). Most of a puzzle -- all of an Easy or Medium one -- is solvable this way alone.
  - A real System-1 model, consulted ONLY for a cell neither technique can place (this is where
    "probabilities for each number" actually comes from -- choose() over that cell's legal
    candidates). Its pick is checked against the puzzle's true solution before being shown: if
    it's right, great; if not, that disagreement is shown too rather than hidden, and the correct
    value is placed so the walkthrough keeps moving instead of stalling on one wrong guess.

Deliberately stateless, same reasoning as live.py: nothing here touches Job/DB. The client holds
the current grid and the solution (this is a transparency demo, not an adversarial game -- there
is no one for the solution to be secret from) and resends both with each "next step" click; the
server never remembers a puzzle between requests.
"""
import threading

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.core.deps import get_registry, get_settings
from app.core.errors import BackendError, ValidationFailed
from app.core.templating import render
from app.engine import sudoku as S

router = APIRouter()
_step_lock = threading.Lock()   # same single-flight guard as live.py: one model call at a time,
                                 # regardless of double-clicks or an auto-play timer overlapping


def _validate_grid(grid: list[int], name: str) -> None:
    if len(grid) != 81 or any(not isinstance(v, int) or v < 0 or v > 9 for v in grid):
        raise ValidationFailed(f"'{name}' must be 81 integers from 0 to 9.")


class NewIn(BaseModel):
    difficulty: str = "easy"


class StepIn(BaseModel):
    grid: list[int] = Field(..., min_length=81, max_length=81)
    solution: list[int] = Field(..., min_length=81, max_length=81)
    backend: str | None = None


@router.get("/sudoku", response_class=HTMLResponse)
def page(request: Request, settings=Depends(get_settings), registry=Depends(get_registry)):
    prof = settings.profile("sudoku")
    return render(request, "sudoku.html", difficulties=S.DIFFICULTIES, default_backend=prof.backend,
                  backends=registry.availability())


@router.post("/api/sudoku/new")
def new_puzzle(body: NewIn):
    if body.difficulty not in S.DIFFICULTIES:
        raise ValidationFailed(f"Unknown difficulty '{body.difficulty}'.",
                               hint=f"Use one of: {', '.join(S.DIFFICULTIES)}.")
    puzzle, solution = S.make_puzzle(body.difficulty)
    return {"puzzle": puzzle, "solution": solution, "given": [v != 0 for v in puzzle],
            "clues": sum(1 for v in puzzle if v), "empties": sum(1 for v in puzzle if v == 0)}


@router.post("/api/sudoku/step")
def step(body: StepIn, settings=Depends(get_settings), registry=Depends(get_registry)):
    _validate_grid(body.grid, "grid")
    _validate_grid(body.solution, "solution")
    grid, solution = list(body.grid), body.solution
    if any(g and g != s for g, s in zip(grid, solution)):
        raise ValidationFailed("The grid has a value that doesn't match its own solution.",
                               hint="This shouldn't happen from normal use of the page -- start a new board.")

    if all(grid):
        return {"done": True}

    naked = S.find_naked_single(grid)
    if naked:
        i, v, cands = naked
        return _placed(i, v, cands, "naked_single", grid)

    hidden = S.find_hidden_single(grid)
    if hidden:
        i, v, cands, kind = hidden
        return _placed(i, v, cands, "hidden_single", grid, unit_kind=kind)

    nxt = S.next_guess_cell(grid)
    if nxt is None:
        return {"done": True}
    i, cands = nxt
    if not cands:
        raise ValidationFailed("This grid has no legal value for an empty cell.",
                               hint="The grid is inconsistent with its own solution -- start a new board.")

    context = S.render_context(grid, i)
    labels = [str(d) for d in sorted(cands)]
    if not _step_lock.acquire(blocking=False):
        return _placed(i, solution[i], cands, "classifier", grid,
                       note="A previous step is still being processed by the model; placing the correct value directly.")
    try:
        prof = settings.profile("sudoku")
        backend_name = settings.backend_for("sudoku", override=body.backend)
        model = registry.get(backend_name)
        picked = model.choose(context, prof.main_category_question, labels, debias=1)
    except BackendError as e:
        return _placed(i, solution[i], cands, "classifier", grid,
                       note=f"Model unavailable: {e.message} {e.hint or ''}".strip())
    finally:
        _step_lock.release()

    model_pick = int(picked["choice"])
    correct = model_pick == solution[i]
    value = model_pick if correct else solution[i]
    return _placed(i, value, cands, "classifier", grid, context=context, backend=backend_name,
                   model_pick=model_pick, correct=correct,
                   probabilities={k: round(v, 4) for k, v in picked["probabilities"].items()})


def _placed(i: int, value: int, cands: set[int], method: str, grid: list[int], **extra) -> dict:
    remaining = sum(1 for v in grid if v == 0) - 1
    return {"done": False, "cell": i, "row": i // 9, "col": i % 9, "value": value,
            "method": method, "candidates": sorted(cands), "remaining_empty": remaining, **extra}
