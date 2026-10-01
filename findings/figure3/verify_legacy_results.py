"""Recompute the archived 400-example aggregate accuracies from raw predictions."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "legacy_400/fixed_all_results.json").read_text(encoding="utf-8"))
frozen = json.loads((HERE / "legacy_400/frozen_fixed_all_results.json").read_text(encoding="utf-8"))
assert len(data) == len(frozen) == 400
assert {r["id"] for r in data} == {r["id"] for r in frozen}
normal = sum(r["groups"]["normal"]["correct"] for r in data)
blocked = sum(r["groups"]["blocked"]["correct"] for r in data)
fixed = sum(r["frozen"]["correct"] for r in frozen)
assert (normal, blocked, fixed) == (280, 223, 175)
print(json.dumps({"samples": 400, "normal_percent": normal / 4,
                  "blocked_percent": blocked / 4, "frozen_percent": fixed / 4,
                  "current_paper_figure": False}, indent=2))
