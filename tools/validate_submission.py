#!/usr/bin/env python3
# Validate ARC-AGI-2 submission.json for 2 attempts with correct grid shapes.
import json, sys
from typing import Any, Dict, List

def is_grid(g: Any) -> bool:
    if not isinstance(g, list) or not g: return False
    row_len = None
    for r in g:
        if not isinstance(r, list) or not r: return False
        if row_len is None: row_len = len(r)
        if len(r) != row_len: return False
        for v in r:
            if not isinstance(v, int) or not (0 <= v <= 9): return False
    return True

def same_shape(a: List[List[int]], b: List[List[int]]) -> bool:
    return len(a) == len(b) and len(a[0]) == len(b[0])

def main(path: str) -> int:
    data: Dict[str, Any] = json.loads(open(path, "r").read())
    if not isinstance(data, dict):
        print("Top-level must be an object {task_id: [ {attempt_1: grid, attempt_2: grid} ] }"); return 2
    bad = 0
    for tid, arr in data.items():
        if not isinstance(arr, list) or not arr:
            print(f"[{tid}] value must be non-empty list"); bad += 1; continue
        rec = arr[0]
        if not isinstance(rec, dict) or "attempt_1" not in rec or "attempt_2" not in rec:
            print(f"[{tid}] must contain dict with attempt_1 and attempt_2"); bad += 1; continue
        a1, a2 = rec["attempt_1"], rec["attempt_2"]
        if not is_grid(a1) or not is_grid(a2):
            print(f"[{tid}] attempts must be 2D int grids"); bad += 1; continue
        if not same_shape(a1, a2):
            print(f"[{tid}] attempt_1 and attempt_2 shapes must match"); bad += 1; continue
        # extra: ensure attempts are distinct
        if a1 == a2:
            print(f"[{tid}] attempts are identical; diversity required"); bad += 1; continue
    if bad == 0:
        print(f"[OK] {path} looks valid.")
    return 0 if bad == 0 else 3

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: tools/validate_submission.py /kaggle/working/submission.json")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
