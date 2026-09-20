#!/usr/bin/env python3
"""Add an import-safe entry point to debug_one.py only.

This utility does NOT modify ECSP solvers, settings, handoffs, or run results.
It corrects the unguarded debug harness and propagates caught failures as
non-zero process exits. It does not solve a thermochemical range violation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

MARKER = "# ECSP_DEBUG_SPAWN_GUARD_V1"


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    t = node.test
    if not isinstance(t, ast.Compare) or len(t.ops) != 1:
        return False
    if not isinstance(t.ops[0], ast.Eq) or len(t.comparators) != 1:
        return False
    a, b = t.left, t.comparators[0]
    def name(n):
        return isinstance(n, ast.Name) and n.id == "__name__"
    def value(n):
        return isinstance(n, ast.Constant) and n.value == "__main__"
    return (name(a) and value(b)) or (value(a) and name(b))


def transform(source: str, filename: str) -> tuple[str, int]:
    tree = ast.parse(source, filename=filename)
    if any(is_main_guard(n) for n in tree.body):
        raise ValueError("A main guard already exists. Refusing to wrap an unknown script layout.")
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main"
           for n in tree.body):
        raise ValueError("main() already exists. No automatic double-wrapping is allowed.")
    if any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in tree.body):
        raise ValueError("Unexpected future import. No file was changed.")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id in {"run_post_onset", "run_post_onset_batch"}]
    if len(calls) != 1:
        raise ValueError(f"Expected one post-onset call; found {len(calls)}. No file was changed.")

    # The supplied debug script prints exceptions but swallows them. Retain
    # diagnostic printing and ensure a failed calculation exits with failure.
    changed_handlers = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        has_print_exc = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "traceback" and n.func.attr == "print_exc"
            for statement in node.body for n in ast.walk(statement)
        )
        if has_print_exc and not isinstance(node.body[-1], ast.Raise):
            node.body.append(ast.Raise(exc=None, cause=None))
            changed_handlers += 1
    if changed_handlers != 1:
        raise ValueError("Expected one diagnostic exception handler. No file was changed.")

    # All original code, including data loading and helper definitions, runs
    # only inside main(). Spawned workers can import this file without starting
    # another pool or reading/rewriting candidate output.
    wrapper = ast.parse("def main():\n    pass\n")
    wrapper.body[0].body = tree.body
    wrapper.body.extend(ast.parse(
        "if __name__ == '__main__':\n"
        "    import multiprocessing as _mp\n"
        "    _mp.freeze_support()\n"
        "    main()\n"
    ).body)
    ast.fix_missing_locations(wrapper)
    updated = MARKER + "\n" + ast.unparse(wrapper) + "\n"
    compile(updated, filename, "exec")
    return updated, changed_handlers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="Exact path to the existing debug_one.py")
    args = parser.parse_args()
    p = args.script.expanduser().resolve(strict=True)
    if p.name != "debug_one.py" or not p.is_file():
        parser.error("Only a file named debug_one.py may be patched by this utility.")
    raw = p.read_bytes()
    source = raw.decode("utf-8")
    if MARKER in source:
        print("Already patched:", p)
        return 0
    updated, n = transform(source, str(p))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    backup = p.with_name(p.name + ".bak_spawn_" + stamp)
    shutil.copy2(p, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=p.parent,
                                         prefix=p.name + ".", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            f.write(updated)
            f.flush()
            os.fsync(f.fileno())
        shutil.copymode(p, temporary)
        os.replace(temporary, p)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print("Patched:", p)
    print("Backup :", backup)
    print("Before SHA256:", hashlib.sha256(raw).hexdigest())
    print("After  SHA256:", hashlib.sha256(p.read_bytes()).hexdigest())
    print(f"Guard installed; {n} exception handler now re-raises failures.")
    print("No solver/config/handoff/result files were changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        raise SystemExit(f"STOP: {exc}")
