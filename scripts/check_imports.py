#!/usr/bin/env python3
"""
Kiểm tra import cục bộ trong các file Python có bị broken không.

Logic:
  Với mỗi import statement trong file được kiểm tra:
    1. Nếu file .py tương ứng tồn tại trong project → OK
    2. Nếu module có thể tìm thấy qua importlib (stdlib / third-party) → OK
    3. Không thỏa cả hai → broken (có thể file đã bị đổi tên / xóa)

Usage:
  python scripts/check_imports.py file1.py file2.py ...
  (thường được gọi bởi .git/hooks/pre-commit)
"""

import ast
import importlib.util
import io
import sys
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

_SKIP_DIRS = {
    ".venv", ".git", "__pycache__", "node_modules",
    "scripts", "checkpoints", "logs", "plots",
}

# Imports không cần kiểm tra (chắc chắn là stdlib hoặc third-party)
_ALWAYS_SKIP = {
    "__future__", "typing", "abc", "os", "sys", "re", "io", "ast",
    "math", "time", "datetime", "pathlib", "logging", "random",
    "collections", "itertools", "functools", "dataclasses",
    "importlib", "inspect", "copy", "json", "yaml",
    "subprocess", "threading", "multiprocessing", "argparse",
    "torch", "numpy", "np", "matplotlib", "tqdm",
    "pyspiel", "PIL", "cv2", "sklearn", "scipy", "pandas",
}


# ─────────────────────────────────────────────────────────────────────────────
# sys.path setup — để importlib tìm được module trong project + venv
# ─────────────────────────────────────────────────────────────────────────────

def _setup_path():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # .venv Windows
    win_sp = ROOT / ".venv" / "Lib" / "site-packages"
    if win_sp.exists() and str(win_sp) not in sys.path:
        sys.path.insert(1, str(win_sp))

    # .venv Unix
    for sp in ROOT.glob(".venv/lib/python*/site-packages"):
        if sp.exists() and str(sp) not in sys.path:
            sys.path.insert(1, str(sp))


# ─────────────────────────────────────────────────────────────────────────────
# Kiểm tra khả năng resolve
# ─────────────────────────────────────────────────────────────────────────────

def _local_path(module: str) -> Path | None:
    """
    Tìm file .py tương ứng với tên module dotted trong project.
    VD: 'utils.losses' → ROOT/utils/losses.py
    """
    parts = module.split(".")
    # dạng file: foo.bar → ROOT/foo/bar.py
    f = ROOT.joinpath(*parts[:-1], parts[-1] + ".py")
    if f.exists():
        return f
    # dạng package: foo.bar → ROOT/foo/bar/__init__.py
    p = ROOT.joinpath(*parts, "__init__.py")
    if p.exists():
        return p
    return None


def _can_import(module_top: str) -> bool:
    """Kiểm tra qua importlib — tìm stdlib và installed packages."""
    try:
        return importlib.util.find_spec(module_top) is not None
    except Exception:
        return False


def _suggest_rename(module: str) -> str:
    """Gợi ý file nào trong project có tên gần giống module bị broken."""
    top = module.split(".")[0].lower()
    candidates = []
    for p in ROOT.rglob("*.py"):
        if any(s in p.parts for s in _SKIP_DIRS):
            continue
        if p.stem.lower() == top or top in p.stem.lower() or p.stem.lower() in top:
            candidates.append(str(p.relative_to(ROOT)))
    if candidates:
        return "Tìm thấy file tương tự: " + ", ".join(candidates[:3])
    return "Không tìm thấy file nào tương tự trong project."


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_imports(filepath: Path) -> list[tuple[str, int, str]]:
    """Dùng ast để parse import — không thực thi file."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        print(f"  ⚠  SYNTAX ERROR  {filepath.name}:{e.lineno}  {e.msg}")
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((alias.name, node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # absolute only
                names = ", ".join(a.name for a in node.names)
                results.append((node.module, node.lineno, f"from {node.module} import {names}"))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Checker
# ─────────────────────────────────────────────────────────────────────────────

def check_file(filepath: Path) -> list[dict]:
    """
    Trả về danh sách broken imports trong filepath.

    Một import bị đánh dấu broken khi:
    - File .py tương ứng KHÔNG tồn tại trong project
    - VÀ importlib KHÔNG tìm thấy module (không phải stdlib/installed)
    """
    issues = []
    seen_tops: set[str] = set()

    for module, lineno, stmt in _parse_imports(filepath):
        top = module.split(".")[0]

        # Bỏ qua danh sách whitelist
        if top in _ALWAYS_SKIP:
            continue
        # Bỏ qua top-level đã kiểm tra
        if top in seen_tops:
            # Vẫn check full dotted path cho lần đầu gặp
            pass

        # 1. Tồn tại trong project → OK
        if _local_path(module) is not None:
            seen_tops.add(top)
            continue

        # 2. Tìm thấy qua importlib (stdlib / third-party) → OK
        if _can_import(top):
            seen_tops.add(top)
            continue

        # 3. Không tìm thấy ở đâu → broken
        seen_tops.add(top)
        issues.append({
            "lineno": lineno,
            "stmt":   stmt,
            "module": module,
            "hint":   _suggest_rename(module),
        })

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    _setup_path()

    input_args = [f for f in sys.argv[1:] if f.endswith(".py")]
    if not input_args:
        print("  (không có file .py nào để kiểm tra)")
        return 0

    files: list[Path] = []
    for f in input_args:
        p = Path(f)
        abs_p = p if p.is_absolute() else ROOT / p
        if abs_p.exists():
            files.append(abs_p)

    if not files:
        print("  (tất cả file đã staged không còn tồn tại — có thể bị xóa)")
        return 0

    has_error = False
    for filepath in files:
        issues = check_file(filepath)
        if not issues:
            continue

        has_error = True
        try:
            display = filepath.relative_to(ROOT)
        except ValueError:
            display = filepath  # file ngoài ROOT (ví dụ /tmp/...)
        print(f"\n  ✗  {display}")
        for iss in issues:
            print(f"     line {iss['lineno']:>4}:  {iss['stmt']}")
            print(f"             → {iss['hint']}")

    if has_error:
        print("\n  ✗ Có import bị broken. Sửa trước khi commit.\n")
        return 1

    print(f"  ✓  {len(files)} file(s) — tất cả import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
