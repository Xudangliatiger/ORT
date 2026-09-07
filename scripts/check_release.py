"""Dependency-free syntax and accidental-sensitive-file checks (not GPU tests)."""
import ast
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
issues = []
count = 0
patterns = [r"/Users/", r"/gpfs/", r"/leonardo/", r"/p/project/", r"/e/project", r"/e/scratch",
            r"-----BEGIN .*PRIVATE KEY-----", r"hf_[A-Za-z0-9]{20,}",
            r"gh[pousr]_[A-Za-z0-9]{20,}"]
for path in root.rglob("*"):
    if not path.is_file() or any(p in {".git", ".venv", "__pycache__", ".pytest_cache"} for p in path.parts):
        continue
    if path.suffix in {".bin", ".pt", ".pth", ".npz", ".safetensors"}:
        issues.append(f"Unexpected binary artifact: {path.relative_to(root)}")
        continue
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        continue
    text = path.read_text()
    if path.suffix == ".py":
        ast.parse(text, filename=str(path))
        count += 1
    if path.name != "check_release.py" and any(re.search(p, text) for p in patterns):
        issues.append(f"Potential private content: {path.relative_to(root)}")
if issues:
    raise SystemExit("\n".join(issues))
print(f"Parsed {count} Python files; no matched private-path/key or binary patterns.")
print("This bounded scan is not a full credential or license audit.")
