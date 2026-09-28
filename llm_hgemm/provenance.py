"""Reproducibility fingerprints; called during setup, never per GEMM."""
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def code_fingerprint():
    digest = sha256()
    paths = []
    for folder in ("csrc", "llm_hgemm", "benchmarks"):
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in
                     {".py", ".cu", ".cpp", ".h", ".cuh", ".so"})
    if (ROOT / "setup.py").exists():
        paths.append(ROOT / "setup.py")
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
