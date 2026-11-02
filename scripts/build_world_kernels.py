"""Reproduce the checked-in PTX from WeightWorld's CUDA source using LLVM."""
import os
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
clang = os.environ.get("CLANG") or "/opt/homebrew/opt/llvm/bin/clang"
if not Path(clang).is_file():
    clang = shutil.which("clang") or clang
targets = subprocess.check_output([clang, "--print-targets"], text=True)
if "nvptx" not in targets:
    raise SystemExit("Use an LLVM clang with the NVPTX backend; set CLANG=/path/to/clang")
for name in ("clear", "advance", "draw"):
    src = root / f"kernels/world_{name}.cu"
    subprocess.run([clang, "-x", "cuda", "--cuda-device-only", "-nocudainc", "-nocudalib",
                    "--cuda-gpu-arch=sm_70", "-Xclang", "-target-feature", "-Xclang", "+ptx70",
                    "-O2", "-S", str(src), "-o", str(src.with_suffix(".ptx"))], check=True)
    print(src.with_suffix(".ptx").name)
