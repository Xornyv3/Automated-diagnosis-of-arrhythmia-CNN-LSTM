"""Quick smoke test: compile sources to bytecode to catch syntax errors.

This test avoids importing heavy dependencies (e.g., TensorFlow) so it can
run even if the ML stack isn't installed yet. It returns a non-zero exit code
if compilation fails for any .py file under src/ or scripts/.
"""
import sys
import compileall
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src_ok = compileall.compile_dir(str(root / "src"), force=True, quiet=1)
scripts_ok = compileall.compile_dir(str(root / "scripts"), force=True, quiet=1)
print("SRC_OK", bool(src_ok))
print("SCRIPTS_OK", bool(scripts_ok))
if not (src_ok and scripts_ok):
	sys.exit(1)
print("Smoke compile passed.")
