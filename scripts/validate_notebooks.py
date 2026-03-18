from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def prepare_env() -> None:
    temp_root = Path(r"C:\Users\41424\.codex\memories\tmpml")
    temp_root.mkdir(exist_ok=True)
    os.environ["TMP"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    os.environ["ML_DL_TMP"] = str(temp_root)
    os.environ["MPLBACKEND"] = "Agg"


def execute_notebook(path: Path) -> None:
    namespace = {"__name__": "__main__"}
    nb = nbformat.read(path, as_version=4)
    for cell_index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source.strip()
        if not source:
            continue
        try:
            exec(compile(source, f"{path.name}#cell{cell_index}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILED] {path.name} cell {cell_index}: {exc}")
            traceback.print_exc()
            raise
    print(f"[OK] {path.name}")


def main() -> None:
    prepare_env()
    if len(sys.argv) > 1:
        execute_notebook(Path(sys.argv[1]))
        return

    env = os.environ.copy()
    for notebook_path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(notebook_path)],
            check=True,
            env=env,
            cwd=str(ROOT),
        )


if __name__ == "__main__":
    main()
