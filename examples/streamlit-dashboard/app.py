"""Streamlit Community Cloud entrypoint for the public dashboard demo."""
from pathlib import Path
import runpy
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

runpy.run_path(str(REPOSITORY_ROOT / "app.py"), run_name="__main__")
