import sys
from pathlib import Path

# Repo root on sys.path so `tools.gis_pipeline.*` and `roboto_core.*` import
# without an editable install. Keeps the offline test loop dependency-free.
ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "ros2_ws" / "src" / "roboto_core"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
