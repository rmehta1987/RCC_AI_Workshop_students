import sys
from pathlib import Path

# Make `common`, `vep_scorer`, etc. importable from the scripts/ dir.
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
