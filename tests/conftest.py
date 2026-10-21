import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Scripts import sibling scripts (e.g. plot_profiles, scripts_path) by name.
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "scripts"))
