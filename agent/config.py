from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

WORKSPACE = Path(os.environ.get("AUTOWORK_WORKSPACE", ROOT / "workspace"))
WORKSPACE_SEED = ROOT / "workspace_seed"
RUNS_DIR = ROOT / "runs"
PLAYBOOK_PATH = ROOT / "data" / "playbook.json"
WORLD_URL = os.environ.get("WORLD_URL", "http://localhost:8001")


def reset_workspace():
    """Restore the shared workspace to its seed contents (outputs from earlier runs are removed)."""
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(WORKSPACE_SEED, WORKSPACE)
