"""Keep this approved campaign available for verified synchronization."""
from pathlib import Path
import time
terminal=Path(__file__).resolve().parent / "SYNC_TERMINAL.json"
while not terminal.exists():
    time.sleep(30)
