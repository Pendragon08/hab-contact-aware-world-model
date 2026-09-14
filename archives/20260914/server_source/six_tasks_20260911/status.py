"""Read-only consolidated status for all six task streams."""
import json,pathlib
from sample import snapshot
ROOT=pathlib.Path(__file__).resolve().parent
s=snapshot()
s['manager_state']=json.loads((ROOT/'state.json').read_text()) if (ROOT/'state.json').exists() else None
print(json.dumps(s,indent=2))
