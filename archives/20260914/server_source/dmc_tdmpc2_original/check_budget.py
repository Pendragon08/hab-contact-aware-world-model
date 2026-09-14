"""Validate budget change against the previously witnessed training implementation."""
import ast
import hashlib
import json
from pathlib import Path
import csv
import runner
import finalize

p = Path(__file__).resolve().parent
assert runner.verify_sources() == '57158282b46ebc5c329c5be9cfe2b0094126d1ca'
old = (p/'history/budget_4m_20260910/runner.py').read_text()
new = (p/'runner.py').read_text()
old_line = "f'walker_walk_seed{seed}_4m'"
new_line = "f'walker_walk_seed{seed}_{PLAN[\"environment_steps_per_seed\"]//1000000}m'"
assert old.replace(old_line, new_line) == new, 'Unexpected training implementation change'
previous = json.loads((p/'history/budget_4m_20260910/configs/plan.json').read_text())
current = runner.PLAN
assert {k:v for k,v in current.items() if k not in ['environment_steps_per_seed','decisions_per_seed','budget_revision']} == {k:v for k,v in previous.items() if k not in ['environment_steps_per_seed','decisions_per_seed','budget_revision']}
assert current['seeds'] == [1,2,3] and current['environment_steps_per_seed'] == 1000000
for seed in [1,2,3]:
    assert runner.configuration(seed,False).steps == 500000
assert runner.configuration(1,True).steps == 3000
assert finalize.BUDGET == 1000000 and finalize.LABEL == '1m'
ref = [x for x in csv.DictReader((p/'reference/walker-walk.csv').open()) if int(x['step']) == 1000000]
assert len(ref) == 3
frozen = json.loads((p/'artifacts/witness_20260910_075830/sources.json').read_text())
for n in ['runtime.sh','run.sh','configs/env-spec.json']:
    assert hashlib.sha256((p/n).read_bytes()).hexdigest() == frozen[n]
for n in ['runner.py','finalize.py']:
    ast.parse((p/n).read_text())
print('DMC_BUDGET_1M_CHECK_PASS',json.dumps({'environment_steps_per_seed':1000000,'decisions_per_seed':500000,'seeds':[1,2,3],'training_core_unchanged':True,'runtime_unchanged':True,'reference_returns':ref}))
