# Independent agent-follows-doc validation

Read the run-experiment skill and ENV_LEDGER.md. Run this exact command ONCE from PowerShell:

```
ssh -o BatchMode=yes -o ConnectTimeout=25 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/six_tasks_20260911/validate.sh"
```

It may wait for completion of a five-minute read-only four-task baseline, then runs the two short seeded GPU witnesses concurrently alongside current training. Expected exit0, `WITNESS_EXIT_CODES cheetah-run=0 stack-cube=0`, `SIX_TASK_WITNESSES_PASS`. Do not modify files, retry validation, install dependencies, stop existing processes, or launch formal training. On failure preserve/report actual logs and stop. You may inspect outputs via read-only SSH commands and report doc-vs-reality divergence.

Outputs: six_tasks_20260911/{cheetah_run,stack_cube}/artifacts/witness_*/complete.json, resolved-config.yaml, sources.json, runtime.json, updates.jsonl, final.pt, videos/*.mp4. Logs: validation/{cheetah-run,stack-cube}.log. Record exact artifact paths and entry.py/run.sh/registry.json hashes. Confirm finite logged updates, five modules changed, seeded native trace/render checks, correct GPU binding and no GPU0 CUDA for StackCube. Original training processes must remain live or naturally complete (do not mistake normal seed progression for interruption).

Cheetah witness expects6000 native train steps/3000decisions/2999optimizer updates,2000native evalsteps,obs17/action6,384x384x3render. Stack witness expects2400native train/1200decisions/1199updates,400native evalsteps,obs55/action4,384x640x3render,synthetic successpositiveTrue/negativeFalse. Discount.99/.95 is asserted by each runner but not serialized as a resolved-config field. Witness scores certify execution, not performance. Formal3seed/1Mbudget execution and scheduler handoff are outside this witness's certification.
