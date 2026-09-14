# Independent validation of two additional tasks

Read only the run-experiment skill, ENV_LEDGER.md and this document as input. Execute this PowerShell invocation once, without modifying files, inspecting implementation, retrying the invocation, or launching formal training:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/validate.sh"
```

The invocation runs two short witnesses concurrently, one per GPU, alongside the two existing training processes. Follow the same tool session to completion; never relaunch after a disconnection. Expected exit0 and ADDITIONAL_TASK_WITNESSES_PASS, WITNESS_EXIT_CODES dmc=0 ms2=0. It normally takes several minutes. On failure report the exact logs and do not fix or retry.

Read the resulting artifacts independently. Logs: validation/dmc.log and validation/ms2.log. Latest output folders: walker_run/artifacts/witness_<timestamp>/ and lift_cube/artifacts/witness_<timestamp>/. Read complete.json, resolved-config.yaml, sources.json and relevant runtime logs. Check task identities, state/action dimensions, GPU evidence, finite updates, all_five_modules_changed and nonempty final checkpoint/videos. Verify active original training processes still exist; seed transition is allowed if original queue naturally advances.

Walker Run expected:6000native training steps/3000decisions/2999optimizer updates,2000native evaluation steps, obs24/action6, episode500, discount0.99. Independent native-vs-official action-repeat witness over500decisions has observation/reward error<=1e-10 and384x384x3render.

Lift Cube expected:2400native training steps/1200decisions/1199updates,400native evaluation steps, obs42/action4, episode100, discount0.95. Repeated seed trace error<=1e-6, native reward/control, nonemptyRGB and synthetic native success positive=true/negative=false. Synthetic placement tests are not learned-policy scores. Physical compute GPU1; GPU0 may contain at most8MiB graphics-only context for this process.

The source snapshots freeze each task's runner, runtime, run.sh, plan and environment spec. Additional launcher provenance is in provenance/*.json and freezes shared_entry.py/shared_run.sh/manifest.json. Report completion paths and these entry hashes for the root to recheck. Do not write an approval marker. Short witness success rates do not establish final task performance. This validation does not launch or certify the separate formal queue manager.
