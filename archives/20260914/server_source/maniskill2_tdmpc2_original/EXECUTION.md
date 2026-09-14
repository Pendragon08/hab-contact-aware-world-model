# Independent ManiSkill2 entrypoint validation

Run this command verbatim once from PowerShell. Do not modify files, repair failures, retry the command, or start formal training. Do not inspect the runner implementation. Read-only output inspection is allowed.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/maniskill2_tdmpc2_original/run.sh witness"
```

If the tool returns a running session, continue reading that same session to completion. Expected exit0 and terminal sentinel `MS2_ORIGINAL_WITNESS_PASS` with JSON evidence. Estimated duration several minutes; preserve any failure and report it unchanged.

This runs original TD-MPC2 PickCube-v0 with model5M on physical GPU1. GPU0 DMControl must remain running. SAPIEN rendering is explicitly pci:65. Driver enumeration leaves about4MiB of graphics-only context on GPU0; evidence must show no GPU0 compute and <=8MiB context, with GPU1 compute/graphics allocation.

Expected simulation witness: observation51, action4, 100decisions/200native steps per episode, repeated seeded traces error<=1e-6, native dense reward/pd_ee_delta_pos control, nonempty RGB image. Native success predicate must return true for a synthetic goal/static configuration and false after moving the object away. These synthetic checks are not learned-policy scores.

Expected actual optimization: 1200training decisions=2400native steps, 1199optimizer updates; 400native evaluation steps (one episode initially and one finally), all five modules changed, finite metrics. Additional simulation validation400native steps is separate. Final success may be0 after such a short run; this is an execution check, not a performance claim.

Output: remote `maniskill2_tdmpc2_original/artifacts/witness_<UTC timestamp>/`. Separately read the latest created complete.json and check mode, counts, module changes and environment evidence; confirm two videos,12training trajectories,2evaluation trajectories and nonempty final.pt. A failed run has no valid completion marker. Report exact artifact path, exit code and every divergence. Do not write an approval marker or modify this document.

Formal training is separately configured for seeds1/2/3 sequentially,1million native environment steps per seed. Never launch it during this witness.
