# Independent bounded concurrency trial

The user explicitly authorized two training processes on each GPU. Existing GPU0 DMControl PID1789362 and GPU1 ManiSkill2 PID1795158 must remain running. This is warm reuse of env hashes cba67740 and eff3d0dd; no dependency or core algorithm changes. Earlier idle-device requirements are replaced only for this user-authorized co-residency test. The added seed2 runs are disposable performance pilots, never formal1M results.

Run the following PowerShell invocation exactly once. Do not retry, modify files, inspect implementation, or launch anything else:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/launch.sh"
```

The command should exit0 and list a detached screen named tdmpc2_throughput_2pergpu. The controller runs in the background. Use read-only SSH calls to inspect controller.log and JSON files under the same remote directory, at intervals of roughly30–60seconds, until result.json or failure.json appears. Do not resubmit if a connection drops; inspect current artifacts instead.

Expected procedure: retain preceding300seconds of original training logs as baseline; add one Walker Walk/GPU0 and one Pick Cube/GPU1 seed2 pilot. Keep model5M, batch256, horizon3, 500k-entry replay capacity,4CPU library threads and original simulation/rendering settings. Added pilots have a24000-native-step cap, and finish at a full episode boundary once the300second measurement ends. Comparison begins only after each pilot has reached3501decisions, excluding seed pretraining/initial evaluation. Regular throughput calculations exclude update intervals crossing scheduled50k-decision evaluations; separate wall-window rates retain pauses.

After measurement the new processes save their final checkpoint and run10evaluation episodes, then exit. Expected total10–20minutes, hard controller timeout30minutes. Original processes must still be alive and their frozen-source hashes unchanged. Data disk reserve15GiB; freeGPU memory12GiB per card. Check result.json completed=true, both pilot completions show real optimizer updates, all_five_modules_changed=true, and actual counts remain under24000native steps. Read baseline.json, measurement.json and samples.jsonl; independently check aggregate per-GPU speed factors and evidence that four compute processes ran during the measurement. Report divergences, failures or limited comparability; do not create an approval file. This test does not establish the maximum useful concurrency or task performance.
