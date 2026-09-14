# DMControl original-method reproduction

- User requested DMControl reproduction on 2026-09-10 after checking GPU0 availability. This experiment uses GPU0; the existing HAB experiment remains on GPU1.
- Reserve at least 12 GiB free on GPU0. Never stop another user's process or modify the running HAB experiment.
- Reuse tdmpc2_original/upstream read-only at commit 57158282b46ebc5c329c5be9cfe2b0094126d1ca. Verify the upstream manifest.
- Own all new dependencies, caches, logs, results and failed attempts under this directory. Borrow the existing Python/PyTorch installation read-only.
- Match Walker Walk paper task, reward, action repeat, model and optimizer. User reduced per-seed budget to 1M environment steps on 2026-09-10; retain three seeds. Compare against paper at the same 1M budget, not its 4M final result. Document runtime differences and the exact-budget loop-bound correction.
- No success-rate claims for DMControl: report raw episode return, then return/10 if normalized to 0–100.
