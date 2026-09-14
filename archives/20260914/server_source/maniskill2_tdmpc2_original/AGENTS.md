# Original-paper ManiSkill2 reproduction

- User explicitly replaced HAB with original-paper ManiSkill2 on 2026-09-10. Use physical GPU1. Do not stop, migrate, or change DMControl on GPU0.
- Task: PickCube-v0 / pick-cube, using ManiSkill2 0.4.1 and SAPIEN 2.2.1, official TD-MPC2 environment wrapper and core at commit 57158282b46ebc5c329c5be9cfe2b0094126d1ca.
- Budget: 1M environment steps per seed = 500k action decisions at action repeat2. Seeds1/2/3 sequentially. Report same-budget paper comparison, not complete4M replication.
- Borrow interpreter and pinned upstream sources read-only. All new dependencies/caches/logs/results under this directory. Preserve failed attempts and existing HAB data.
- wandb:false. Do not touch other users, ACS, base, ecg-lfm or running DMControl files.
- Preserve at least12GiB free on GPU0 and15GiB disk. Never silently switch simulation/backend/task if legacy version fails.
