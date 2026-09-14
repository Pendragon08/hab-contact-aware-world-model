# Warm-reused task environments

Provider ouyanghao@10.147.17.32. Root /data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911.

GPU0 new task Walker Run, official walker-run, obs24/action6, episode500decisions/action-repeat2. Warm environment hashcba67740: dm-control1.0.8/MuJoCo2.3.1/NumPy1.23.5/Gym0.21.0.

GPU1 new task Lift Cube, official LiftCube-v0/lift-cube, obs42/action4, episode100decisions/action-repeat2, native dense reward/pd_ee_delta_pos. Warm environment hasheff3d0dd: ManiSkill2 0.4.1/SAPIEN2.2.1/NumPy1.23.5/Gym0.21.0. Native LiftCube success requires cube height>=goal_height+cube_half_size[2] and robot static; native task implementation remains unchanged. Renderer pci:65, CUDA_VISIBLE_DEVICES=1, GPU0 retains4MiB graphics-only context per SAPIEN process.

Both use borrowed Python3.10.21/PyTorch2.5.1+cu121/TorchRL0.6 read-only and official TD-MPC2 commit57158282b46ebc5c329c5be9cfe2b0094126d1ca. Own caches, configs, runner copies and artifacts. No dependency installation/rebuild. Resource guard allows one authorized existing co-resident per GPU, because the user explicitly requested additional tasks and earlier two-per-GPU throughput trials passed. Reserve12GiB GPU free and15GiB disk.

Existing Walker Walk/GPU0 and Pick Cube/GPU1 training must keep running. Current original seed2 PIDs1819620/1824540 at last preflight. Formal new tasks each have3seeds and1M native training steps per seed; only the independent short witness is allowed in EXECUTION.md. Formal manager handoff is a separate root action after witness passes.
