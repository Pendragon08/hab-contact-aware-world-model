# Warm-reused environments for a six-task expansion

Provider: ouyanghao@10.147.17.32. Project: /data/code/ouyanghao/oyh/hab-contact-aware-world-model. Expansion: six_tasks_20260911.

Cheetah Run: GPU0, CUDA_VISIBLE_DEVICES=0, obs17/action6, 500 decisions per episode, repeat2. Environment spec canonical hash cba67740: dm-control1.0.8/MuJoCo2.3.1/NumPy1.23.5/Gym0.21.0. The native suite is cheetah/run. Native/wrapped observation and reward traces must agree within 1e-10.

Stack Cube: GPU1, CUDA_VISIBLE_DEVICES=1, renderer pci:65, obs55/action4, 100 decisions per episode, repeat2. Environment spec canonical hash eff3d0dd: ManiSkill2 0.4.1/SAPIEN2.2.1/NumPy1.23.5/Gym0.21.0. Native StackCube-v0, dense reward, pd_ee_delta_pos, unchanged success (cubeA on cubeB, static and not grasped). Synthetic positive/negative predicate checks are separate from scores. Vulkan may retain <=8MiB graphics-only allocation onGPU0; all compute must be onGPU1.

Both reuse the prior read-only dependencies and Python3.10.21/PyTorch2.5.1+cu121/TorchRL0.6. Environments are unchanged, no installation/rebuild. Official TD-MPC2 core commit57158282b46ebc5c329c5be9cfe2b0094126d1ca. Own writable caches/artifacts. Runtime and reduced-budget differences from full historical paper reproduction persist. New-task formal budget3seeds x1M native training steps; eval steps separate. Entry modifies only resource co-residency checks, not the learner. At most3compute jobs perGPU,12GiB free GPU and15GiB disk reserved. User explicitly requested expansion beyond2perGPU.

Existing Walker Walk/Walker Run onGPU0 and Pick Cube/Lift Cube onGPU1 must continue. Validation waits for the non-invasive five-minute four-task baseline before running exactly two short witnesses. Formal successor scheduler deployment is a separate root action after validation.
