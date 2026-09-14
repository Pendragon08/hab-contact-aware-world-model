# Warm-reuse ledger for throughput trial

Provider ouyanghao@10.147.17.32; root /data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial.

Uses existing verified environments unchanged: DMControl cba67740, ManiSkill2 eff3d0dd. Borrowed Python3.10.21/PyTorch2.5.1+cu121/TorchRL0.6. Original DMControl MuJoCo2.3.1/dm-control1.0.8; original ManiSkill2 0.4.1/SAPIEN2.2.1; NumPy1.23.5/Gym0.21.0. Own cloned runner/config/cache/artifact directories; original dependencies and source files read-only.

RTX A6000 GPU0 reserved for Walker Walk; GPU1 for Pick Cube. GPU1 rendering pci:65, CUDA_VISIBLE_DEVICES=1. SAPIEN retains4MiB of graphics-only context on GPU0 per process. User explicitly authorized sharing each device between exactly two training jobs; never alter protected original jobs. Snapshot before preparation: GPU0 742MiB, GPU1 924MiB, disk29GiB free,251GiB RAM,10physicalCPU cores/20threads.

Original environment witnesses passed independently earlier this session. This fresh document-following pass validates the bounded parallel invocation and resulting artifacts; no environment rebuild. Prepared manifest records original/clone source hashes. See EXECUTION.md for the exact single invocation and expected evidence.
