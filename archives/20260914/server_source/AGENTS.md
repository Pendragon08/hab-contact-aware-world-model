# Experiment constraints

## Current deployment (2026-09-11)

The user superseded the historical GPU1-only and two-process policies: run three task streams per GPU, preserving existing training. GPU0: Walker Walk, Walker Run, Cheetah Run. GPU1: Pick Cube, Lift Cube, Stack Cube. Each task uses seeds 1/2/3 sequentially, 1M native training steps per seed. The active scheduler is `six_tasks_20260911/manager.py`, screen `tdmpc2_six_tasks_1m`; inspect `six_tasks_20260911/status.py` and `state.json` for all six tasks. The old four-task scheduler is retired; do not restart it or resume its inherited queue shells. Sources of existing learners are frozen and must not be edited while running. See `six_tasks_20260911/README.md`, `handoff.json`, and `launch-report.md` for evidence and resource/throughput details. Preserve12GiB GPU free and15GiB disk. No external notifications unless separately authorized.

## Historical HAB deployment constraints

- Use GPU1 only; leave GPU0 untouched, user requires at least 12 GiB free.
- Do not stop other users' processes. All experiments preserve attempts and unmodified HAB success conditions.
- Remote root: `/data/code/ouyanghao/oyh/hab-contact-aware-world-model` via `ouyanghao@10.147.17.32`.
- Borrowed interpreter read-only: `/data/code/ouyanghao/oyh/world-model-risk-planning/.env/bin/python`.
- Reuse pinned sources, vendor dependencies and assets read-only from the existing HAB demo. Mutable caches and results belong to this project.
- Do not touch ACS, base, ecg-lfm or old experimental results. wandb: false.
- User explicitly authorizes creation of this GitHub repository and uploading project code, experimental data and results. Repository defaults private.
- No credentials, machine-specific auth files or environments in Git. Full generated raw data go into Release assets with checksums; third-party assets retain upstream terms and provenance.
- Do not claim untrained modules, pilot results or development results as final research results.
