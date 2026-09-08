# Experiment constraints

- Use GPU1 only; leave GPU0 untouched, user requires at least 12 GiB free.
- Do not stop other users' processes. All experiments preserve attempts and unmodified HAB success conditions.
- Remote root: `/data/code/ouyanghao/oyh/hab-contact-aware-world-model` via `ouyanghao@10.147.17.32`.
- Borrowed interpreter read-only: `/data/code/ouyanghao/oyh/world-model-risk-planning/.env/bin/python`.
- Reuse pinned sources, vendor dependencies and assets read-only from the existing HAB demo. Mutable caches and results belong to this project.
- Do not touch ACS, base, ecg-lfm or old experimental results. wandb: false.
- User explicitly authorizes creation of this GitHub repository and uploading project code, experimental data and results. Repository defaults private.
- No credentials, machine-specific auth files or environments in Git. Full generated raw data go into Release assets with checksums; third-party assets retain upstream terms and provenance.
- Do not claim untrained modules, pilot results or development results as final research results.
