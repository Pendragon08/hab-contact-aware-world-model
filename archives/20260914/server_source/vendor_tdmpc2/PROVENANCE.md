# TD-MPC2 vendored implementation

Source: https://github.com/haosulab/ManiSkill/tree/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/examples/baselines/tdmpc2

Copied `tdmpc2.py`, `common/`, and `config.yaml` from the pinned ManiSkill TD-MPC2 baseline. The upstream repository Apache-2.0 license is preserved in `LICENSE`.
Original algorithm: https://github.com/nicklashansen/tdmpc2 and https://www.tdmpc2.com/ (ICLR 2024).

The simulation installation uses a different, HAB-compatible ManiSkill revision (see environment spec). Vendoring the algorithm modules avoids importing the incompatible simulator version. The custom state adapter, training loop and episode-bounded replay sampler are in `scripts/`, and are not the upstream HAB training recipe.

No changes to the model or optimizer equations in the copied TD-MPC2 modules are intended. Exact file checksums are recorded with experiment provenance.
