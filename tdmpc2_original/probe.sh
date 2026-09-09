#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
"$HAB_PYTHON" - <<'PY'
import mujoco_py, numpy,torch,sys
print('VERSIONS',sys.version,numpy.__version__,torch.__version__,mujoco_py.__version__,flush=True)
from omegaconf import OmegaConf
from envs.metaworld import make_env
c=OmegaConf.create({'task':'mw-pick-place','seed':1,'obs':'state'})
e=make_env(c);o=e.reset();print('ORIGINAL_ENV_RESET',o.shape,flush=True)
for i in range(100):o,r,d,info=e.step(numpy.zeros(4))
assert d and o.shape==(39,)
print('ORIGINAL_CPU_PROBE_PASS',info,flush=True)
PY
