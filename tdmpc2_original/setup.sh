#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/../local-settings.sh"
export PIP_CACHE_DIR="$ROOT/cache/pip" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
mkdir -p "$ROOT"/{vendor_deps,cache,logs,reports,native,tmp}
(cd "$ROOT/cache" && apt-get download libglew-dev=2.1.0-4 && dpkg-deb -x libglew-dev_2.1.0-4_amd64.deb "$ROOT/native/sysroot")
"$HAB_PYTHON" -m pip install --no-deps --target "$ROOT/vendor_deps/build" pip==23.0.1 setuptools==65.5.0 wheel==0.38.4
PYTHONPATH="$ROOT/vendor_deps/build" "$HAB_PYTHON" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$ROOT/cache/wheels" gym==0.21.0
PYTHONPATH="$ROOT/vendor_deps/build" "$HAB_PYTHON" -m pip install --no-deps --target "$ROOT/vendor_deps/runtime" "$ROOT"/cache/wheels/gym-0.21.0-*.whl mujoco-py==2.1.2.14 Cython==0.29.37 fasteners==0.19 glfw==2.9.0 patchelf==0.17.2.0 cffi==1.17.1 pycparser==2.22
ROOT="$ROOT" "$HAB_PYTHON" - <<'PY'
import os,urllib.request,tarfile,pathlib,hashlib,json
p=pathlib.Path(os.environ['ROOT']); rows=[]
f=p/'vendor_deps/runtime/mujoco_py/builder.py'
s=f.read_text(); original=hashlib.sha256(f.read_bytes()).hexdigest()
if 'MUJOCO_PY_NVIDIA_LIB_DIR' not in s:
 s=s.replace('def get_nvidia_lib_dir():', 'def get_nvidia_lib_dir():\n    if os.environ.get("MUJOCO_PY_NVIDIA_LIB_DIR"):\n        return os.environ["MUJOCO_PY_NVIDIA_LIB_DIR"]')
 f.write_text(s)
(p/'reports/mujoco-build-patch.json').write_text(json.dumps({'before_sha256':original,'after_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'reason':'project-local NVIDIA library discovery only'},indent=2))
for name,url,target in [('mujoco210.tar.gz','https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz',p/'native'),('metaworld.tar.gz','https://codeload.github.com/Farama-Foundation/Metaworld/tar.gz/04be337a12305e393c0caf0cbf5ec7755c7c8feb',p/'vendor_deps')]:
 f=p/'cache'/name
 if not f.exists():
  temp=f.with_suffix('.partial');urllib.request.urlretrieve(url,temp);temp.rename(f)
 with tarfile.open(f) as t:
  for x in t.getmembers():
   dest=(target/x.name).resolve()
   if not dest.is_relative_to(target.resolve()) or x.issym() or x.islnk():raise RuntimeError('unsafe archive path')
  t.extractall(target)
 rows.append({'file':name,'source':url,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(p/'reports/install-sources.json').write_text(json.dumps(rows,indent=2))
print('ORIGINAL_ENV_INSTALLED_NOT_YET_VALIDATED')
PY
