#!/usr/bin/env bash
# Local compiler-only tests; no Torch import, simulator gate or NPU launch.
set -euo pipefail
checkout=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
primary=/Users/wzz/Desktop/Research/kernel-tool/cannbot-skills/tilelang-ascend
python=/Users/wzz/miniconda3/bin/python
z3_site=/private/tmp/ascend-schedule-solver-venv/lib/python3.9/site-packages
test -f "$checkout/build-auto-schedule/libtilelang_module.dylib"
test -d "$primary/3rdparty/tvm/python"
test -d "$z3_site/z3"
export PYTHONPATH=$checkout:$primary/3rdparty/tvm/python
export TILELANG_LIBRARY_PATH=$checkout/build-auto-schedule
export TVM_LIBRARY_PATH=$primary/3rdparty/tvm/build
cd "$checkout"
if test "$#" -eq 0; then
    set -- -q \
        testing/python/transform/test_ascend_schedule_stage_skew.py \
        testing/python/transform/test_ascend_schedule_uniform_clock.py \
        testing/python/transform/test_ascend_schedule_phases.py \
        testing/python/transform/test_ascend_schedule_l1_prefetch.py \
        testing/python/transform/test_ascend_schedule_materializer.py \
        testing/python/transform/test_ascend_schedule_buffer_versions.py
    set -- "$@" testing/python/transform/test_ascend_schedule_cv_stage_waves.py
fi
# Append (not prepend) the old isolated site directory: its setuptools58 is
# incompatible with Python3.13 and must not shadow Conda's installed packages.
exec "$python" -c 'import sys; sys.path.append(sys.argv.pop(1)); import tilelang,tvm,z3,pytest; print("tilelang",tilelang.__file__,tilelang._LIB_PATH); print("tvm",tvm.__file__,tvm._ffi.base._LIB._name); print("z3",z3.get_version_string()); raise SystemExit(pytest.main(sys.argv[1:]))' "$z3_site" "$@"
