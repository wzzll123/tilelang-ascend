#!/usr/bin/env bash
# Same-card serial A/B using the original designer profiler, after precision.
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
compiler=/home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128
baseline=/home/wenzhongzhen/tilelang-ascend-wt/fa128-v06-baseline-e471
shared=/home/wenzhongzhen/tilelang-ascend
plugin=/home/wenzhongzhen/kernel-tool/cannbot-skills-0817/plugins-community/tilelang2ascendc-ops-generator
task=$plugin/workflows/templates/archive_tasks/flash_attention
profiler=$plugin/skills/tilelang2ascend-tilelang-designer/scripts/perf_tilelang.py
python=/tmp/a2a3-auto-schedule-fa128-venv.JsRZee/bin/python
cases=(0 2 6 7 10 12 13 14 15 16 36 37 41 42)
# Never launch alongside the precision process. Call only after its exit0.
"$python" -c '
import json
from pathlib import Path
for case in [0,2,6,7,10,12,13,14,15,16,36,37,41,42]:
    p=Path(f"/tmp/fa128-score-2c83b27d-case{case}-stability20.json")
    r=json.loads(p.read_text())
    assert r["ok"] and r["device"]=="npu" and r["check_finite"] and r["repeat_checks"]==20, (case,r)
    assert "all 20 repeated launches matched" in r["comparison"], (case,r)
print("All fourteen repeated finite/precision gates passed",flush=True)
'
test "$(git -C "$compiler" rev-parse HEAD)" = 2c83b27dc1763f908f7b1a2412ad6ad41532186f
test -z "$(git -C "$compiler" status --porcelain)"
test "$(git -C "$baseline" rev-parse HEAD)" = e47171e32f6e64ff613cc6f398ec4c32a7c76d9f
test "$(sha256sum "$compiler/build-auto-schedule/libtilelang_module.so" | cut -d' ' -f1)" = 4ce6649be570c447d58f5fa1e5b236eb1a1473a6a9f88a010de4e4578eddd98b
test "$(sha256sum "$baseline/build-baseline-deps/libtilelang_module.so" | cut -d' ' -f1)" = 6c5740dca57791696c5165a01942d8163d4078cce5acd451b67499f095bb1c21
test "$(sha256sum "$shared/build/tvm/libtvm.so" | cut -d' ' -f1)" = 2df7999bbed69490640f5bc50166a5a650ac543bdea22514878e86ae31ccd36e
test "$(sha256sum "$task/design/tile_level/ladder/fa_v06.py" | cut -d' ' -f1)" = b345fc6813b7ede3d710e4fb89dd45ee51a64674009987b2433c44bffe606f77
test "$(sha256sum "$task/design/tile_level/ladder/fa_auto128.py" | cut -d' ' -f1)" = bd5bbaa5aba60dc802129b071da0708fcea256fb5ab7c1182e3ec737136ac164
test "$(sha256sum "$profiler" | cut -d' ' -f1)" = c54714b2875f19c9610a91e0dff4d84936aaa6ab1e99f3e400913be36cd6b70e
: "${SCORE_CACHE_DIR:?Set the pinned score build isolated cache}"
export ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_LAUNCH_BLOCKING=1
export TVM_LIBRARY_PATH=$shared/build/tvm
common_lib=$shared/build/tvm:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/python3.12.13/lib
for round in 1 2 3; do
    for case_id in "${cases[@]}"; do
        for implementation in v06 auto128; do
            if test "$implementation" = v06; then
                checkout=$baseline
                native=$baseline/build-baseline-deps
                cache=/tmp/fa128-v06-cache-e471-binary2d176
            else
                checkout=$compiler
                native=$compiler/build-auto-schedule
                cache=$SCORE_CACHE_DIR
            fi
            export PYTHONPATH=$checkout:$shared/3rdparty/tvm/python
            export TILELANG_LIBRARY_PATH=$native TILELANG_CACHE_DIR=$cache
            export LD_LIBRARY_PATH=$native:$common_lib FA_STEP=$implementation
            cd "$checkout"
            report_dir=/tmp/fa128-score-2c83b27d-round${round}-${implementation}
            test ! -e "$report_dir/case_${case_id}.json"
            date -u
            npu-smi info
            sha256sum "$profiler" "$task/model_new_tilelang.py"
            "$python" -c 'import tilelang,tvm,torch,z3; print("tilelang",tilelang.__file__,tilelang._LIB_PATH); print("tvm",tvm.__file__,tvm._ffi.base._LIB._name); print("torch",torch.__version__,"z3",z3.get_version_string())'
            timeout 600 "$python" "$profiler" "$task" --case "$case_id" \
                --warmup 3 --repeat 20 --no-boost --no-sol --json-dir "$report_dir"
        done
    done
done
