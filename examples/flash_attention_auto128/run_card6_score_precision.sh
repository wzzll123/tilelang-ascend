#!/usr/bin/env bash
# Original designer verification only; no simulator prerequisite or device reset.
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
compiler=/home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128
shared=/home/wenzhongzhen/tilelang-ascend
plugin=/home/wenzhongzhen/kernel-tool/cannbot-skills-0817/plugins-community/tilelang2ascendc-ops-generator
task=$plugin/workflows/templates/archive_tasks/flash_attention
verifier=$plugin/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py
python=/tmp/a2a3-auto-schedule-fa128-venv.JsRZee/bin/python
test "$(git -C "$compiler" rev-parse HEAD)" = 2c83b27dc1763f908f7b1a2412ad6ad41532186f
test -z "$(git -C "$compiler" status --porcelain)"
test "$(sha256sum "$compiler/build-auto-schedule/libtilelang_module.so" | cut -d' ' -f1)" = 4ce6649be570c447d58f5fa1e5b236eb1a1473a6a9f88a010de4e4578eddd98b
test "$(sha256sum "$shared/build/tvm/libtvm.so" | cut -d' ' -f1)" = 2df7999bbed69490640f5bc50166a5a650ac543bdea22514878e86ae31ccd36e
test "$(sha256sum "$verifier" | cut -d' ' -f1)" = fe881e92db75fb3a8e5f13f6e47fe5124af72c269ad117a19ddbdd231858c1d2
test "$(sha256sum "$task/design/tile_level/ladder/fa_auto128.py" | cut -d' ' -f1)" = bd5bbaa5aba60dc802129b071da0708fcea256fb5ab7c1182e3ec737136ac164
export PYTHONPATH=$compiler:$shared/3rdparty/tvm/python
export TILELANG_LIBRARY_PATH=$compiler/build-auto-schedule TVM_LIBRARY_PATH=$shared/build/tvm
export LD_LIBRARY_PATH=$compiler/build-auto-schedule:$shared/build/tvm:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/python3.12.13/lib
export ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_LAUNCH_BLOCKING=1 FA_STEP=auto128
# Required: use only the fresh cache created for the pinned committed build.
: "${TILELANG_CACHE_DIR:?Set a new isolated cache directory for this build}"
cd "$compiler"
"$python" -c 'import tilelang,tvm,torch,z3; print("tilelang",tilelang.__file__,tilelang._LIB_PATH); print("tvm",tvm.__file__,tvm._ffi.base._LIB._name); print("torch",torch.__version__,"z3",z3.get_version_string())'
# Small shapes and a causal case first, then all remaining fixed IDs.
for case_id in 0 42 10 2 6 7 12 13 14 15 16 36 37 41; do
    report=/tmp/fa128-score-2c83b27d-case${case_id}-stability20.json
    test ! -e "$report"
    date -u
    npu-smi info
    timeout 300 "$python" "$verifier" "$task" --case "$case_id" \
        --check-finite --repeat-checks 20 --json "$report"
    "$python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["ok"] and r["device"]=="npu" and r["check_finite"] and r["repeat_checks"]==20; assert "all 20 repeated launches matched" in r["comparison"]; print("verified",sys.argv[1],flush=True)' "$report"
done
