#!/usr/bin/env bash
# Run only after the three-round performance process has terminated.
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
compiler=/home/wenzhongzhen/tilelang-ascend-wt/auto-schedule-fa128
baseline=/home/wenzhongzhen/tilelang-ascend-wt/fa128-v06-baseline-e471
shared=/home/wenzhongzhen/tilelang-ascend
plugin=/home/wenzhongzhen/kernel-tool/cannbot-skills-0817/plugins-community/tilelang2ascendc-ops-generator
task=$plugin/workflows/templates/archive_tasks/flash_attention
python_bin=/tmp/a2a3-auto-schedule-fa128-venv.JsRZee/bin
test "$(git -C "$compiler" rev-parse HEAD)" = 2c83b27dc1763f908f7b1a2412ad6ad41532186f
test -z "$(git -C "$compiler" status --porcelain)"
test "$(sha256sum "$compiler/build-auto-schedule/libtilelang_module.so" | cut -d' ' -f1)" = 4ce6649be570c447d58f5fa1e5b236eb1a1473a6a9f88a010de4e4578eddd98b
test "$(sha256sum "$baseline/build-baseline-deps/libtilelang_module.so" | cut -d' ' -f1)" = 6c5740dca57791696c5165a01942d8163d4078cce5acd451b67499f095bb1c21
test "$(sha256sum "$task/design/tile_level/ladder/fa_auto128.py" | cut -d' ' -f1)" = bd5bbaa5aba60dc802129b071da0708fcea256fb5ab7c1182e3ec737136ac164
: "${SCORE_CACHE_DIR:?Set the pinned score build cache}"
export ASCEND_RT_VISIBLE_DEVICES=6 ASCEND_LAUNCH_BLOCKING=1
export TVM_LIBRARY_PATH=$shared/build/tvm PATH=$python_bin:$PATH
command -v python3
command -v msprof
common_lib=$shared/build/tvm:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/python3.12.13/lib
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
    tag=score2c83b27d_${implementation}_case14
    test ! -e "$task/analysis/raw/msprof_$tag"
    cd "$checkout"
    date -u
    npu-smi info
    sha256sum "$plugin/skills/tilelang2ascend-tilelang-designer/scripts/msprof_tilelang.sh" \
        "$plugin/skills/tilelang2ascend-tilelang-designer/scripts/run_once_tilelang.py"
    "$python_bin/python" -c 'import tilelang,tvm; print("tilelang",tilelang.__file__,tilelang._LIB_PATH); print("tvm",tvm.__file__,tvm._ffi.base._LIB._name)'
    timeout 300 bash "$plugin/skills/tilelang2ascend-tilelang-designer/scripts/msprof_tilelang.sh" \
        "$task" "$tag" --case 14 --warmup 5 --repeat 20
done
