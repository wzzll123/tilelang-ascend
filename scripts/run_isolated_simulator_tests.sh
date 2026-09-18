#!/usr/bin/env bash
# Run simulator tests from an isolated worktree against a shared, built runtime.
#
# A Git worktree contains Python sources but does not duplicate TVM submodules
# or build artifacts.  Loading it with only `source set_env.sh` can therefore
# accidentally resolve CANN's incompatible libtvm.so.  Keep the two roots
# explicit: TEST_ROOT supplies Python; RUNTIME_ROOT supplies libtilelang and
# libtvm built from the matching shared checkout.
set -euo pipefail

TEST_ROOT="${TILELANG_TEST_ROOT:-$(pwd)}"
RUNTIME_ROOT="${TILELANG_RUNTIME_ROOT:-/home/wenzhongzhen/tilelang-ascend}"

required=(
  "${TEST_ROOT}/tilelang/__init__.py"
  "${RUNTIME_ROOT}/3rdparty/tvm/python/tvm/__init__.py"
  "${RUNTIME_ROOT}/build/libtilelang_module.so"
  "${RUNTIME_ROOT}/build/tvm/libtvm.so"
)
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing isolated-simulator runtime component: ${path}" >&2
    exit 2
  fi
done

export ACL_OP_INIT_MODE=1
export PYTHONPATH="${TEST_ROOT}:${RUNTIME_ROOT}/3rdparty/tvm/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${RUNTIME_ROOT}/build:${RUNTIME_ROOT}/build/tvm${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec python3 -m pytest "$@"
