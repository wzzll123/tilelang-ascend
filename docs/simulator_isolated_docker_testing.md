# Isolated Docker simulator tests

When testing a patch in a Git worktree inside the shared CANN Docker container,
do not run only `source set_env.sh`. A worktree has the Python sources but
normally lacks TVM's submodule checkout and the built shared libraries. Python
may then load CANN's unrelated `libtvm.so`, which fails with an undefined TVM
symbol or produces an ABI mismatch.

Use the repository launcher instead. It separates the code being tested from
the matching prebuilt runtime:

```bash
cd /tmp/tilelang-ascend-<commit>
TILELANG_RUNTIME_ROOT=/home/wenzhongzhen/tilelang-ascend \
  scripts/run_isolated_simulator_tests.sh \
  testing/python/simulator/test_scheduler.py -q
```

The launcher requires and wires exactly these components:

| Purpose | Location |
| --- | --- |
| Patched Python sources | `TILELANG_TEST_ROOT` (defaults to current directory) |
| TVM Python package | `$TILELANG_RUNTIME_ROOT/3rdparty/tvm/python` |
| TileLang module | `$TILELANG_RUNTIME_ROOT/build/libtilelang_module.so` |
| TVM shared library | `$TILELANG_RUNTIME_ROOT/build/tvm/libtvm.so` |

It exits before pytest if any component is missing. This lets multiple agents
test independent worktrees without modifying the shared checkout.
