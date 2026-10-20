# TVM submodule patches

Minimal local patches applied on top of the **pinned** `3rdparty/tvm` submodule
commit. `apply_tvm_patches.sh` in this directory applies every `tvm_*.patch`
right after `git submodule update --init --recursive`, and is invoked by every
build entry point — `install_ascend.sh`, `build_wheel_ascend.sh`, and `setup.py`
(i.e. `USE_ASCEND=true pip install -e .`) — so every build path (including CI)
picks them up. Application is:

- **idempotent** — an already-applied patch is detected (`git apply --reverse
  --check`) and skipped, so re-running install / incremental builds is safe;
- **fatal on failure** — if a patch cannot apply (e.g. the pinned tvm was bumped
  and the context no longer matches) the install aborts instead of silently
  building an unpatched TVM.

Files must be named `tvm_*.patch` to be picked up.

## Patches

- **`tvm_backtrace_symbol_name_length.patch`** — `BacktraceSyminfoCallback`
  builds the demangling input from the NUL-terminated `symname` instead of
  `std::string(symname, symsize)`. libbacktrace's `symsize` is the size of the
  code symbol, not the length of its name, so the old code read past the name
  and could run off the end of the mapped string table, turning an expected
  `ICHECK` / `InternalError` into a SIGSEGV while its backtrace was being
  built. Upstream TVM's new FFI (apache/tvm #17920, now apache/tvm-ffi) no
  longer uses `symsize` here, but that code arrives only with the FFI migration
  and cannot be cherry-picked into the pinned commit.

## Fixes already included in the pinned TVM

The pinned commit `6dcf3f0d3169e693b3f5f0525556a40d23eaf703` already treats
explicit unit-step slices (`a:b:1`) as ordinary `BufferRegion` slices. This
supports dynamic stops without entering the constant-lane `Ramp` path. The old
`tvm_slice_step_fix.patch` is therefore removed: its equivalent edit has different
comments and cannot pass either the forward or reverse patch check on this pin.
Do not reapply it or weaken the generic patch script's conflict checks.

This Python slicing fix is independent of the Apache TVM `96b8257` interval
relaxation backport, included earlier at `88f6cfaf4`. The latter fixes recursive
arithmetic-analysis expansion in C++; it is not a replacement slice patch.

## Regenerating a patch after a submodule bump

    cd 3rdparty/tvm
    # apply the intended edit to the source file, then:
    git diff -- src/runtime/logging.cc > ../patches/tvm_backtrace_symbol_name_length.patch

When a fix becomes part of the pinned commit, remove its obsolete patch and
update the build-entry regression instead of retaining a redundant textual edit.
