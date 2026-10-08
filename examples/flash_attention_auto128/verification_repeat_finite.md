# Repeated precision compatibility patch

`verification_repeat_finite.patch` adds two optional switches to the existing
designer verification script, not a new evaluation framework:

- `--check-finite`: reject NaN/infinity in candidate outputs on CPU.
- `--repeat-checks 20`: compare every launch against the original CPU FP64 and
  native references. Inputs are cloned for each launch; stop on first mismatch.

The CANN comparator, thresholds and reference implementation are unchanged.
Defaults remain a single launch without the extra finite gate. Combining these
options with `--compile-only` is rejected.

Deployment was applied to the existing, already-modified SDK file with its
other changes preserved. Snapshot before patch:
`b54c1a2bb2a0735513ea1750fa9bb3c88bbb6abfafea3c79455f586dc8e74bf5`.
Snapshot after patch:
`fe881e92db75fb3a8e5f13f6e47fe5124af72c269ad117a19ddbdd231858c1d2`.
Backup in Docker: `/tmp/fa128-verification-before-repeat-finite.py`.

The SDK Git root is `cannbot-skills-0817`, not its plugin subdirectory. Apply
from the root, with the plugin prefix, after checking the current file matches
the expected snapshot (otherwise rebase the patch; do not overwrite it):

```bash
git apply --check --directory=plugins-community/tilelang2ascendc-ops-generator /tmp/fa128-verification-repeat-finite.patch
git apply --directory=plugins-community/tilelang2ascendc-ops-generator /tmp/fa128-verification-repeat-finite.patch
```

Local regression: `python -m pytest -q
examples/flash_attention_auto128/test_verification_repeat_finite.py`:
10 control-flow checks passed. These use minimal test doubles and do not
constitute NPU precision evidence. Docker deployment passed Python syntax
compilation and the exact post-patch hash check. A separate CPU test in the
Docker Python environment extracted the deployed `_require_finite` function
and used real Torch: nested finite floating/integer/complex outputs passed;
NaN, positive infinity and negative infinity each raised `ValueError`.

The serial NPU twenty-launch validation completed with exit code0. All fourteen
fixed cases passed all twenty numerical comparisons with the finite gate
enabled. Each original JSON is archived under `results/` as
`fa128-prefetch-1aeedc56-caseN-stability20.json`. An independent read-back
checked every report's `ok`, `device`, `check_finite`, `repeat_checks` and
twenty-launch comparison message before performance measurement started.
This establishes stability for these fixed inputs across these 280 launches,
not all inputs or all future scheduling variants.

Hardware validation uses the original script with the original task directory:

```bash
verification_tilelang.py <flash_attention_task_dir> --case N --check-finite --repeat-checks 20 --json /tmp/fa128-prefetch-1aeedc56-caseN-stability20.json
```

Use the isolated compiler/build environment documented in
`results/a2-prefetch-first-hardware.md`, `FA_STEP=auto128`, physical card6, and
the fixed fourteen case IDs. Serial fail-fast execution, no simulator gate,
clock changes, device resets or other-process stops. A repeated precision pass
is not a performance pass and cannot establish the >=0.90 acceptance ratio.
