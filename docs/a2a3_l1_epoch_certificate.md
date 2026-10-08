# L1 prefetch epoch coverage

The joint stage/version driver must prove that the first write of each K/V epoch
initializes every byte later consumed by L1→L0. `l1_epoch.py` provides this proof
from native copy geometry. It is not yet wired into production version selection.

For FP16, admit compact L1 rectangles with both axes divisible by 16. The native
template dimensions must match either the original 2D allocation or its actual
flattened allocation. `copy_gm_to_l1` must write the full row/column rectangle in
every active owner iteration. Its TIR ABI is source first; codegen reverses it for
the C++ helper. A literal zero dimension denotes the helper's full-dimension
default, not an empty copy. Neither pointer extent nor offset-zero clearing is
accepted as a full-write proof.

Readers must use matching L1→L0A/B native template geometry, correct source
pointer dtype/mode and zero logical tile offset. Transposed readers reverse the
row/column bounds. The conservative source footprint is the entire physical zN
tile, rather than a fictitious contiguous row-major subrectangle.

Native references: `src/tl_templates/ascend/common.h` (`copy_gm_to_l1`,
`copy_l1_to_l0a/b`) and `src/target/codegen_ascend.cc` (`need_clear` selection).
For partial tails, the initial slot at offset zero receives a clear, but the
current codegen can suppress that clear after rotation to a nonzero slot.
Consequently, tails cannot obtain this full-overwrite certificate yet. This is
not removal of FA tail support: the existing unrotated path remains unchanged.
Completing multiversion tail scheduling requires an explicit whole-slot clear
contract without breaking splice copies that must preserve previously written
subregions.

Regression coverage uses the production FA factory and real LowerAndLegalize /
workspace lowering / GEMM decomposition for causal and noncausal K/V copies.
Other tests reject partial/unbounded rows, splice offsets, reversed TIR operands,
mixed dtypes, mismatched allocations and escaping readers. These are compiler
tests, not simulator tests or claims of multiversion hardware validation.

Next integration steps:

1. Verify unique owner and all accesses, then attach overwrite/read footprints
   to task accesses; retain persistent online-softmax state at one version.
2. Select K/V versions jointly with stages and total L1 capacity.
3. Physical version materialization now consumes certified flattened L1 geometry
   and rechecks each native DMA endpoint before rewriting. Other Cube scopes
   continue to require aligned 2D allocation geometry. This step is complete;
   it does not independently choose versions or generate ring synchronization.
4. Reconstruct executable stage waves together with ready/release selectors,
   initialization and drain. Then run real A2 precision and performance tests.
