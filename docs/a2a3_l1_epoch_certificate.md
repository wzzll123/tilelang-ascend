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

`l1_prefetch.py` now discovers input-prefetch candidates from the actual task
graph. All accesses must belong to one owner, each native copy must have geometry
coverage, and GM sources must be readonly kernel inputs rather than generated
workspaces. It also runs physical materialization as a proof preflight to reject
escaped or malformed accesses. Production FA tests identify K/V while rejecting
Q (prefix access) and P (current-iteration Vector-produced workspace).

The two-stage partition places input writers in stage0 and consumers in stage1.
Joint feasibility validation keeps persistent allocations single-version and
counts every fixed allocation against A2/A3 capacities. Candidate slot counts
currently use the certified lifecycle span+1, rather than optimizing unconstrained
modulo start residues: a one-version timestamp solution can require a different
within-wave issue order from the lexical wave materializer. This constraint must
not be relaxed until the chosen executable order is modeled by the solver.

These discovery/solve functions are not yet called by the production driver.
Integration must replace original-source same-pipe FIFO constraints with those
of the reconstructed wave order, without dropping any true RAW/WAR/WAW edges;
otherwise moving V before the current P workspace receive conflicts with an
artificial old FIFO edge. Protocol minimization must be rechecked for that order.

`materialize_l1_prefetch` now applies the selected counts to all candidates,
refreshing the rebuilt owner after each rewrite. It then re-extracts physical
storage identities and supplies ring access footprints, complete body
dependencies and per-edge event slot counts. Both distance0 ready/release pairs
and carried reuse pairs use the selected count; setting only carried counts
would leave a one-slot same-iteration protocol and can deadlock stage0 against
stage1 in the same wave. Persistent storage keeps its original distance1 edges.

Tests exercise combined K/V rewrites for actual A2/A3 causal/noncausal FA input,
and a scoped native DMA fixture feeding the real mode2 event emitter and stage
wave reconstruction. `emit_three_phase_sync(prefetch=...)` now consumes that
exact materialization, verifies complete ring dependencies/slot counts, emits
boundary notifications and seed/drain outside the wave, and reconstructs whole
task/event bundles. Actual FA tests reach OptimizeForTarget and native AscendC
source generation on A2/A3. Omitting distance0 slot counts is a fail-closed
mutation test. The production entry still does not select this path, and no
multiversion hardware execution is claimed yet.

Integration exposed an overly conservative compiler event registry: it grouped
different local HardEvents by destination pipe alone. Official CANN 9.0 source
on NPU3 confirms independent `eventPool_ + EventToIndex(evt)` bitmaps:
`aarch64-linux/ascendc/include/basic_api/impl/kernel_tpipe_impl.h:450–487`.
`kernel_event.h:80–87,141–153,233–242` maps `MTE2_MTE1` and `M_MTE1` to different
AIC indices, with AIC/AIV handling for 2201/3510. Native codegen also emits
`HardEvent::<src>_<dst>`. The compiler registry now keys local pools by core and
both pipes; each pool still has IDs0–7, and no lifetime reuse was introduced.
Cross-core IDs remain shared across directions. Simulator semantics were not
changed as part of this compiler fix.
