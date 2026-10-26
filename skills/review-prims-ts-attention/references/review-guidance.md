# Reviewing FlashInfer Prims-TS attention

Apply this guidance to changes in `flashinfer/attention/prims_ts`, its preparation/JIT bindings, and corresponding tests. It distills PerkzZheng's reviews of context, decode, MLA, block-sparse, Sage, and Q-token sparse attention. Use the current PR's scope and base-to-head diff; historical requests do not establish today's supported shapes or APIs. [Review evidence](review-evidence.md) records sources, exceptions and coverage.

The central question is: **does this change extend a reusable attention generator with a clear contract, minimal machinery and sufficient evidence?** A passing local experiment alone does not establish that the proposed code or tests belong in the PR.

## Separate validation work from tests shipped in the PR

- Do not add local-validation-only tests to the PR, including functionality probes written to check an implementation during development. Keep temporary probes, tuning scripts, standalone benchmark helpers, exhaustive matrices and debug instrumentation in a local workspace or the agreed validation repository.
- Apply the same scope rule to documents: exclude experiment diaries, local paths, benchmark histories and review reports from the source PR. Retain concise documentation only for behavior, requirements or contracts changed by the implementation; an extensive campaign does not require an extensive kernel README. When requested, reduce that README to supported combinations and target model recipes, keeping detailed helper/layout explanations beside the code.
- Retain compact accuracy tests and important public-contract/regression tests. For each addition, identify the observable failure it detects and the gap in existing coverage. Prefer an added case in the existing context, decode, MLA, sparse, or shared-feature suite over another test module.
- Existing public execution coverage is generally more valuable than a new test of flags, defaults, private fields, exact resource counts, selected tiles/splits, internal call ordering, source strings, or a mocked path that bypasses the behavior under review. Do not pin a mutable policy choice as if it were an API guarantee.
- A real GPU test can still be unnecessary: repeating the same feature across a large parameter product, building a generic fixture for a few cases, or asserting wrapper internals does not justify its maintenance cost. Keep the independent numerical reference and representative boundaries that actually matter.
- Do not interpret test reduction as permission to remove meaningful coverage. Variable Q/KV lengths, tails, masking, changed-input graph replay, and a distinct affected kernel family may require a small maintained case. A targeted sanitizer test can also be justified when numerical output alone cannot reveal an invalid read.
- When a test's purpose is questioned, identify the distinct execution branch before deciding to remove it. Random inputs may exercise only a held softmax anchor; late keys that raise the maximum can be needed to test re-anchoring. Explain that purpose in the test name and merge related layout cases. Conversely, avoid repeating full type/range and validation-mode matrices for each new tuning scalar when existing coverage already owns those contracts.
- Keep trace examples and JSON changes to the smallest set required by the agreed public interface. Avoid a new template, fixture matrix, or test utility package solely to support the review campaign.

Apply this as a scope decision before writing test infrastructure, not as cleanup after adding hundreds of lines. The historical validation repository is `PerkzZheng/flashinfer-prims-ts-validation`; confirm the intended comparator and workflow for the current task rather than hard-coding that repository into unrelated reviews.

## Audit names across kernel families

Read the analogous context, FMHA decode and MLA paths before adding a name. Make a short mapping of the changed concepts and their peer names when the change crosses families.

| Concept | Review rule |
| --- | --- |
| Q/KV tile extent | Prefer established `tile_size_q` / `tile_size_kv` terminology. If another field measures grouped rows or an allocation extent, explain how it differs rather than introducing a synonym. |
| Precision | Distinguish raw Q/K/V storage dtypes from the QK/PV MMA compute dtypes. Reuse the existing QK/PV or BMM1/BMM2 convention consistently; document any mapping. |
| General capabilities | Use capability names such as `sparse_mla`, attention sinks or PDL, rather than a customer/model name or a particular metadata producer. |
| Variant helpers | Use a recognizable, consistent feature prefix, such as `sage_`, where imported helpers could be confused with generic helpers. Do not mix competing prefixes for the same concept. |
| Boolean settings | Use booleans for boolean semantics and the peer `use`/`has`/`enable` convention. Separate independent facts instead of hiding one inside an unrelated profile predicate. |
| Dimensions and offsets | State whether a value counts bytes, elements, words, tokens, heads, stages, instances or physical pages. Name semantic and storage extents separately. |
| Helper behavior | Distinguish a range check that rejects invalid values from clamping, normalization or preparation. Name the operation the implementation actually performs; do not infer changed semantics from a reviewer's tentative wording. |

Avoid vague abbreviations such as `mu`, `VC`, `direct`, `uniform`, or `coalesced` without defining the mathematical or memory-layout meaning. Rename new ambiguity at its source and update callers and documentation together; avoid unrelated repository-wide renaming.

### Avoid leading `_` where possible

Default to descriptive names without leading underscores for helpers, types, fields and modules. PerkzZheng finds `_` unclear; marking something as an implementation detail alone is not a sufficient reason to add the prefix.

- Prefer `planning.py`, `policy.py` and `sage_load_scale` over `_planning.py`, `_policy.py` and `_sage_load_scale`. Use clear names, feature prefixes and package organization to communicate purpose and ownership.
- Flag unnecessary leading underscores in code being added or modified. Apply the same naming choice to equivalent helpers across context, decode, MLA and other attention variants.
- Keep an underscore-prefixed spelling only for a concrete requirement, such as a Python protocol, an external interface, or compatibility with an existing supported name. Explain the requirement rather than treating private visibility as the default justification.
- Check callers, imports and exports when renaming. Removing a prefix does not automatically promote a helper to a supported public API; keep intentional exports explicit.
- Keep the cleanup within the reviewed scope and avoid redundant aliases introduced only for cosmetic renaming.
- Ordinary `snake_case` word separators remain appropriate for established Python names such as `tile_size_q` and `tile_size_kv`; this preference concerns unclear leading/private prefixes.

## Use a cohesive tuning configuration

When adding related fine-tuning options, prefer a small immutable typed configuration, such as the proposed `MlaDecodeTuning`, over repeatedly extending wrapper, one-shot and planning signatures with scalar keywords. Reuse an existing appropriate config and add fields for real supported behavior; do not introduce a generic tuning framework or speculative options.

Document each field's units, bounds and override semantics. For example, an unspecified softmax threshold can retain an entry point's default, while explicit `0.0` requests exact rescaling; those values are not interchangeable if a sparse schedule already defers correction. Keep intentional dense/sparse default differences explicit and resolve them consistently through callers and compilation.

The tuning object groups justified compile-time choices. It does not make batch size, sequence length, per-request tensors or other changing inputs static by default. Its fields still need the JIT-cost review below; immutability is not evidence that every value deserves a separate kernel.

## Keep specialized preparation and helpers cohesive

Keep variant-specific preparation, scale/index layouts, validation and helper math together under an appropriate subdirectory, for example `flashinfer/attention/prims_ts/sage/` or a justified `vc_attention/` package. A possible division is:

```text
prims_ts/<variant>/
    __init__.py
    config.py      # recipe and run-input contract
    layout.py      # prepared representation
    prepare.py     # preparation implementation
    index.py       # addressing shared by consumers
```

These are responsibilities, not a required file checklist. Create only the boundaries the feature needs. A package is not mandatory for a small cohesive helper set: one feature-named module with section dividers can expose the structure more clearly than several tiny files. Honor an explicit consolidation request and update imports/callers together. Keep task/resource ownership in its kernel family. The Sage package also demonstrates shared mode names and INT8-score helpers.

Keep reusable attention computation in the existing core kernels. A general QK-16/PV-8 capability should extend their precision/configuration model rather than add model-specific branches throughout the task graph. Share preparation and layouts between context and decode when their contracts agree; each kernel can retain its own resources, scheduling and softmax integration.

Decide which preparation belongs to the framework. Avoid silently adding input quantization, scale composition, repacking, or metadata updates to every attention call when the producer can prepare the consumed representation directly. If FlashInfer provides that preparation as part of the variant, isolate it in the feature package and make its cost and lifetime explicit.

Package organization does not justify duplicating MLA frontends, load/MMA tasks, reducers, CLC implementations or experimental-package proxies. Extend the existing family when it already owns that computation. Use a separate package for a coherent feature/preparation boundary, not another copy of the core kernel.

## Reuse mechanisms and explain necessary special cases

- Reuse existing TMA descriptors, inputs, SMEM/TMEM resources, tasks, reducers and predicates where semantics and lifetimes match. Trace each new parameter, helper, buffer and launch to an actual consumer.
- Prefer an existing CuTe/NVVM primitive over added inline PTX. When a primitive cannot express the operation or supported compiler versions need a fallback, explain why and keep the implementation in the appropriate helper boundary.
- Preserve meaningful K/V or stage distinctions. Sharing code does not require conflating different dtypes, live ranges, descriptor ownership or barriers.
- Derive generic loops and tail handling from supported geometry. Do not spell out one new head dimension or customer shape throughout the implementation when a shared loop can express it.
- Use small functions for repeated conditions or coherent transformations. Remove unused alternatives and wrappers that only forward calls; avoid a second configuration framework or custom debug TaskManager.

### Document every helper

Every helper function in the reviewed code must have a purpose docstring or an adjacent function comment. This includes existing helpers within the review scope, newly added or modified helpers, nested functions, resource helper methods, and CuTe DSL/JIT/device helpers across the affected kernel families. Enumerate the actual helper definitions when checking coverage; checking only files named `helpers` misses functions elsewhere.

- Explain the operation and result in concrete terms. State relevant input/output units, shapes, layouts, dtypes, side effects and caller preconditions where they are needed to use or understand the helper. Distinguish conversion, view/address calculation, validation and synchronization.
- Explain non-obvious implementation steps beside the code: formulas and constants, index/scale mapping, rounding or saturation, masking and tails, and the reason for a performance-sensitive instruction or branch. For synchronization helpers, identify producer/consumer ownership, the data being published or acquired, and the buffer lifetime protected by the operation.
- Explain parameters and configuration flags where they are defined: what the enabled/disabled states mean, their units and any caller promise. Expand ambiguous abbreviations such as a warp-specialized MMA flag or a retained-payload flag. Explain the difference between similarly named copies, staging paths and reductions, including why distinct paths remain necessary.
- Derive addressing constants from tile dimensions, element widths and hardware column/row units. For named PTX layouts such as Layout E/F, link the authoritative section and describe the relevant thread/row/column mapping. State which partials a reduction combines and why; a layout name alone is insufficient.
- Document each source pool's payload, scale placement, physical stride and index units near descriptor construction, using a small byte/axis diagram when useful. Validation adapters and numerical references must use each pool's actual record stride rather than a logical feature width shared by other formats.
- Keep simple helpers concise; one accurate sentence can be sufficient. Comments must add information beyond repeating the function name or narrating obvious statements. An explanation in a PR description or distant README does not replace documentation at the helper definition.
- Check comments against the implementation and callers, including the applicable 1CTA/2CTA or Keeps/Swaps variants. Update stale explanations when behavior changes. Do not claim a numerical, ordering or performance guarantee that the code does not establish.
- Report missing, misleading or insufficient helper documentation with source locations during review. During authorized cleanup, fix it before handoff. Keep the audit within the agreed review scope rather than expanding it into unrelated repository-wide edits.

### Explain prefetch and buffering precisely

For a prefetch change, identify the payload (for example, page IDs rather than K/V values), storage location, producer warp, current/next buffer lifetime, and lead time before first use. Distinguish reusing a register-held window from fetching its successor early. Existing loader warps prefetching into registers and a dedicated producer staging into SMEM have different resource and synchronization costs; a shared label such as “prefetch” does not make them the same design.

Check the actual overlap with slot waits, payload loads and computation. A benefit on FP8 does not establish a BF16 or sparse-path benefit: their payload latency, producer roles and bottlenecks may differ. Compare the relevant controls and, where a timing claim depends on it, confirm the emitted load schedule. Look for reusable mechanisms, but respect an explicitly accepted small optimization whose broader producer/SMEM redesign is deferred.

## Separate policy facts, runtime state and compile identity

Put register/stage policy in the owning configuration and have task factories consume it. Reuse an existing setting when its semantics match; do not force loader and MMA roles to share a budget if their live state differs. A derived configuration property can remove a task-side literal without adding another static tuning field. Explain budget arithmetic and validate the resulting resource allocation.

Write down the agreed primary/extra format pairs before simplifying source selection. Predicate names should describe the actual capability and source roles. Remove unreachable reverse-pool or extra-dtype branches when they are outside the requested contract; do not silently broaden the matrix for symmetry. Keep support validation consistent for public planning and independently callable kernel construction without duplicating validation machinery.

Review tile/split decisions using the actual work unit, including `num_instances_kv * tile_size_kv`, grouped Q/head rows, SM occupancy and complete resource footprint. Explain measured crossover choices and correctness limits separately. A specific batch-size exception or unexplained threshold is not a substitute for a policy model.

### Distinguish implementation limits from missing validation

For every unsupported kernel combination or admission guard, identify the actual reason. Do not equate “not validated yet” with “not supported.”

| Reason | Review action |
| --- | --- |
| Hardware or ABI constraint | Identify the instruction, representation or addressing requirement. Retain and document a guard that enforces it. |
| Implementation or resource limitation | Trace the missing path, layout restriction, synchronization requirement or resource bound. Explain what must change to support the combination. |
| Validation has not been performed | Treat this as work to complete, not a permanent support restriction. Validate the structurally legal combinations and enable them when the evidence supports it. |

A page-size restriction such as `page_size == 32` is not justified merely because only page size 32 was tested. Check the actual page/fragment divisibility, addressing, masking, resource and scheduling requirements. If other sizes satisfy them, run representative qualification across the affected paths and replace the arbitrary whitelist with the genuine structural conditions.

When a restriction comes from shared window or staging geometry, audit every consumer, including BF16 and FP8, rather than guarding only the newly optimized dtype. Consider whether padding or grouping several tiles could relax it. If the reviewer explicitly accepts the current recipe's narrower geometry for a small performance change, document that implementation assumption and follow-up separately from an architectural limit; do not force the larger redesign into the accepted scope.

Avoid carrying a historical validation matrix into runtime support policy without investigating it. Run broader qualification locally; commit only the minimal durable accuracy or regression coverage justified by the test-scope rules above. Document implementation constraints and completed validation coverage separately.

Do not remove a real correctness guard or claim untested combinations have passed. If validation cannot be completed, report the exact coverage gap and required follow-up rather than presenting it as a fundamental kernel limitation.

### Plan and compile identity

Distinguish static model/device facts, generated-kernel identity, workspace capacity, and live request values. Avoid recompilation or policy-key proliferation when different runtime sizes generate the same kernel; preserve fields that genuinely change layouts, baked validation bounds, resource topology, compiler options or hardware behavior. Explain cache hashing costs and `compare=False` / `hash=False` exclusions. Do not let a global policy cache retain per-layer tensors or mutable scratch.

Check that `plan()` / `run()` ownership and workspace sizing remain coherent across context, decode and MLA. Prefer reuse of caller-owned buffers and original cache strides over repeated allocation, fill/copy, transposition or host readback. Exact static contracts and runtime bounds must still follow the current API; do not mechanically impose one historical PR's plan signature on every path.

### Audit new static parameters for real-inference JIT cost

Treat each new compile-time parameter as a review decision. A static Python argument, `Constexpr`, captured plan field, concrete tensor shape/stride, generated-code option or compile-cache-key component can create another specialization. Trace where it becomes static rather than relying on its name. Pay particular attention to batch size, total query tokens, per-request query/KV lengths, maximum-length bounds and derived shape tuples, which can vary frequently during real inference.

Prefer runtime arguments or symbolic extents when the generated kernel can handle those values without recompilation. Being known at `plan()` time, convenient for the wrapper, or fixed in a benchmark is not sufficient justification for specializing it. A shape-dependent CPU policy lookup is not automatically a distinct compiled kernel; inspect actual generated signatures/artifacts and reuse across equivalent configurations.

For each added or newly static parameter, record:

- **Necessity:** the layout, resource topology, instruction/code-generation constraint or measured optimization that requires specialization; explain why a runtime value cannot meet the same need.
- **Inference variability:** the expected values and how often they change across requests, decode steps, batching policies, layers and graph buckets. Check whether the new parameter multiplies variants already created by dtypes, head geometry and other static choices.
- **Compilation cost:** the expected unique kernel count, cold JIT time, startup/warmup or first-use latency, and whether realistic workload changes trigger more compilation after warmup. Report estimates as estimates when measurements are unavailable; a warm disk cache alone does not establish acceptable cold-start behavior.
- **Benefit and alternative:** compare any warmed execution gain against compilation and cache growth. Prefer runtime handling or a small justified set of reusable profiles/buckets over an exact-size specialization for every observed batch or sequence length. Preserve necessary capacity checks and every cache field that actually changes generated behavior.

Present this specialization inventory to the user and explicitly ask them to check whether the newly added compilation parameters, especially batch size and sequence length, are reasonable for their deployment. First use workload/configuration evidence already available. If ranges or shape churn are unknown, ask a focused question such as: “What batch-size and Q/KV-length ranges change during serving, and can these proposed static parameters reuse a bounded set of kernels?” Continue independent review work while that information is pending; do not invent a narrow workload or require repetitive approval for facts already established.

Validate compile reuse over representative changing shapes locally. Keep qualification sweeps and compilation probes out of the PR unless a small durable regression test is justified by a real coverage gap. Do not remove a necessary specialization or omit a genuine compile-key dependency merely to reduce variant counts; the goal is justified, bounded compilation cost with correct runtime behavior.

## Check the entire dependency and numerical path

Trace producer → preparation → attention → reduction → output for affected families, including the corresponding Keeps/Swaps and 1CTA/2CTA paths when relevant.

- Put a PDL acquire before the first dependent read, under the PDL capability condition; do not disable PDL or tie it only to split-KV as a workaround. Preserve independent prologue overlap where the dependency permits it.
- Express ownership, publication and reuse through the existing TS resources/pipelines. Review extra GMEM→SMEM→register staging, vectorization, transaction counts and all consumers before introducing manual synchronization.
- Check dense versus boundary tiles, partial dimensions, inactive/empty requests, invalid indices and all-masked rows. Preserve row maxima/sums, FP8 probability scaling, sink mass and reduction/final-output conventions.
- Explain new constants, formulas and performance mechanisms beside the code. Comments should make tile units, ownership and head/mainloop/tail progression understandable without the review conversation.
- Check composition with existing output quantization and other requested epilogues. Removing an admission guard is insufficient: account for extra scratch/registers, stage lifetimes and final versus split-reduction stores, then validate output values, scales and graph replay through the public API. Keep a real resource constraint documented while fixing the missing combination.
- Review changes shared by sparse and dense paths against both. Device admission, instruction support and completed accuracy/performance qualification are separate claims.

## Validate once at the owning boundary

Remove duplicated checks and avoid host synchronization for properties already validated at an authoritative boundary. Preserve checks that protect earlier work, ABI/alignment/capacity requirements or genuinely different contracts. Name a validation helper as validation when it does not prepare or allocate data.

Explicit caller preconditions remain obligations: removing overlap checks does not enable in-place execution, and masking scores does not necessarily neutralize NaNs in padded V. Follow the final reviewed contract; PerkzZheng accepted optional V-tail handling after discussion and broader performance evidence. Do not turn an early review question into a blanket ban on a necessary check or fallback.

## Require useful evidence and a concise handoff

When timing process-local kernel patches, ensure compilation/cache identity includes the patched implementation or force fresh compilation. Record patch/source identity; changing an in-memory function does not necessarily invalidate a persistent kernel cache.

Measure equivalent layouts, precision, inputs, metadata, graph/cache conditions and hardware. Include preparation and reduction in complete-path claims, and identify component-only results separately. Explain the source of a speedup and show relevant unchanged paths for regressions. Compare the new supported configuration against the previous implementation where that is the claimed benefit.

Check ordinary and localized placement separately when tuning resource profiles. A gain under one placement does not justify a regressing automatic default elsewhere. When the requested scope retains such tuning, keep it as an explicit opt-in, document its resource/toolchain requirements and report the tradeoff. Do not dispatch on hidden benchmark-only placement information.

Keep the PR description focused on final behavior, supported contracts/limits, the relevant baseline and latest absolute performance numbers with units. Keep campaign history and local artifacts out. Do not promote one review's tolerance, shape cohort or instruction preference into an unconditional rule.

Before handoff, verify that the final diff contains the needed core changes, a coherent preparation boundary, consistent names/visibility, accurate documentation for every in-scope helper, minimal maintained tests, current API documentation, and validation evidence for the actual reviewed head. Fix the common cause across affected kernel families when appropriate; do not broaden the PR into a full kernel unification merely because a review discussed it as future work.

## Prepare an authorized PR

For optimization consolidation, inventory the earlier promising kernel profiles and ideas before reducing the diff. Mark each as integrated, already represented upstream, rejected by current-base evidence, or still unqualified. A default-dispatch matrix does not requalify all stage-count, tile, resource-layout and prefetch candidates. Recheck the promising omitted candidates and retain useful placement-dependent choices as explicit profiles when that is the requested scope; exclude local artifacts without discarding justified kernel tuning.

When the user asks for a rebase or publication, preserve the previous head before rewriting history and fetch the public repository's requested base. Resolve conflicts against the current implementation, then review the complete base-to-head diff; a clean rebase alone does not validate behavior after upstream changes. Re-run affected correctness and performance checks when the rebase or cleanup changes generated code, dispatch or the comparator. Do not present earlier-base measurements as results against the new main.

Keep local qualification outside the PR. Prefer adding a numerical case to an existing suite over shipping a dedicated architecture-probe module, a planner spy, or exact heuristic snapshots. Retain the independent reference and the distinct execution branch the case protects; removing debugging machinery must not remove meaningful coverage.

Use the destination repository's PR template and keep the authored content short: the concrete problem, final behavior, and relevant current validation. Performance claims need the GPU, shape, precision and absolute before/after values with units. Distinguish memory placement and compute partitions when they change the comparison. Leave tuning history, raw logs and local artifact paths outside the description.

For commits authorized on PerkzZheng's behalf, use his established Git identity for both author and committer, and add a matching `Signed-off-by: Name <email>` trailer to each task-owned commit. His configured and historical name is `Perkz Zheng`; use the established public address `67892460+PerkzZheng@users.noreply.github.com` for public GitHub pushes, since the configured private email is blocked by GitHub's GH007 protection. Preserve account privacy settings. Do not override his identity with Codex or another agent identity. Recover missing identity details from repository configuration or prior commits; ask only if they remain unclear. This is a DCO sign-off, not a cryptographic signature. Respect later explicit identity instructions and preserve unrelated contributors' commits. When correcting published metadata within authorized scope, back up the old head, verify every rewritten commit keeps its original tree, and push with an explicit lease; source-identical metadata changes do not require repeating GPU benchmarks.

Push only within the user's authorized scope. For a fork PR, verify the head owner/branch and public base, then check that the remote and PR heads match the validated local head. Reuse an existing PR for that branch when appropriate. Skill updates, CI-trigger comments and other external changes require their own authorization; a review request alone does not authorize them.
