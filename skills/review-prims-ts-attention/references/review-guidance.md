# Reviewing FlashInfer Prims-TS attention

Apply this guidance to changes in `flashinfer/attention/prims_ts`, its preparation/JIT bindings, and corresponding tests. It distills PerkzZheng's reviews of context, decode, MLA, block-sparse, Sage, and Q-token sparse attention. Use the current PR's scope and base-to-head diff; historical requests do not establish today's supported shapes or APIs. [Review evidence](review-evidence.md) records sources, exceptions and coverage.

The central question is: **does this change extend a reusable attention generator with a clear contract, minimal machinery and sufficient evidence?** A passing local experiment alone does not establish that the proposed code or tests belong in the PR.

## Separate validation work from tests shipped in the PR

- Do not add local-validation-only tests to the PR, including functionality probes written to check an implementation during development. Keep temporary probes, tuning scripts, standalone benchmark helpers, exhaustive matrices and debug instrumentation in a local workspace or the agreed validation repository.
- Retain compact accuracy tests and important public-contract/regression tests. For each addition, identify the observable failure it detects and the gap in existing coverage. Prefer an added case in the existing context, decode, MLA, sparse, or shared-feature suite over another test module.
- Existing public execution coverage is generally more valuable than a new test of flags, defaults, private fields, exact resource counts, selected tiles/splits, internal call ordering, source strings, or a mocked path that bypasses the behavior under review. Do not pin a mutable policy choice as if it were an API guarantee.
- A real GPU test can still be unnecessary: repeating the same feature across a large parameter product, building a generic fixture for a few cases, or asserting wrapper internals does not justify its maintenance cost. Keep the independent numerical reference and representative boundaries that actually matter.
- Do not interpret test reduction as permission to remove meaningful coverage. Variable Q/KV lengths, tails, masking, changed-input graph replay, and a distinct affected kernel family may require a small maintained case. A targeted sanitizer test can also be justified when numerical output alone cannot reveal an invalid read.
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

Avoid vague abbreviations such as `mu`, `VC`, `direct`, `uniform`, or `coalesced` without defining the mathematical or memory-layout meaning. Rename new ambiguity at its source and update callers and documentation together; avoid unrelated repository-wide renaming.

### Avoid leading `_` where possible

Default to descriptive names without leading underscores for helpers, types, fields and modules. PerkzZheng finds `_` unclear; marking something as an implementation detail alone is not a sufficient reason to add the prefix.

- Prefer `planning.py`, `policy.py` and `sage_load_scale` over `_planning.py`, `_policy.py` and `_sage_load_scale`. Use clear names, feature prefixes and package organization to communicate purpose and ownership.
- Flag unnecessary leading underscores in code being added or modified. Apply the same naming choice to equivalent helpers across context, decode, MLA and other attention variants.
- Keep an underscore-prefixed spelling only for a concrete requirement, such as a Python protocol, an external interface, or compatibility with an existing supported name. Explain the requirement rather than treating private visibility as the default justification.
- Check callers, imports and exports when renaming. Removing a prefix does not automatically promote a helper to a supported public API; keep intentional exports explicit.
- Keep the cleanup within the reviewed scope and avoid redundant aliases introduced only for cosmetic renaming.
- Ordinary `snake_case` word separators remain appropriate for established Python names such as `tile_size_q` and `tile_size_kv`; this preference concerns unclear leading/private prefixes.

## Give specialized preparation code a feature package

Keep variant-specific preparation, scale/index layouts, validation and helper math together under an appropriate subdirectory, for example `flashinfer/attention/prims_ts/sage/` or a justified `vc_attention/` package. A possible division is:

```text
prims_ts/<variant>/
    __init__.py
    config.py      # recipe and run-input contract
    layout.py      # prepared representation
    prepare.py     # preparation implementation
    index.py       # addressing shared by consumers
```

These are responsibilities, not a required file checklist. Create only the boundaries the feature needs. The Sage package also demonstrates shared mode names and INT8-score helpers.

Keep reusable attention computation in the existing core kernels. A general QK-16/PV-8 capability should extend their precision/configuration model rather than add model-specific branches throughout the task graph. Share preparation and layouts between context and decode when their contracts agree; each kernel can retain its own resources, scheduling and softmax integration.

Decide which preparation belongs to the framework. Avoid silently adding input quantization, scale composition, repacking, or metadata updates to every attention call when the producer can prepare the consumed representation directly. If FlashInfer provides that preparation as part of the variant, isolate it in the feature package and make its cost and lifetime explicit.

Package organization does not justify duplicating MLA frontends, load/MMA tasks, reducers, CLC implementations or experimental-package proxies. Extend the existing family when it already owns that computation. Use a separate package for a coherent feature/preparation boundary, not another copy of the core kernel.

## Reuse mechanisms and explain necessary special cases

- Reuse existing TMA descriptors, inputs, SMEM/TMEM resources, tasks, reducers and predicates where semantics and lifetimes match. Trace each new parameter, helper, buffer and launch to an actual consumer.
- Prefer an existing CuTe/NVVM primitive over added inline PTX. When a primitive cannot express the operation or supported compiler versions need a fallback, explain why and keep the implementation in the appropriate helper boundary.
- Preserve meaningful K/V or stage distinctions. Sharing code does not require conflating different dtypes, live ranges, descriptor ownership or barriers.
- Derive generic loops and tail handling from supported geometry. Do not spell out one new head dimension or customer shape throughout the implementation when a shared loop can express it.
- Use small functions for repeated conditions or coherent transformations. Remove unused alternatives and wrappers that only forward calls; avoid a second configuration framework or custom debug TaskManager.

## Separate policy facts, runtime state and compile identity

Review tile/split decisions using the actual work unit, including `num_instances_kv * tile_size_kv`, grouped Q/head rows, SM occupancy and complete resource footprint. Explain measured crossover choices and correctness limits separately. A specific batch-size exception or unexplained threshold is not a substitute for a policy model.

### Distinguish implementation limits from missing validation

For every unsupported kernel combination or admission guard, identify the actual reason. Do not equate “not validated yet” with “not supported.”

| Reason | Review action |
| --- | --- |
| Hardware or ABI constraint | Identify the instruction, representation or addressing requirement. Retain and document a guard that enforces it. |
| Implementation or resource limitation | Trace the missing path, layout restriction, synchronization requirement or resource bound. Explain what must change to support the combination. |
| Validation has not been performed | Treat this as work to complete, not a permanent support restriction. Validate the structurally legal combinations and enable them when the evidence supports it. |

A page-size restriction such as `page_size == 32` is not justified merely because only page size 32 was tested. Check the actual page/fragment divisibility, addressing, masking, resource and scheduling requirements. If other sizes satisfy them, run representative qualification across the affected paths and replace the arbitrary whitelist with the genuine structural conditions.

Avoid carrying a historical validation matrix into runtime support policy without investigating it. Run broader qualification locally; commit only the minimal durable accuracy or regression coverage justified by the test-scope rules above. Document implementation constraints and completed validation coverage separately.

Do not remove a real correctness guard or claim untested combinations have passed. If validation cannot be completed, report the exact coverage gap and required follow-up rather than presenting it as a fundamental kernel limitation.

### Plan and compile identity

Distinguish static model/device facts, generated-kernel identity, workspace capacity, and live request values. Avoid recompilation or policy-key proliferation when different runtime sizes generate the same kernel; preserve fields that genuinely change layouts, baked validation bounds, resource topology, compiler options or hardware behavior. Explain cache hashing costs and `compare=False` / `hash=False` exclusions. Do not let a global policy cache retain per-layer tensors or mutable scratch.

Check that `plan()` / `run()` ownership and workspace sizing remain coherent across context, decode and MLA. Prefer reuse of caller-owned buffers and original cache strides over repeated allocation, fill/copy, transposition or host readback. Exact static contracts and runtime bounds must still follow the current API; do not mechanically impose one historical PR's plan signature on every path.

## Check the entire dependency and numerical path

Trace producer → preparation → attention → reduction → output for affected families, including the corresponding Keeps/Swaps and 1CTA/2CTA paths when relevant.

- Put a PDL acquire before the first dependent read, under the PDL capability condition; do not disable PDL or tie it only to split-KV as a workaround. Preserve independent prologue overlap where the dependency permits it.
- Express ownership, publication and reuse through the existing TS resources/pipelines. Review extra GMEM→SMEM→register staging, vectorization, transaction counts and all consumers before introducing manual synchronization.
- Check dense versus boundary tiles, partial dimensions, inactive/empty requests, invalid indices and all-masked rows. Preserve row maxima/sums, FP8 probability scaling, sink mass and reduction/final-output conventions.
- Explain new constants, formulas and performance mechanisms beside the code. Comments should make tile units, ownership and head/mainloop/tail progression understandable without the review conversation.
- Review changes shared by sparse and dense paths against both. Device admission, instruction support and completed accuracy/performance qualification are separate claims.

## Validate once at the owning boundary

Remove duplicated checks and avoid host synchronization for properties already validated at an authoritative boundary. Preserve checks that protect earlier work, ABI/alignment/capacity requirements or genuinely different contracts. Name a validation helper as validation when it does not prepare or allocate data.

Explicit caller preconditions remain obligations: removing overlap checks does not enable in-place execution, and masking scores does not necessarily neutralize NaNs in padded V. Follow the final reviewed contract; PerkzZheng accepted optional V-tail handling after discussion and broader performance evidence. Do not turn an early review question into a blanket ban on a necessary check or fallback.

## Require useful evidence and a concise handoff

Measure equivalent layouts, precision, inputs, metadata, graph/cache conditions and hardware. Include preparation and reduction in complete-path claims, and identify component-only results separately. Explain the source of a speedup and show relevant unchanged paths for regressions. Compare the new supported configuration against the previous implementation where that is the claimed benefit.

Keep the PR description focused on final behavior, supported contracts/limits, the relevant baseline and latest absolute performance numbers with units. Keep campaign history and local artifacts out. Do not promote one review's tolerance, shape cohort or instruction preference into an unconditional rule.

Before handoff, verify that the final diff contains the needed core changes, a coherent preparation boundary, consistent names/visibility, minimal maintained tests, current API documentation, and validation evidence for the actual reviewed head. Fix the common cause across affected kernel families when appropriate; do not broaden the PR into a full kernel unification merely because a review discussed it as future work.
