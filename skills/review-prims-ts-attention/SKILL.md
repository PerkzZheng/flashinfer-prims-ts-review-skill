---
name: review-prims-ts-attention
description: Review or clean up FlashInfer Prims-TS attention PRs and branches using PerkzZheng's guidance on test scope, consistent naming, avoiding leading underscores, helper documentation, preparation packages, support constraints, JIT specialization cost, and kernel correctness and performance. Use for task-scheduled context, decode, MLA, sparse and Sage attention changes; not as a general policy for unrelated CUDA projects.
---

# Review Prims-TS Attention

Review changes as extensions to a reusable attention generator, with clear contracts, minimal machinery and evidence for the affected paths. This skill distills PerkzZheng's public FlashInfer reviews and explicit preferences. Current task instructions and the actual implementation take precedence over historical examples.

Read [the review guidance](references/review-guidance.md) before reviewing. Consult [the source reviews](references/review-evidence.md) when a rule's rationale, exceptions or historical context matters. This package is self-contained; it does not require another review skill.

## Establish scope and recover the change

- Determine whether the task is review-only or authorizes edits, tests, replies, pushes or thread resolution. A request to review does not itself authorize external changes.
- Read repository instructions and identify the base, current head, merge base and complete diff. Preserve unrelated work and keep local validation artifacts outside the PR.
- Read the PR description, review summaries, inline threads and follow-ups. Check each claim against the current head; distinguish valid, stale, already-addressed and unreachable findings.
- Record the intended public behavior, changed kernel families, layouts, dtypes, shapes, hardware, runtime inputs, static configuration and validation claims.
- Compare analogous context, FMHA decode, MLA, sparse and Sage paths before introducing a new convention or changing shared code.

## Apply the core preferences

1. **Separate local validation from maintained tests.** Do not ship exploratory functionality probes, exhaustive qualification matrices, tuning scripts or debug scaffolding merely because they helped development. Retain minimal accuracy, important public-contract and reachable-regression coverage, preferably in existing suites. A numerical test can still be redundant.
2. **Keep names consistent across kernels.** Use established Q/KV, tile, scale, layout and resource terminology. Distinguish raw storage precision from QK/PV MMA precision and state units explicitly. Prefer capability names over customer/model names.
3. **Avoid leading `_` where possible.** Prefer descriptive names, feature prefixes and coherent modules. Being an implementation detail alone is not a sufficient reason for the prefix. Keep an underscore-prefixed name only for a concrete requirement or compatibility; inspect callers and exports when renaming. Ordinary Python `snake_case` remains appropriate.
4. **Isolate specialized preparation.** Group variant preparation, layouts and helpers in a feature package such as `prims_ts/sage/`. Share compatible components across kernels while retaining native task/resource ownership. Extend existing core mechanisms instead of copying an entire kernel family.
5. **Explain unsupported combinations.** Distinguish hardware/ABI constraints, real implementation/resource limitations and missing validation. A page-size-32-only test history does not justify a page-size-32-only runtime guard. Validate structurally legal combinations and broaden admission when supported by evidence; report unresolved coverage gaps without calling them fundamental limitations.
6. **Avoid unnecessary static compilation parameters.** Keep changing inference dimensions, especially batch size and Q/KV sequence lengths, runtime unless specialization has a concrete implementation need or measured benefit that justifies its JIT cost. Inventory newly added compile-time parameters and ask the user to check their necessity and expected value ranges against real inference traffic. Use known workload information first; ask for missing shape/churn assumptions rather than silently accepting exact-size specializations. Check cold compilation, first-use latency and the number of compiled variants, not only warmed kernel speed.
7. **Group related tuning controls.** Prefer a small immutable typed tuning configuration over a growing list of scalar tuning keywords. State units, bounds and the distinction between an unspecified value and an explicit override such as zero. Keep runtime request metadata separate, and apply the JIT-specialization audit to every compile-time field.

8. **Document every helper.** Require a concise docstring or adjacent function comment for every helper in the reviewed code, including shared helpers, nested helpers, resource methods and JIT/device helpers. Explain what it does and the contract callers rely on. Add inline comments for non-obvious implementation choices, layout/precision rules and synchronization. Audit existing helpers within the review scope as well as new ones; see [helper documentation](references/review-guidance.md#document-every-helper).

## Trace correctness, reuse and cost

Follow changed data through producer/preparation, public plan/run bindings, policy selection, compilation, attention, reduction and output. Inspect affected dense and sparse paths, Keeps/Swaps layouts and 1CTA/2CTA families as relevant.

Check masking, tails, empty or inactive requests, stride/alignment, cache coordinates, workspace capacity, resource lifetime, synchronization, scales and numerical conventions. Put dependent-launch waits before the first dependent read under the actual capability condition. Prefer existing TS resources, tasks and supported DSL primitives; justify new buffers, descriptors, launches or inline PTX.

For prefetch or caching changes, identify what moves, where it is stored, which warp produces it and when it becomes usable. Distinguish register-window reuse from lookahead prefetch, and existing-loader register buffering from dedicated-producer SMEM staging. Check shared geometry assumptions across dtypes and compare performance before generalizing the optimization to other paths.

Review policy with actual work geometry, occupancy and resource limits rather than unexplained shape exceptions. Keep runtime values out of compile/cache identity unless they genuinely change generated behavior or baked constraints. Make caller-owned storage, plan lifetime, graph capture and concurrent workspace ownership explicit. Remove duplicated checks at redundant boundaries while preserving checks required for correctness.

For performance claims, compare equivalent layouts, inputs, precision, graph/cache conditions and hardware. Include preparation and reduction in complete-path measurements and label component-only results separately. Report absolute values and units, relevant baselines and regressions. Do not reuse a historical tolerance or shape cohort as a universal gate.

## Validate and report

Match validation to the actual change. Use existing focused correctness tests, then affected suites and representative hardware checks where warranted. Broader local campaigns need not be committed. Do not weaken numerical references or tolerances to obtain a pass, and do not claim unavailable hardware or unexecuted CI as validated.

For review-only work, lead with actionable findings, their source locations, observable consequences and the smallest coherent remedy. Explain the reason for a support restriction rather than only labeling it unsupported. When compilation parameters change, include their specialization rationale, expected variant growth and JIT/inference impact, and request the user's workload check. State any evidence gaps explicitly.

For authorized cleanup, keep the final diff limited to the agreed behavior, coherent structure, necessary tests and current public documentation. Verify the final head and report what changed, what ran and what remains unresolved. Keep PR descriptions focused on final behavior and current evidence, not the development campaign. Follow the user's authorization for publication and thread resolution.

For authorized commits on PerkzZheng's behalf, use his established Git identity for author, committer and `Signed-off-by`, rather than an agent identity. Use his public GitHub noreply address for public GitHub pushes; follow the identity and sign-off details in the PR handoff guidance below.

When preparing an authorized rebase or PR, use [the PR handoff guidance](references/review-guidance.md#prepare-an-authorized-pr): preserve the previous head, compare with the fetched public base, exclude unnecessary documents and local debug tests, and validate the final diff before publication. Keep the authored title and description short while following the repository's PR template.
