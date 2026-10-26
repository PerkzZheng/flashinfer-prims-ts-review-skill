# Prims-TS review evidence

This is the supporting evidence for [Prims-TS attention review](review-guidance.md). Quotes below are from `PerkzZheng`; the review rule alongside each quote is a synthesis, not an assertion that every historical implementation is still current. Current task instructions take precedence over these historical preferences.

## Coverage

Retrieved public FlashInfer PR evidence at `2026-10-22T05:48:36.451376+00:00` with the authenticated GitHub CLI. Six fully paginated searches produced 220 unique candidate PRs: `prims`, `attention-ts`, `task-scheduled`, `reviewed-by:PerkzZheng`, `commenter:PerkzZheng`, and `author:PerkzZheng`, all restricted to `repo:flashinfer-ai/flashinfer is:pr`.

Changed-file screening identified 68 Prims-TS attention PRs. Their complete paginated inline-comment, review and discussion lists contained 713 entries authored by PerkzZheng: 498 nonempty bodies and 215 empty review events. Written entries occur on 35 PRs; 39 have review activity when empty approval/comment events are counted. Counts include acknowledgements, bot-run commands and author follow-ups, not 713 distinct suggestions. Changed-file counts agree with GitHub PR metadata for all 68 selected PRs.

The scope includes the task-scheduled attention tree, its metadata/bindings, and attention-ts tests. Unrelated GEMM/MoE, TRT-LLM-only and SM120 non-TS work was screened out. Search discovers the available history at retrieval time; deleted comments, unpublished draft text and future feedback are not claimed. This package includes the public source links and selected excerpts that support the review guidance; raw authenticated API responses are not distributed.

### Follow-up requested on 2026-10-25

The full paginated inline-comment, review and discussion lists for PRs [#7019](https://github.com/flashinfer-ai/flashinfer/pull/7019) and [#7026](https://github.com/flashinfer-ai/flashinfer/pull/7026) were read separately from the initial collection. PerkzZheng authored 18 entries: 11 nonempty inline comments and seven empty review events. Author replies were read to resolve context; no new kernel measurements are claimed by this skill update.

Retrieved at `2026-10-26T01:58:17.286837+00:00`. Reviewed heads:

- [#7019: perf(prims-ts): prefetch page IDs for FP8 2CTA dense MLA K/V loaders](https://github.com/flashinfer-ai/flashinfer/pull/7019) — `688e04b2e6f59b7fbcc60eb251634477b51433b8`.
- [#7026: feat(prims-ts): opt-in softmax_rescale_threshold_log2 (skip correction) for dense and sparse MLA decode](https://github.com/flashinfer-ai/flashinfer/pull/7026) — `0df72d08f0a6349cc2f9824b0d10ae7fef9ff2e4`.

## Explicit preferences from PerkzZheng

- Do not add local-validation-only tests to PRs, especially functionality-related probes.
- Keep naming consistent across context, decode, MLA and the other kernels.
- Avoid leading `_` where possible; prefer clear descriptive names. PerkzZheng clarified: “we should avoid "_" if possible. it is not clear.”
- Put specialized attention variants with preparation code in a separate directory, following `prims_ts/sage/`.
- Distinguish real implementation limitations from combinations that have simply not been validated. Avoid support restrictions based only on missing validation, such as an arbitrary page-size limitation.
- Avoid unnecessary static compilation parameters, particularly batch size and sequence lengths, because they can cause excessive JIT compilation in real inference. Explicitly ask users to check newly added specialization parameters and their workload assumptions. This preference was clarified on 2026-10-23; it supplements the historical PR evidence rather than reporting a new benchmark result.

- Document every helper function with proper comments explaining what it does. PerkzZheng explicitly requested this local and remote skill update on 2026-10-26. This is a direct preference supplement, not a claim about a newly retrieved public PR review. The guidance applies it to all helpers within the agreed review scope, with concise purpose documentation and inline explanations for non-obvious behavior.

The underscore preference comes directly from PerkzZheng's clarification. Default to avoiding leading/private prefixes, with exceptions only for concrete naming requirements or compatibility. Ordinary snake_case word separators remain consistent with the requested cross-kernel Python naming style. Historical review comments are supporting context, not a claim that every underscore spelling is prohibited.

## Representative source comments

### Test and artifact scope

**Keep the balanced cost-model benchmark test local.** [PR #5235](https://github.com/flashinfer-ai/flashinfer/pull/5235#discussion_r4021931199)

> same, I think this should only be kept locally

**Consolidate a few useful cases into an existing attention suite instead of shipping a new validation matrix.** [PR #5229](https://github.com/flashinfer-ai/flashinfer/pull/5229#discussion_r4012319635)

> let us avoid conservative/over-enginnering tests for MR. remove it or only keep few in tests/attention/test_attention_ts_decode.py

**Retain accuracy and important validation coverage; this is not a ban on functional tests.** [PR #5434](https://github.com/flashinfer-ai/flashinfer/pull/5434#discussion_r4069370902)

> clean those tests. I want to only keep accuracy tests (or important validation tests) in attention/test_prims_ts_sparse_mla.py. we don't need experimental/prims_ts_sparse_mla anymore

**Reuse existing dtype coverage instead of adding a dedicated BF16 test file.** [PR #6529](https://github.com/flashinfer-ai/flashinfer/pull/6529#discussion_r4291457401)

> we don't need extra file for bf16 tests. the ones in test_attention_ts_decode.py is enough.

**Use the shared feature suite for maintained non-accuracy feature coverage.** [PR #6529](https://github.com/flashinfer-ai/flashinfer/pull/6529#discussion_r4291453999)

> can you merge this with [test_attention_ts_mask.py](https://github.com/PerkzZheng/flashinfer/blob/231dd82029b5fd81274f13ca65f428f270526697/tests/attention/test_attention_ts_mask.py), and rename it test_attention_ts_features.py so future feature tests can be added to it except for accuracy ones.

**Question internal resource-layout assertions added for an optimization.** [PR #6937](https://github.com/flashinfer-ai/flashinfer/pull/6937#discussion_r4344209788)

> this seems unnecessary.

**Even a GPU functional matrix needs a distinct maintained purpose; local confidence alone does not justify it.** [PR #6925](https://github.com/flashinfer-ai/flashinfer/pull/6925#discussion_r4333684543)

> unnecessary tests.

**Variable Q/KV lengths are meaningful correctness coverage.** [PR #4414](https://github.com/flashinfer-ai/flashinfer/pull/4414#discussion_r4032848161)

> the tests should cover more variable seqlens_q and seqlens_kv.

**A real masked-tile correctness distinction warrants corresponding tests.** [PR #4839](https://github.com/flashinfer-ai/flashinfer/pull/4839#discussion_r3931693251)

> this only works for dense tiles. for tiles that need masking, we should use normal LDTM + Masking + MAX reduction.  Please add corresponding tests.

**Do not add a local benchmark campaign to the production PR.** [PR #5434](https://github.com/flashinfer-ai/flashinfer/pull/5434#discussion_r4069329782)

> remove and only keep it for local benchmarking.

**Keep generated trace additions small and tied to the agreed interface.** [PR #5229](https://github.com/flashinfer-ai/flashinfer/pull/5229#discussion_r4016168274)

> I still think you are adding too many traces. simplify them with as few changes as possible.

### Naming and shared concepts

**Use consistent Q/KV tile names across kernels.** [PR #4599](https://github.com/flashinfer-ai/flashinfer/pull/4599#discussion_r3843782947)

> I would rename it as tile_size_q and tile_size_kv. please help clean other codes if needed. Thanks!

**Give Sage helpers a recognizable feature prefix.** [PR #6617](https://github.com/flashinfer-ai/flashinfer/pull/6617#discussion_r4344752759)

> I prefer adding sage prefix to all sage helpers to help people distinguish with other helpful functions.

**Name the capability, not the model.** [PR #5447](https://github.com/flashinfer-ai/flashinfer/pull/5447#discussion_r4078731898)

> let us not make specific to dsv4. I prefer sparse_mla. Please rename for all places.

**Represent generic mixed precision through compute-dtype concepts rather than a variant-specific config.** [PR #6296](https://github.com/flashinfer-ai/flashinfer/pull/6296#discussion_r4291814287)

> I prefer adding mma_qk_dtype and mma_pv_dtype to support this mixed precision kernels instead of adding specialized config for vc. this is more general and easier to be read. Similar for all the changes you have made. you can have some flag like "16bit_qk_8bit_pv_mma"

**PDL is an attention capability, not a metadata-producer-specific option.** [PR #4996](https://github.com/flashinfer-ai/flashinfer/pull/4996#discussion_r3946386392)

> we shouldn't use uses_qsa_metadata_pdl option. PDL should be a general feature that doesn't rely on metadata kernels. it is just use_pdl or not.

**Use booleans for boolean configuration.** [PR #5235](https://github.com/flashinfer-ai/flashinfer/pull/5235#discussion_r4023249285)

> let us use boolean config instead of integers.

**Encode the resource geometry explicitly; do not leave hidden assumptions in an ambiguous name.** [PR #4414](https://github.com/flashinfer-ai/flashinfer/pull/4414#discussion_r4043187348)

> oh, you are talking about the 128B leading dimension limitation. then we should make it clear by using num_head_dim_smem_tiles or something (feel free to rephrase it). remember to not make implicit assumption

### Preparation packages and reuse

**Share compatible Sage preparation/helpers across context and decode inside the feature package.** [PR #6617](https://github.com/flashinfer-ai/flashinfer/pull/6617#discussion_r4344735629)

> do we think this can be shared by context and decode. we can move it to sage directory if so.

**Move feature-owned host preparation out of the generic context adapter.** [PR #6617](https://github.com/flashinfer-ai/flashinfer/pull/6617#discussion_r4344772835)

> can we move those to sage directory ?

**Keep VC-attention preparation in its own feature directory.** [PR #6296](https://github.com/flashinfer-ai/flashinfer/pull/6296#discussion_r4344249319)

> how about we move it flashinfer/attention/prims_ts/vc_attention ? I want to keep the kernels clean.

**Organize reusable scheduling code by feature, with future GQA use in mind.** [PR #5235](https://github.com/flashinfer-ai/flashinfer/pull/5235#discussion_r4021937238)

> please create a sub-directory as flashinfer/attention/prims_ts/balanced_scheduler/, and put all those files under it. In the future, we might want to reuse the codes to support GQA decode.

**Extend the generator with existing inputs and mechanisms instead of duplicating a model-specific kernel.** [PR #5447](https://github.com/flashinfer-ai/flashinfer/pull/5447#discussion_r4078769472)

> can we reuse some parameters instead of always creating new ones ? for example, existing kv cache + one extra kv cache. Similar for other parameters, please clean them. Remember we are building a kernel "generator", not specific DS v4 kernels.

**Reuse an existing MMA task when the operation is unchanged.** [PR #5447](https://github.com/flashinfer-ai/flashinfer/pull/5447#discussion_r4078759832)

> similar here, can we reuse the mma fused task ? I don't think the mma task should make a difference.

**Check whether native reducers already own the required computation.** [PR #5434](https://github.com/flashinfer-ai/flashinfer/pull/5434#discussion_r4071570166)

> why we need a separate reduction kernel, can we reuse the original one ?

**Prefer supported primitives and isolate justified inline PTX in helpers.** [PR #6296](https://github.com/flashinfer-ai/flashinfer/pull/6296#discussion_r4291865524)

> you should move those inline ptx to helpers. and can you explain why existing nvvm ops cannot support them ? same for other instructions you have added.

### Policy and runtime ownership

**Avoid ad hoc batch-size-specific policy.** [PR #4414](https://github.com/flashinfer-ai/flashinfer/pull/4414#discussion_r3780663834)

> this seems hacky, we should make it general instead of relied on batch size (dynamic value).

**Replace unsupported threshold proliferation with a general model.** [PR #4830](https://github.com/flashinfer-ai/flashinfer/pull/4830#discussion_r3892254529)

> we seem to add many thresholds here. Can you remove them and make it general ?

**Select policy using the correct grouped-Q geometry before specializing KV geometry.** [PR #4474](https://github.com/flashinfer-ai/flashinfer/pull/4474#discussion_r3773009058)

> is it true that we will always use this config when the conditions are meet ? we should first run cost model to select tile size q (which considers both seqlen_q and head group size), and then decide tile size kv IIUC

**Treat repeated policy-key hashing as part of host overhead.** [PR #6925](https://github.com/flashinfer-ai/flashinfer/pull/6925#discussion_r4334738677)

> how we hash the key and is it costly ?

**Review whether live sizes belong in run rather than the model plan; follow this PR's actual contract.** [PR #6925](https://github.com/flashinfer-ai/flashinfer/pull/6925#discussion_r4335055316)

> same here, I assume we should remove batch size / max seqlen length from plan(). Do you see blockers/issues ?

**Support reusable caller-owned metadata/workspace instead of repeated allocation.** [PR #4771](https://github.com/flashinfer-ai/flashinfer/pull/4771#discussion_r3921649885)

> I think we should modify plan() to not allocate seqlens and workspace inside it. Instead, we should construct them once and pass it to plan() like other attention backends. Can you help modify that ? Thanks.

### Validation and synchronization

**Remove duplicated validation when the same contract is already checked at the owning boundary.** [PR #5144](https://github.com/flashinfer-ai/flashinfer/pull/5144#discussion_r4002572623)

> those checks seem duplicated as shown in flashinfer/attention/prims_ts/_block_sparse/runtime.py

**Guard dependency waits by the capability they implement and inspect related kernels.** [PR #6605](https://github.com/flashinfer-ai/flashinfer/pull/6605#discussion_r4321115580)

> I think this should be guarded by enable_pdl flag instead of restricting to split_kv ? do we see same issues for context and decode kernels ?

**Express additional data movement and synchronization through the TS resource model.** [PR #5123](https://github.com/flashinfer-ai/flashinfer/pull/5123#discussion_r4069290795)

> seems that you are loading from gmem to smem and from smem to regiters in softmax task at the same time, without explicit warp-specialization pipelines. This is confusing for TS, is it possible to use extra warps to implement this otherwise it could easily cause data races/dead locks without TS validation support.

**Instruction support is architecture-specific and not necessarily monotonic across newer chips.** [PR #5123](https://github.com/flashinfer-ai/flashinfer/pull/5123#discussion_r4068690820)

> I think INT8 MMAs are only supported with sm100, while removed with sm103 and next. we should guard it.

**Do not raise a shared dependency floor incidentally to test a new optimization.** [PR #4839](https://github.com/flashinfer-ai/flashinfer/pull/4839#discussion_r3931731397)

> we shouldn't modify package requirement directly for compatibility reasons as many repos still adopt 4.6.2.  we should only update the package version for testing. see https://github.com/Vinnie6167/flashinfer/blob/main/scripts/test_utils.sh#L297. @saltyminty @bkryu can correct me if I am wrong about the script.

### Explanations and performance evidence

**Explain each non-obvious branch and constant in terms of computation and memory ownership.** [PR #5434](https://github.com/flashinfer-ai/flashinfer/pull/5434#discussion_r4071479148)

> you should explain what each code block (if-else branches) is doing, and what are those constant values mean.

**Document policy selection and its tradeoffs.** [PR #5434](https://github.com/flashinfer-ai/flashinfer/pull/5434#discussion_r4071565615)

> for this policy, you should add more comments about how each configuration is chosen, what are the tradeoffs.

**Use the agreed regression/performance harness; qualification work does not all belong in the PR.** [PR #4414](https://github.com/flashinfer-ai/flashinfer/pull/4414#pullrequestreview-4933377221)

> Thanks for the contribution. Can you also help validating there is no perf regression with https://github.com/PerkzZheng/flashinfer-prims-ts-validation ? and please add mixed precision kernel regression tests to that repo.

**Keep PR performance reporting current and give explicit units.** [PR #5059](https://github.com/flashinfer-ai/flashinfer/pull/5059#pullrequestreview-5154705144)

> Finally, about the perf numbers in the MR description, you should only keep the latest prims_ts numbers + trtllm-gen, and please mention the unit (ms).

**Focus performance claims on the benefit of the newly added configuration versus the prior implementation.** [PR #5229](https://github.com/flashinfer-ai/flashinfer/pull/5229#issuecomment-5675470073)

> for "Standalone performance" part of the MR description, we should only mention how the newly added configurations (like G=8) helps with the performance compared to previous implementation.

## Follow-up source comments: PRs #7019 and #7026

### PR #7019: perf(prims-ts): prefetch page IDs for FP8 2CTA dense MLA K/V loaders

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7019#discussion_r4357863371) on `flashinfer/attention/prims_ts/kernels/mla_decode/throughput_2cta/config.py`

> should BF16 also have similar restriction ? ideally, this should not be restricted as we can pad it or even multiple k_tiles's page offsets in one stage, but I am okay with the current implementation to improve performance with fewer changes.

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7019#discussion_r4357878063) on `flashinfer/attention/prims_ts/kernels/mla_decode/throughput_2cta/kernel.py`

> seems that we previously only added prefetching for BF16 ?

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7019#discussion_r4357885034) on `flashinfer/attention/prims_ts/kernels/mla_decode/throughput_2cta/kernel.py`

> I was thinking that we also used prefetching for sparse attention kernels, but it might end up not having gains in those cases. Let me re-visit it later.

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7019#discussion_r4357992117) on `flashinfer/attention/prims_ts/kernels/mla_decode/throughput_2cta/resources.py`

> what does this mean ? does it mean we load one more page offset ahead in registers ?

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7019#discussion_r4373346207) on `flashinfer/attention/prims_ts/kernels/mla_decode/throughput_2cta/kernel.py`

> it makes sense as BF16 loading takes more latency. Thanks.

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7019#discussion_r4373378024) on `flashinfer/attention/prims_ts/kernels/mla_decode/throughput_2cta/kernel.py`

> okay, I took a further look and it seems that you are prefetching page indices by current warps to register buffers instead of smem buffers. I am good with this changes for now, but we can revisit smem buffers with producers later (i.e. reuse sparse mla's work if possible).

**Thread context.** The author [confirmed that BF16 and FP8 share the window-geometry assumption](https://github.com/flashinfer-ai/flashinfer/pull/7019#discussion_r4359468769) and applied the guard to both dense paths. Padding or packing multiple tiles is a possible extension, not proof that the present recipe can ignore its invariant.

The author [distinguished BF16 register-window caching from prefetch](https://github.com/flashinfer-ai/flashinfer/pull/7019#discussion_r4359451926) and reported no obvious gain from adding BF16 lookahead in that experiment. [Sparse MLA uses a separate producer warp](https://github.com/flashinfer-ai/flashinfer/pull/7019#discussion_r4358181425), whereas the accepted dense change uses existing loaders and register buffers. PerkzZheng accepted this scoped improvement and deferred possible SMEM/producer reuse; do not turn that follow-up into a mandatory rewrite or a universal prohibition on SMEM staging.

### PR #7026: feat(prims-ts): opt-in softmax_rescale_threshold_log2 (skip correction) for dense and sparse MLA decode

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7026#discussion_r4358026888) on `flashinfer/attention/prims_ts/mla_decode.py`

> this seems more like limiting the threshold to the upper bound ? maybe rename it.

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7026#discussion_r4358041134) on `tests/attention/test_attention_ts_mla_decode.py`

> this seems unnecessary.

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7026#discussion_r4358041464) on `tests/attention/test_attention_ts_mla_decode.py`

> this seems unnecessary.

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7026#discussion_r4358047668) on `tests/attention/test_attention_ts_mla_decode.py`

> what this test is used for ?

[Review comment](https://github.com/flashinfer-ai/flashinfer/pull/7026#discussion_r4373369386) on `flashinfer/attention/prims_ts/mla_decode.py`

> @yunruis It might be better to create a struct for tuning configs (make it extensive).
>
> for example,
> ```
> @dataclass(frozen=True)
> class MlaDecodeTuning:
>     """Compile-time fine-tuning for one PrimTS MLA decode plan.
>     ``softmax_rescale_threshold_log2`` is in binades. ``None`` keeps the
>     entry point's default: exact rescale for dense, and each sparse
>     schedule's existing deferral for sparse. An explicit value, including
>     ``0.0``, is compiled into the kernel.
>     """
>     softmax_rescale_threshold_log2: float | None = None
> ```

**Thread context.** The author [clarified that the range helper rejects rather than clamps](https://github.com/flashinfer-ai/flashinfer/pull/7026#discussion_r4359520839). Its name should describe that behavior; the reviewer's tentative “limiting” wording does not authorize changing numerical semantics.

The [range-matrix test](https://github.com/flashinfer-ai/flashinfer/pull/7026#discussion_r4359522098) and [separate validation-disabled bound test](https://github.com/flashinfer-ai/flashinfer/pull/7026#discussion_r4359526757) were removed. In contrast, the author [explained the re-anchor test's distinct branch](https://github.com/flashinfer-ai/flashinfer/pull/7026#discussion_r4359533044): random inputs exercised only the frozen anchor, so spiking late keys was needed to force re-anchoring. The test was renamed and related Keeps cases consolidated. This supports minimal branch-specific coverage, not deleting every test whose purpose was initially questioned.

The proposed frozen tuning struct groups compile-time fine-tuning and preserves unspecified versus explicit-zero semantics. It does not make live batch/sequence metadata part of tuning, nor does it waive the specialization-cost audit. The class name and threshold default are a concrete design example, not a required universal schema or proof that the proposal was already implemented at the retrieved head.
## Read the final discussion before generalizing

- **V padding is not a blanket prohibition on clearing.** The initial request to move work to the KV writer in [#4829](https://github.com/flashinfer-ai/flashinfer/pull/4829#discussion_r3946208438) was followed by a request for broader performance checks and [acceptance of optional tail handling](https://github.com/flashinfer-ai/flashinfer/pull/4829#discussion_r3964260638). Preserve the negotiated correctness contract.
- **Small sanitizer coverage can be necessary.** In [#5351](https://github.com/flashinfer-ai/flashinfer/pull/5351#discussion_r4067754289), the retained live-prefix initcheck case caught speculative reads that could still produce numerically correct output. This is distinct from a duplicated 128-route benchmark-like test that was removed.
- **Core unification was exploratory, not a prerequisite for every feature.** The [#5123 discussion](https://github.com/flashinfer-ai/flashinfer/pull/5123#issuecomment-5772967528) accepts adding context Sage separately; [#6617](https://github.com/flashinfer-ai/flashinfer/pull/6617#pullrequestreview-5609492763) asks about future interface/core convergence. Share coherent components without forcing a large rewrite into an unrelated PR.
- **A question about a bound or key is not a command to delete it.** [#6529](https://github.com/flashinfer-ai/flashinfer/pull/6529#discussion_r4289993377) distinguishes equivalent derived geometry from limits actually baked into validation code. Likewise, [#4771](https://github.com/flashinfer-ai/flashinfer/pull/4771#discussion_r3921642651) accepts retaining the explained limit for the current scope.
- **Scope changes can make a former abstraction unnecessary.** [#5434](https://github.com/flashinfer-ai/flashinfer/pull/5434#discussion_r4291916823) removes the separate final merge after agreeing on a shared-KV-scale contract. That decision is not proof that every cross-source finalizer is redundant under every scale contract.
- **Do not freeze an old tuning gate.** The reviews ask for structural policies but also [accept an explained customer-tuned static choice](https://github.com/flashinfer-ai/flashinfer/pull/5235#discussion_r4043154935). Specific 5% or 0.94 performance gates in historical reports are not universal acceptance thresholds.

The support-limit clarification also matches PerkzZheng's [page-size review in #6937](https://github.com/flashinfer-ai/flashinfer/pull/6937#discussion_r4344202129), which notes that a page-size-32 limitation can be relaxed, and [#5351](https://github.com/flashinfer-ai/flashinfer/pull/5351#discussion_r4056212656), which questions whether a fixed-query restriction is a real limitation and asks about variable or other fixed query lengths. These comments motivate investigating and qualifying the general case; they are not proof that every untested geometry is correct.

## Author clarification: preserve the useful tuning campaign (2026-10-25)

After seeing the rebased Rubin decode PR's benchmark table, PerkzZheng
questioned whether the PR retained only the Q64/KV256 specialization and
reminded the agent that multiple kernels, KV-stage choices and other ideas
had been tuned. The author confirmed requalifying the remaining stage-count
and prefetch candidates and retaining useful explicit profiles or justified
dispatch rules. This is working-session scope guidance: artifact cleanup must
not be presented as complete optimization consolidation when promising
candidates remain unqualified against the new base.

## Author clarification: commit identity and sign-off (2026-10-25)

After publication of the Rubin decode PR, PerkzZheng asked to sign off as
PerkzZheng instead and record that preference in the skill. Apply this to
authorized commits on his behalf: use his configured or established Git
identity for author and committer and include the matching `Signed-off-by`
trailer. GitHub rejected the repository-configured private email with GH007;
use his existing public GitHub noreply identity without changing privacy settings.
This is direct working-session guidance, not a public-review quote
or standing authorization to change unrelated contributors' history.

## Author clarification: placement-sensitive resource tuning (2026-10-25)

During the Rubin decode PR cleanup, current-main comparisons showed that
larger resource profiles helped localized runs while regressing ordinary
placement. Given the choice of removing those profiles or retaining them as
explicit opt-ins with conservative automatic defaults, PerkzZheng selected
“Keep explicit opt-in profiles.” This preference is specific to the requested
PR scope; it is not a claim that every resource profile improves performance.

## Author clarification: PR scope and publication (2026-10-25)

In an explicit cleanup-and-publication request, PerkzZheng asked to rebase onto
latest public FlashInfer main, keep only changes needed for the PR, exclude
unnecessary documents and local debugging scripts/tests, and use a short,
precise PR description. This is direct author guidance from the working
session, not a quotation from a public review thread.

Apply that preference by separating qualification artifacts from maintained
code and minimal regression coverage. Preserve concise documentation of real
requirements and changed contracts. Publication and updates to the skill
repository were explicitly authorized in that task; this is not standing
authorization for future review-only requests.

## Sage organization at the reviewed revision

At upstream main `eb38f470d04d1cf625313267349f3174db2866ce`, [the Sage package](https://github.com/flashinfer-ai/flashinfer/tree/eb38f470d04d1cf625313267349f3174db2866ce/flashinfer/attention/prims_ts/sage) contains `config.py`, `layout.py`, `prepare.py`, `index.py`, `modes.py`, `int32_scores.py` and `__init__.py`. Its package documentation explicitly shares these components between context and decode while each kernel retains its own resources, pipelines and softmax integration. Treat this as an ownership example, not a mandatory seven-file template.

## PRs from the initial collection containing written entries

| PR | Title | Nonempty entries |
| --- | --- | --- |
| [#4357](https://github.com/flashinfer-ai/flashinfer/pull/4357) | Add task-scheduled Blackwell attention kernels | 16 |
| [#4413](https://github.com/flashinfer-ai/flashinfer/pull/4413) | [prims-ts] Refactor FMHA decode padding-task scheduling | 1 |
| [#4414](https://github.com/flashinfer-ai/flashinfer/pull/4414) | [prims-ts] Mixed precision Fmha decode | 16 |
| [#4474](https://github.com/flashinfer-ai/flashinfer/pull/4474) | feat(attention): add PrimTS Q64/KV256 and paged GQA block-sparse attention | 6 |
| [#4599](https://github.com/flashinfer-ai/flashinfer/pull/4599) | Adding var-window FMHA context support for PrimsTS. | 8 |
| [#4664](https://github.com/flashinfer-ai/flashinfer/pull/4664) | fix(attention): make PrimTS paged block-sparse metadata live | 4 |
| [#4771](https://github.com/flashinfer-ai/flashinfer/pull/4771) | feat(prims-ts): CUDA graph mode for the prims-ts decode backend | 4 |
| [#4829](https://github.com/flashinfer-ai/flashinfer/pull/4829) | refactor(attention): unify PrimTS plan/run and page-table contracts | 19 |
| [#4830](https://github.com/flashinfer-ai/flashinfer/pull/4830) | feat(prims_ts): support no-padding MLA query rows | 15 |
| [#4839](https://github.com/flashinfer-ai/flashinfer/pull/4839) | feat(attention): opt-in LDTM.STAT for PrimTS context row_max | 4 |
| [#4872](https://github.com/flashinfer-ai/flashinfer/pull/4872) | feat(prims-ts): support proxy-compensated block-sparse attention | 12 |
| [#4879](https://github.com/flashinfer-ai/flashinfer/pull/4879) | feat(prims-ts): support QK-BF16/PV-FP8 in FMHA context | 7 |
| [#4915](https://github.com/flashinfer-ai/flashinfer/pull/4915) | feat(prims-ts): support decode GQA ratios up to 128 | 2 |
| [#4996](https://github.com/flashinfer-ai/flashinfer/pull/4996) | feat(prims-ts): add QToken-KvBlock-Sparse-Attention | 50 |
| [#5002](https://github.com/flashinfer-ai/flashinfer/pull/5002) | perf(prims-ts): Optimize&refine PrimsTS block sparse attention | 3 |
| [#5059](https://github.com/flashinfer-ai/flashinfer/pull/5059) | feat(prims-ts): add non-absorbed MLA prefill | 23 |
| [#5078](https://github.com/flashinfer-ai/flashinfer/pull/5078) | draft. deprecated | 1 |
| [#5082](https://github.com/flashinfer-ai/flashinfer/pull/5082) | [None][refactor] Mark PrimTS attention APIs as experimental | 1 |
| [#5123](https://github.com/flashinfer-ai/flashinfer/pull/5123) | feat(prims-ts): 8-bit Q/K/V sage attention for the PrimTS block sparse decode kernels | 14 |
| [#5144](https://github.com/flashinfer-ai/flashinfer/pull/5144) | refactor(attention): use fixed page tables for paged block-sparse attention | 1 |
| [#5229](https://github.com/flashinfer-ai/flashinfer/pull/5229) | feat(prims-ts): extend QToken-KvBlock-Sparse-Attention shapes and patterns | 44 |
| [#5235](https://github.com/flashinfer-ai/flashinfer/pull/5235) | feat(prims-ts): Add balanced MLA decode scheduling | 17 |
| [#5351](https://github.com/flashinfer-ai/flashinfer/pull/5351) | perf(prims-ts): optimize QToken-KvBlock-Sparse-Attention | 30 |
| [#5434](https://github.com/flashinfer-ai/flashinfer/pull/5434) | feat(prims-ts): add native two-source sparse MLA | 68 |
| [#5447](https://github.com/flashinfer-ai/flashinfer/pull/5447) | feat(attention): add PrimTS DSV4 sparse MLA kernels | 22 |
| [#5478](https://github.com/flashinfer-ai/flashinfer/pull/5478) | fix(prims-ts): support CuTe DSL 4.7 and 4.8 | 1 |
| [#5720](https://github.com/flashinfer-ai/flashinfer/pull/5720) | feat(prim_ts): Add H512 FP8-Q/NVFP4-KV decode support | 11 |
| [#6296](https://github.com/flashinfer-ai/flashinfer/pull/6296) | feat(prims-ts): PrimsTS VC-attention-QK16 for dense FMHA context | 23 |
| [#6529](https://github.com/flashinfer-ai/flashinfer/pull/6529) | feat(prims-ts): cache attention kernels and enable FP8-to-BF16 decode | 13 |
| [#6605](https://github.com/flashinfer-ai/flashinfer/pull/6605) | fix(attention/prims_ts): MLA split-KV producer must wait on its PDL predecessor; QToken default sm_scale | 3 |
| [#6609](https://github.com/flashinfer-ai/flashinfer/pull/6609) | fix(prims_ts): QToken wrapper default sm_scale must not inherit an earlier run's scale | 2 |
| [#6617](https://github.com/flashinfer-ai/flashinfer/pull/6617) | feat(prims-ts): Port sage attention to the context kernel | 9 |
| [#6774](https://github.com/flashinfer-ai/flashinfer/pull/6774) | feat(prims-ts): support D192 sparse and sliding-window attention | 9 |
| [#6925](https://github.com/flashinfer-ai/flashinfer/pull/6925) | feat(prims-ts): broaden attention support and runtime split-KV | 34 |
| [#6937](https://github.com/flashinfer-ai/flashinfer/pull/6937) | perf(prims-ts): shorter Keeps softmax chain and grouped-Q paged selection for decode | 5 |
