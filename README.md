# FlashInfer Prims-TS review skill

A reusable agent skill for reviewing FlashInfer's task-scheduled attention changes, distilled from [PerkzZheng's public PR reviews](skills/review-prims-ts-attention/references/review-evidence.md). It covers context, decode, MLA, block-sparse, Sage and Q-token sparse attention.

The skill emphasizes:

- Keeping local validation probes and benchmark campaigns out of PRs while retaining minimal, meaningful accuracy and regression coverage.
- Consistent names across kernels and avoiding leading underscores where possible.
- Separating variant preparation into coherent packages such as `prims_ts/sage/` while reusing core kernel mechanisms.
- Distinguishing genuine hardware/implementation limitations from missing validation, including arbitrary page-size restrictions.
- Auditing new static compilation parameters, especially batch size and sequence lengths, for specialization growth and real-inference JIT cost, with an explicit user workload check.
- Explicit ownership, synchronization, numerical contracts and comparable performance evidence.

## Install in Codex

Ask Codex:

```text
Use $skill-installer to install the skill from:
https://github.com/PerkzZheng/flashinfer-prims-ts-review-skill/tree/main/skills/review-prims-ts-attention
```

Or, if the standard skill installer is available in your Codex skill directory:

```bash
python "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-installer/scripts/install-skill-from-github.py" \
  --repo PerkzZheng/flashinfer-prims-ts-review-skill \
  --path skills/review-prims-ts-attention
```

The installer places it in `$CODEX_HOME/skills/review-prims-ts-attention` (default: `~/.codex/skills/review-prims-ts-attention`) and refuses to overwrite an existing installation. It is available on the next turn. You can also copy that skill folder into your configured skill directory. The package has no dependency on the author's local review skill or workspace.

## Use

```text
Use $review-prims-ts-attention to review the current branch against main.
Report findings without modifying files. Focus on test scope, naming,
preparation boundaries and whether unsupported combinations have a real constraint.
```

For an authorized cleanup:

```text
Use $review-prims-ts-attention to address the review comments on this branch.
Keep temporary validation scripts local, preserve meaningful correctness coverage,
and explain implementation limits separately from missing qualification.
```

Other agents can read the [SKILL.md entry point](skills/review-prims-ts-attention/SKILL.md) and its references directly. The guidance itself needs no GPU or package installation; executing kernel validation still requires the appropriate repository environment and hardware.

## Package

```text
skills/review-prims-ts-attention/
  SKILL.md
  agents/openai.yaml
  references/review-guidance.md
  references/review-evidence.md
```

[Review guidance](skills/review-prims-ts-attention/references/review-guidance.md) contains the detailed decision rules. [Review evidence](skills/review-prims-ts-attention/references/review-evidence.md) records public sources, context and the reviewed PR inventory. The evidence was collected on 2026-10-22; historical comments do not define current API support, and current task instructions take precedence.

## License

Original skill documentation is available under the [MIT license](LICENSE). Review excerpts are attributed to their source PRs; linked third-party material retains its own licensing.
