# Pre-registration — cross-lingual quantisation ladder (`xtile-1`)

Frozen before any score was computed. Hash of the frozen block, as emitted by
`scripts/xtile_ladder.py --print-prereg-hash`:

```
626ab3585beea8e64803ca0d12b96eef12a89f11daa0cd4f91fa60d822e28b92
```

Timestamp of freezing: 2026-11-03 13:07 UTC, on machine 175. Nothing in this file may be
changed afterwards; a change means a new protocol version (`xtile-2`) and a new hash.

## 1. The one question

On Indonesian e-CARE/XCOPA items, the 0.6B ladder loses most of its letter-scoring accuracy
between the `Q4_K_M` and `Q4_0` rungs. **Does that cliff sit in the same place when the item
language changes and the prompt frame, the scoring protocol and the models do not?**

Nothing else is being tested here. No new model, no new training, no new metric.

## 2. Data

`cambridgeltl/xcopa`, configurations `id`, `vi`, `th`. Each configuration ships exactly
`validation` (100) and `test` (500); there is no `train`.

| rule | frozen statement |
|---|---|
| language choice | `vi` (Latin script, dense diacritics, tone-bearing lexemes) and `th` (non-Latin script, no inter-word spaces) are the two most distant orthographies available in XCOPA; `id` stays in the design as the home-language anchor. Other configurations (`et ht it qu sw ta tr translation-*`) are an extension, not part of this test. |
| item selection | **every item of the split**, in dataset order. No balancing, no sampling, no dropped indices (deliberately different from `eval_rag.load_items("xcopa_val")`, which drops 3). |
| pairing | the `idx` field of the per-language configurations is identical across languages (verified: `idx_aligned_across_langs = true` for validation), so item *i* is the same underlying COPA item in all three languages and all comparisons are item-for-item. |
| label use | labels enter only through the frozen metrics below; no item is chosen, excluded or reordered on the basis of a label. |
| sealed split | `validation` (100/language) is the development slice and was used first. `test` (500/language) is read **exactly once**, in one command, writing one JSON. `scripts/xtile_ladder.py` refuses a second test pass because a result file already exists. |

## 3. Prompt frame — two conditions, both constant across languages

`FRAME_SHAPE = "{head}{premise}\n{question}\nA. {c_left}\nB. {c_right}\n{tail}"`

| frame | head | question | tail |
|---|---|---|---|
| `id` | `Premis: ` | `Apa {w} yang paling mungkin?` | `Jawaban:` |
| `en` | `Premise: ` | `What is the most likely {w}?` | `Answer:` |

`{w}` per item question type: `cause → penyebab / cause`, `effect → akibat / effect`.

The `id` frame is the template the ladder numbers in the paper were produced with; the script
asserts, for every Indonesian item, that its own rendering equals
`eval_content.prompt_pair(ex, question_word(ex))` byte-for-byte. `en` is its literal
translation and exists only as the frame-robustness condition. Both frames are applied to all
three languages, so "item language" and "frame language" vary independently.

## 4. Scoring protocol (imported, not reimplemented)

`CANDS = [" A", "A", " B", "B"]`; per-letter score = max of the two surface forms; margin =
s_B − s_A; each item is scored once as written and once with the options swapped;
`acc_order_averaged` is the primary metric, `auc_order_averaged` and `order_consistency`
secondary; `bootstrap_ci` (2000 resamples, seed 0) per rung; exact two-sided McNemar for every
paired contrast. All of these are imported from `eval_gguf` / `eval_content` /
`probe_teacher_format`, which are the already-used implementations.

Runner: host `gguf_cand_lp` (x86), `--threads 8`, `n_ctx` = the smallest power of two ≥
(max prompt tokens + 2), computed by the teacher tokeniser before any model is scored. For
validation that rule gives **128** (max 96 tokens, `th`/`id` frame). Measured token means:
`id/id` 56.6, `vi/id` 50.3, `th/id` 62.5, and 44–57 for the `en` frame.

## 5. Models

Rungs come from the existing GGUFs; no requantisation. Ladder position is ordered by **bytes on
disk**, which is what the size-vs-accuracy claim is about.

| family | rungs |
|---|---|
| `n06ctl` (primary) | IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XXS IQ3_S IQ3_M IQ4_XS Q2_K Q2_K_imx Q3_K_M Q4_0 Q4_K_M Q8_0 f16 |
| `n06kl`, `n06mid` (student robustness) | IQ2_XXS IQ2_S IQ3_M IQ4_XS Q4_0 Q4_K_M f16 |
| `base` = un-finetuned Qwen3-0.6B (capacity/language reference) | Q4_0 Q4_K_M f16 |

## 6. Analysis, contrasts, and the decision rule

* **Primary contrast**, per (family, language, frame): `d_acc = acc(Q4_0) − acc(Q4_K_M)`, with
  exact McNemar on the order-averaged predictions and a bootstrap 95 % CI of the paired
  difference. Holm correction across the 6 tests of the primary family
  (`id`,`vi`,`th` × `id`,`en`); the other families are reported uncorrected as robustness.
* **Cliff location**, secondary: the largest negative adjacent drop in the size-ordered ladder.
  "The cliff reproduces in language L" requires *both* (i) `d_acc` at the Q4_K_M→Q4_0 pair to be
  the worst adjacent drop of that ladder, and (ii) the same sign as in `id`, with Holm-adjusted
  p < 0.05.
* **Floor-effect guard, pre-declared**: if `acc(Q4_K_M)` in a language has a bootstrap CI whose
  upper bound is ≤ 0.5, that language's cliff is reported as **undefined** ("there is no signal
  for a cliff to remove"), *not* as "the cliff moved" and not as a replication. This is the
  expected failure mode for the Indonesian-finetuned students and it is a result, not a bug.
* **Trim rule, pre-declared**: a family is dropped from the test pass only if it is at chance
  (CI upper bound ≤ 0.5) in *every* language on validation. The decision is recorded before the
  test pass.
* No post-hoc item exclusion, no additional frame, no additional language, no different metric,
  no change of `n_ctx` after results are seen. Reporting more or fewer of the numbers that this
  design already computes is allowed; computing new numbers is not.

## 7. Budget

Measured host cost: ~13 s per (model × language × frame × 200 prompt lines), model load
included. Validation = 32 models × 6 cells ≈ 45 min; the single test pass is 5× larger
(≈ 3.5–4 h), CPU-only, on 175.

## 8. Amendments, both written before any result was read

*(2026-11-03 13:35 UTC. At the time of writing, the validation probe was still running and
the only output visible was the first family's two 2-bit rungs, both at chance — which is what
§6's floor guard already predicted. No `test` item has been touched. The protocol hash above
still covers the frozen block; these amendments add clarity and one statistic, they change no
protocol string, no model, no metric, and no decision rule.)*

1. **What this design can and cannot attribute.** The ladder changes two things at once relative
   to the published cliff: the *item set* (XCOPA instead of e-CARE dev) and the *item language*.
   The `id` column exists to separate them: if the cliff is not at Q4_K_M→Q4_0 on XCOPA-`id`,
   then a vi/th result cannot be attributed to language at all, and the correct report is "the
   cliff is dataset-specific", not "the cliff is language-specific". This reading order is
   binding: **establish the cliff on XCOPA-id first.**
2. **Added cross-language statistic** (possible because the per-language `idx` fields are
   identical, §2): for every family and frame, the *cliff depth* on the same 500 items can be
   differenced between languages, `depth_minus_id = d_acc(L) − d_acc(id)` with a paired
   bootstrap CI over items (2000 resamples, **seed 2**). This is the statistic that answers
   "does the cliff *move*", as opposed to "is the cliff significant in L"; §6's decision rule
   still governs the reproduction claim, and this number is reported alongside it for every
   language, frame and family. Implementation: `xtile_ladder.paired_diff_ci`. The validation
   JSON written before this amendment lacks the field, so validation is re-run once with the
   amended script — repeating the non-sealed slice costs nothing in discipline terms, repeating
   `test` would not.
3. **Orthography note.** Thai validation is 100 % `effect` questions while Vietnamese and
   Indonesian are ~52/48 cause/effect (§2), so a `th` result is simultaneously a script change
   and a question-type change; both are visible in the frozen `idx`/labels arrays and neither
   can be adjusted away without new data. Recorded as a limitation, not analysed around.

## 9. Pre-results implementation corrections (logged after §8, before any cell was read)

Two defects were found in the scoring harness itself, not in the protocol, and both were
fixed before a single result existed to be influenced by the fix. The canonical-JSON hash of
§3 is unchanged by them (`626ab358…e28b92`), which is the check that they touched no
protocol constant.

1. **Prompt-row ordering.** The first validation batch (192 cells) staged its prompts
   interleaved, `(as-written, swapped, as-written, …)`, while `eval_gguf.letter_scores`
   splits the row list at `n_items` and therefore expects the blocked layout that
   `eval_gguf` itself uses. Interleaving pairs each row with the wrong item's label; the
   failure is silent and shows up only as accuracy collapsing toward chance. That batch was
   voided and quarantined on disk as `xtile_validation.INVALID_interleaved_rows.json`
   (nothing deleted, nothing read as a result). `test` was never touched.
2. **End-to-end anchor regression.** A per-item byte-level assertion on the `id` frame passed
   throughout the defect above, because it checks prompt *text* and cannot see row *order*.
   The scoring path therefore now begins with `anchor_check()`: `gguf_a/qwen3_0.6b_f16.gguf`
   on `xcopa_val` (97 items) must reproduce `acc = 0.711 ± 0.005`, the value `eval_gguf`
   published for that model on 2026-11-03, or scoring aborts. The measured value is recorded
   in the result JSON under `meta.anchor_acc`. This is a regression gate, not a hypothesis.
3. **Concurrency.** Cells may run as concurrent `gguf_cand_lp` subprocesses (`--jobs`). Each
   cell keeps `--threads 8` and its own model/process, so concurrency changes wall-clock
   scheduling only, not any scored quantity.
