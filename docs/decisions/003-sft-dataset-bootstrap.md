# ADR-003: Bootstrap a hand-authored SFT dataset before any RLHF/curriculum work

## Status

Proposed (2026-09-21)

## Context

- AI Synthesizer's current training data is "bags of metadata + sounds":
  audio paired with a flat instrument + tag list. Users query with full,
  compositional sentences ("vintage synth with a loud buzz"), and the
  metadata bag has no way to represent which word modifies which tag - the
  model never learns to use modifiers correctly, only to associate tags with
  sounds independently of their relationship in a sentence.
- This diagnosis is not new: co-founder Ben Goertzel gave it in detail in the
  2025-10-22 Istanbul session (transcript ingested into dn-intelligence,
  `08_meetings/fundraising/2025-10-22-ben-goertzel-22-10.md`), it is recorded
  as the company's adopted post-training direction in
  `06_phases/post-training.md` and `06_phases/methodology/post-training-methods.md`,
  and it was independently corroborated by the Figma DN Strategy board. All
  three sources land on the same four-stage plan: **SFT -> RLHF -> curriculum
  learning -> hybrid model (optional)**.
- Stage 2 (human feedback) is already partially live: since 2026-06-27,
  in-app thumbs up/down feedback is persisted and joined against QC scores
  (`dnl-core-backend` `feedbacks`/`sounds` tables) to calibrate the QC
  rejection gate. That is valuable but is **not** the RLHF Ben described -
  it recalibrates a rejection threshold, it does not change what the
  generation model itself learns to do with a prompt.
- Ben's own sequencing is explicit: RLHF should be **gated on the model
  already being right roughly half the time** on the prompts being rated.
  Below that, the reward signal is too sparse to learn from ("if someone
  only ever says bad-bad-bad you learn nothing"). This repo has no SFT
  dataset yet, so that gate has not been reached, and nothing here should be
  read as proposing full-scale RLHF training now.
- This repo already has the exact data format instruction-tuning needs
  (WAV+JSON sidecar pairs with a `text` field, `README_DNL.md` "Data Format")
  and a fine-tuning entrypoint that continues from a checkpoint
  (`train.py --pretrained_ckpt_path`, used by
  `scripts/clearml/train_f1_gpu4.sh` / `scripts/gcp_train_f1_3s.sh`). No new
  training machinery is needed, only a new, small, hand-curated dataset fed
  through the existing pipeline.

## Decision

1. **Bootstrap the SFT dataset by reverse-engineering prompts from existing
   corpus sounds**, per Ben's suggested method and
   `docs/sft_prompt_authoring.md` (this ADR's companion doc): a human good
   with language and sound listens to a sound and writes the prompt a user
   would type for it. Target a few hundred pairs first, sized toward
   roughly 2,000 as the eventual set, re-evaluating after the first
   fine-tune rather than committing to a fixed count up front.
2. **Prioritize two-qualifier prompts before three-plus.** Single-qualifier
   prompts are assumed already solvable by the metadata path; qualifier
   count becomes the ADR-003 curriculum axis for the later RLHF/curriculum
   stage, so the dataset should carry a `qualifier_count` field from
   authoring time (see the sidecar format in `docs/sft_prompt_authoring.md`)
   rather than needing to be re-annotated later.
3. **Reuse the existing WAV+JSON sidecar format and `train.py` fine-tuning
   path** - store the SFT set as its own small `train/`/`valid/`/`test/`
   tree, fine-tuned from the current Foundation-1 checkpoint via
   `--pretrained_ckpt_path`, not as a change to the pre-training pipeline.
4. **Do not build RLHF collection or curriculum-learning infrastructure yet.**
   Those are stage 2 and stage 3 of the programme and are explicitly gated on
   this stage succeeding (roughly 50% prompt-match hit rate on held-out
   two-qualifier prompts). Revisit once that bar is measured.
5. **No fixed held-out evaluation set exists in this repo today.** Before
   the first SFT fine-tune ships, carve out a stable, versioned held-out
   split of authored pairs (not reused for training) so subsequent fine-tunes
   can be compared against a fixed target - this repo currently only has the
   generic `train`/`valid`/`test` directory convention with no documented
   held-out policy for a small hand-authored set.

## Alternatives considered

| Approach | Pros | Cons | Outcome |
|---|---|---|---|
| Go straight to in-app RLHF collection now | No dataset-authoring effort | Ben's own gating logic says this wastes signal below ~50% hit rate; current hit rate on compositional prompts is visibly below that (see the Istanbul demo transcript - "vintage synth with a loud buzz" needed several regenerations) | Rejected for now |
| Auto-generate SFT prompts from existing tags (template-expand tag lists into sentences) | Fast, no human authoring time | Reproduces exactly the metadata-bag structure the SFT stage exists to escape - a templated sentence has the same flat, non-compositional structure as the tags it was built from | Rejected |
| Train a direct sentence-to-sound model now (skip straight to the "hybrid" stage) | Sidesteps the metadata bottleneck entirely | Ben's own view: worth prototyping in parallel eventually, but it is stage 4, an alternative/complement to a working metadata pipeline, not a replacement for fixing stage 1 first | Deferred, not rejected - revisit after SFT + RLHF are running |

## Consequences

- A small, versioned SFT dataset becomes a first-class artifact in this repo
  (own directory tree, own docs page), separate from bulk pre-training data.
- The eventual RLHF stage inherits a `qualifier_count`/`descriptor_categories`
  vocabulary from this dataset's authoring convention, so curriculum staging
  does not require a second annotation pass.
- This does not change `create_dataloader_from_config`, the pre-encoded
  latents path, or any production training config - it is additive (a new
  data tree + a fine-tune invocation), not a change to existing pipelines.

## References

- `docs/sft_prompt_authoring.md` - the operator-facing authoring guide this
  ADR is the rationale for.
- `README_DNL.md` "Data Format" - the WAV+JSON sidecar convention being
  reused.
- `train.py` `--pretrained_ckpt_path` - the fine-tuning entrypoint.
- dn-intelligence `06_phases/methodology/post-training-methods.md` - the
  company-wide four-stage reference (SFT -> RLHF -> curriculum -> hybrid).
- dn-intelligence `06_phases/post-training.md` - the DN-specific programme,
  the ~50% hit-rate RLHF gate, and the Istanbul transcript source.
- dn-intelligence `07_decisions/0014-nine-core-family-taxonomy.md` - the
  nine instrument families referenced for authoring coverage.

## Verification (repeatable)

No training run is required to validate this ADR - it is a data-authoring
and process decision. Verification is:

```bash
# Confirm authored pairs validate against the existing sidecar schema
python3 -c "
import json
from pathlib import Path
# Point at wherever the first authored batch lands, e.g. data/sft_v1/train
for path in Path('data/sft_v1/train').glob('*.json'):
    record = json.loads(path.read_text())
    assert 'text' in record and record['text'], f'{path} missing text'
    assert isinstance(record.get('qualifier_count'), int), f'{path} missing qualifier_count'
print('SFT sidecar records look well-formed.')
"
```

This ADR does not itself add the `data/sft_v1/` tree - that is the dataset
authoring work it authorizes, tracked separately once authoring begins.
