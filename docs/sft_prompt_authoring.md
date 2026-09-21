# SFT prompt authoring guide (Foundation-1 instruction tuning)

This is the operator-facing companion to ADR
[003-sft-dataset-bootstrap](decisions/003-sft-dataset-bootstrap.md). Read that
ADR first for the why; this page is the how for whoever is actually writing
prompts.

## Why this exists

Foundation-1's pre-training data is "bags of metadata + sounds" (instrument +
tag list). Users type full sentences with compositional structure - "vintage
synth with a loud buzz" - and the metadata bag throws that structure away: it
has no way to say *which* tag the word "loud" modifies. This is the missing
instruction-tuning stage (see AI Synthesizer co-founder Ben Goertzel's
diagnosis, 2025-10-22 Istanbul session, mirrored in dn-intelligence
`06_phases/post-training.md`). The fix is a small, hand-authored dataset of
**(full prompt, matching sound)** pairs, fine-tuned on top of pre-training,
the same way instruction tuning turns a base LLM into a chatbot.

## Who should write these

Someone "good with language and good with sounds" - not a database thinker.
The job is: listen to a sound, then write the sentence a user would type to
ask for that sound. It is not tagging. If you catch yourself listing nouns
("synth, vintage, buzz") instead of writing a sentence a person would type,
stop and rewrite it as a sentence.

## Sourcing route

Reverse-engineer descriptions from existing corpus sounds (Ben's suggested
method): pull a sound that already exists, listen to it, and write the
natural-language prompt that should produce it. Do not invent prompts with no
matching audio - every row needs a real (or renderable) sound on the other
end, or the pair teaches the model nothing.

## Sizing and sequencing

- **Target: a few hundred pairs to start, ~2,000 as the eventual size.**
  Start small, fine-tune, listen to the result, and decide whether more
  pairs or better coverage is the bottleneck before scaling up.
- **Qualifier count first, not instrument coverage first.** Single-qualifier
  prompts ("piano") should already work from the metadata path - if they
  don't, something upstream is broken and no amount of SFT data fixes it.
  Prioritize **two-qualifier combinations** ("vintage synth", "whiny
  violin") before moving to three-plus. This also sets up the curriculum
  learning stage (ADR 003) - the dataset should be taggable by qualifier
  count from day one, not relabeled later.
- Spread pairs across the nine governed instrument families (dn-intelligence
  `07_decisions/0014-nine-core-family-taxonomy.md`: Piano, Guitar, Bass/Sub,
  Brass, Flute, Synth, Pad, Pluck, Drums/Perc) so no single family dominates
  the fine-tune.

## Descriptor coverage taxonomy

Track which descriptor *category* each prompt exercises, not just which
words it uses - the goal is breadth of composition, not vocabulary size.
Six categories, drawn directly from the kinds of query Ben's session and the
Figma DN Strategy board used as examples:

| Category | What it covers | Example words |
|---|---|---|
| Timbre | Tone colour / brightness / warmth | vintage, warm, harsh, bright, dark |
| Envelope | Attack / decay / sustain / release shape | plucky, sustained, staccato, swelling |
| Articulation | How notes are played/connected | legato, choppy, glide, stabbed |
| Noise / artefact | Deliberate grit, not a defect | buzz, distortion, grit, hiss |
| Era / style | Period or genre reference | vintage, 80s, modern, lo-fi |
| Instrument / character | The base sound identity | synth, violin, whiny, thin |

A well-formed two-qualifier prompt combines **at least two different
categories** (e.g. "vintage" = era, "loud buzz" = noise, on a Synth
instrument). Prompts that only ever combine words from one category (three
timbre words, say) don't test compositionality the way the model actually
needs, so bias authoring toward cross-category combinations.

## Format

Follow the existing Foundation-1 WAV+JSON sidecar convention (see
`README_DNL.md` "Data Format" and `stable_audio_tools/data/sidecar_metadata.py`)
rather than inventing a new one - the SFT set is just a small, hand-curated
`train/`/`valid/`/`test/` directory fed through the same loader:

```json
{
  "text": "vintage synth with a loud buzz",
  "seconds_start": 0,
  "seconds_total": 3
}
```

Two differences from the bulk pre-training prompt format
(`{Instrument}, {timbre_tags}, {description}, {key}`):

1. `text` is a **full natural-language sentence**, not a comma-joined tag
   list - that compositional structure is the entire point of this dataset.
2. Add (outside the required `text`/`seconds_start`/`seconds_total` fields,
   as extra sidecar keys the loader ignores but tooling can read)
   `qualifier_count` and `descriptor_categories` so the curriculum stage in
   ADR 003 can select training subsets without re-parsing every prompt:

```json
{
  "text": "vintage synth with a loud buzz",
  "seconds_start": 0,
  "seconds_total": 3,
  "qualifier_count": 2,
  "descriptor_categories": ["era", "noise"]
}
```

## What "done" looks like for a batch

Before a batch of authored pairs is merged into the training set:

- Every row has a real sound file next to it (no orphaned prompts).
- At least two-thirds of two-qualifier-or-higher prompts span more than one
  descriptor category (see table above).
- A second person spot-checks a sample by listening blind (prompt hidden,
  guess it from the sound, compare) - if they can't reconstruct something
  close to the original prompt, the prompt is probably vague and should be
  rewritten, not just added anyway.

## Related

- ADR [003-sft-dataset-bootstrap](decisions/003-sft-dataset-bootstrap.md) - the decision this implements.
- `docs/conditioning.md` - how the `text` field is consumed downstream.
- `docs/datasets.md` - dataset loading mechanics.
- dn-intelligence `06_phases/methodology/post-training-methods.md` and
  `06_phases/post-training.md` - the company-wide SSOT record of the full
  SFT -> RLHF -> curriculum -> hybrid programme this dataset is step one of.
