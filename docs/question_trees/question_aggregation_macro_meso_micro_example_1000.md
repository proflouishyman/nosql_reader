# Macro → Meso → Micro Aggregation Examples (1000-doc runs)

This file shows concrete examples of how question aggregation is currently working in adaptive mode for the two latest 1000-document runs.

- OFF run created: `2026-03-05T23:14:00.966333`
- ON run created: `2026-03-06T04:01:04.650734`

## How to read the counts

- `direct_doc_count`: documents directly linked to that node via `EvidenceLink.question_id`
- `scope_doc_count`: documents linked to that node or any downstream nodes in its local graph neighborhood
- `macro_meso_micro overlap`: documents shared across all three tiers in the example

Note: in these runs, aggregation is mostly semantic/coverage-based rather than explicit directed `macro -> meso -> micro` edges.

## Example 1 (OFF run)

### Macro
- Question: `How did unionization affect the outcomes of injury claims among different worker groups on railroads?`
- Direct docs: `148`
- Scope docs: `148`
- Sample evidence blocks:
  - `6939d8073334b77a9b7f0fae::b0`
  - `6939d8063334b77a9b7f0ea0::b0`
  - `6939d8073334b77a9b7f0f81::b0`
  - `6939d8073334b77a9b7f0f65::b0`

### Meso
- Question: `How did unionization influence changes and continuities in injury claim outcomes among different worker groups on railroads between 1862 and 1935?`
- Direct docs: `0`
- Scope docs: `148`
- Interpretation: this node is acting as an organizing/synthesis layer; evidence is carried by aligned child/peer nodes.

### Micro
- Question: `How did injury compensation differ across occupations within the Baltimore & Ohio Railroad Company, and why?`
- Direct docs: `59`
- Scope docs: `63`
- Sample evidence blocks:
  - `6939d8073334b77a9b7f0f75::b0`
  - `6939d8063334b77a9b7f0ef8::b0`
  - `6939d8073334b77a9b7f0fa1::b0`
  - `6939d8063334b77a9b7f0ec2::b0`

### Alignment / roll-up
- Macro ∩ Meso docs: `148`
- Meso ∩ Micro docs: `59`
- Macro ∩ Micro docs: `59`
- Macro ∩ Meso ∩ Micro docs: `59`
- Sample shared docs across all 3:
  - `6939d8053334b77a9b7f0aad`
  - `6939d8053334b77a9b7f0abf`
  - `6939d8053334b77a9b7f0ac2`
  - `6939d8053334b77a9b7f0aca`

## Example 2 (ON run, demographic orientation)

### Macro
- Question: `How did the experiences of railroad injuries differ between skilled and unskilled workers over time?`
- Direct docs: `165`
- Scope docs: `165`
- Sample evidence blocks:
  - `6939d8073334b77a9b7f0fae::b0`
  - `6939d8073334b77a9b7f0fcf::b0`
  - `6939d8073334b77a9b7f0f65::b0`
  - `6939d8073334b77a9b7f0f84::b0`

### Meso
- Question: `How did the patterns of railroad injuries among skilled and unskilled workers change or remain consistent between 1882 and 1935, as evidenced in archival records?`
- Direct docs: `0`
- Scope docs: `165`
- Interpretation: same pattern as OFF; meso is primarily a synthesis bridge with coverage inherited from linked evidence-bearing nodes.

### Micro
- Question: `How did the injury experiences and compensation outcomes differ between carmen like Luigi Mancuso and other occupations within the Baltimore & Ohio Railroad Company?`
- Direct docs: `41`
- Scope docs: `84`
- Sample evidence blocks:
  - `6939d8063334b77a9b7f0cf1::b0`
  - `6939d8063334b77a9b7f0cc9::b0`
  - `6939d8063334b77a9b7f0c59::b0`
  - `6939d8063334b77a9b7f0cfb::b0`

### Alignment / roll-up
- Macro ∩ Meso docs: `165`
- Meso ∩ Micro docs: `41`
- Macro ∩ Micro docs: `41`
- Macro ∩ Meso ∩ Micro docs: `41`
- Sample shared docs across all 3:
  - `6939d8053334b77a9b7f0c24`
  - `6939d8053334b77a9b7f0c26`
  - `6939d8053334b77a9b7f0c28`
  - `6939d8053334b77a9b7f0c2d`

## What this shows about current aggregation

1. Macro questions are broad evidence buckets with high direct coverage.
2. Meso questions often function as synthesis bridges (`direct_doc_count = 0`, high `scope_doc_count`).
3. Micro questions hold the most concrete, direct document-level links.
4. Aggregation quality is visible in the triple-overlap count: how many documents support all three tiers simultaneously.

## Current structural gap

For both 1000-doc runs, there were no explicit directed `macro -> meso -> micro` edge chains among canonical nodes.

- Directed macro->meso->micro paths: `0` (OFF), `0` (ON)

So today the hierarchy is real in content and document overlap, but not consistently encoded as explicit directed edges.
