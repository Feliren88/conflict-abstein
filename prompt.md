# Design: three conflict datasets for `multilingual-vlm-conflict`

## Target schema (exact match to existing `rpg-conflict`)
`image` (Image), `original_caption`, `conflicting_caption`, `question`, `image_bias`, `text_bias`, `distractor`, `serial_no` (int64), `conflict type` (literal space, as in the org), `language` (= `"english"`). 100-row `train` split each.

> Note: the live org schema uses `original_caption`, `serial_no`, and `conflict type` (with a space) — the request said "caption / serial_number / conflict_type", but we match the **existing org columns exactly** so the new sets are consistent with rpg/code/physics/3D.

## New repos (matching org naming)
- `multilingual-vlm-conflict/sea-vl-conflict`
- `multilingual-vlm-conflict/coco-counterfactual-conflict`
- `multilingual-vlm-conflict/worldcuisines-conflict`

## Sampling
Deterministic: `seed=42`, shuffle, take 100. `serial_no` = source identifier "as is" where integer (`index` for SEA-VL, `qa_id` for WorldCuisines); COCO's `id` is `"158956_0"` (non-int) so use sequential 1–100 there.

## Decisions (locked)
- 100 rows each.
- Conflict fields authored by Claude (semantic, hand-crafted) for SEA-VL; derived programmatically + curated for COCO/WorldCuisines.
- WorldCuisines: English prompts only (`lang == 'en'`).

## Per-dataset conflict construction (with a real example each)

A conflict = take the true caption, change **one** object/attribute → `conflicting_caption`; `image_bias` = true value (in image), `text_bias` = altered value (in text), `distractor` = a plausible third value, `question` asks about exactly that changed thing.

### 1. SEA-VL — free-form caption; hand-author the flip of one object/attribute
```
original_caption : "An ornate door entrance in the Pinang Peranakan Mansion"
conflicting_caption: "An ornate gate entrance in the Pinang Peranakan Mansion"
question         : "What architectural feature forms the entrance?"
image_bias       : "door"      text_bias: "gate"     distractor: "archway"
conflict type    : "object"    (vocab: object | attribute | location | count | action | role)
```

### 2. COCO-Counterfactuals — minimal pairs already encode the conflict
Diff `caption_0`/`caption_1` to extract the swapped noun automatically; author question + distractor.
```
image            : image_0 (the true image)
original_caption : "A woman is cutting into a cake with a large knife."
conflicting_caption: "A baker is cutting into a cake with a large knife."
question         : "Who is cutting into the cake?"
image_bias       : "woman"     text_bias: "baker"    distractor: "chef"
conflict type    : "person-role"  (derived from swapped-noun category)
```

### 3. WorldCuisines VQA (English prompts only) — multi-choice gives ready-made distractors
```
image            : from image_url
original_caption : "This dish is Kenchin-jiru."
conflicting_caption: "This dish is Sayur asem."
question         : <the row's English open_ended_prompt, e.g. "What is the common name of this Japanese dish?">
image_bias       : "Kenchin-jiru" (true answer)  text_bias: "Sayur asem" (a wrong choice)
distractor       : "Draw soup" (another wrong choice)
conflict type    : "food-identity"
```

## Pipeline & build
One Python pipeline per source: download → resample 100 → fetch images (SEA-VL PIL from the dataset; COCO via `load_dataset(..., trust_remote_code=True)` image_0; WorldCuisines via `image_url` with retry/skip) → assemble rows (SEA-VL merges authored conflict JSON; COCO/WorldCuisines derive programmatically) → `datasets.Dataset` with an `Image()` feature → `push_to_hub` to the org repo. Code + authored SEA-VL conflict file live under the working dir; a README/dataset card per repo.
