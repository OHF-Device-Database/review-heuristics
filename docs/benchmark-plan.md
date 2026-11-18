# Contribution checks stack: benchmark plan

This plan is about choosing models. Testing a change to the configuration (thresholds, prompts, messages) uses the benchmark cases instead: see [Configuration](configuration.md).

Companion documents: [Spec](spec.md), [Configuration](configuration.md), [Model shortlist](model-shortlist.md), [Text contribution guidelines](text-contribution-guidelines.md)

## At a glance

We're building one CPU-only service that runs every check the Device Database contribution bot needs. That lets us compare candidate models on the same data and hardware, then ship the winners.

- **Images:**
  - Freepik's EVA-02 obscenity model, tested against two smaller ViT models.
  - SigLIP 2 for classifying photo types (packshot, packaging, label and so on).
- **Text,** applying the rules in the [text contribution guidelines](text-contribution-guidelines.md):
  - Vale for the mechanical style rules.
  - Detoxify for abuse.
  - Small classifier heads on mmBERT-small for opinion, marketing copy, complaints, vagueness and text in the wrong field.
  - An NLI model (DeBERTa-v3) to catch text that repeats or contradicts the structured data.
  - GlotLID for the English-only rule.
  - Pattern rules plus a GLiNER model for personal data and secrets.

Every candidate is exported to ONNX and judged in three steps:
1. It must match the original model's output.
2. It must meet a quality floor on our own labeled examples.
3. It must fit the CPU and memory budget.

Most models are pretrained. Only the image classifier and the text classifier heads need training labels. The rest of the labeling, roughly 6–7k images and 5–6k text items, exists to prove the checks work and to set their thresholds.

### What we test and the data it needs

Sizes marked * are proposals; the sections below define the sets but don't fix their size yet.

| Area | First candidate | Baselines | Examples needed |
| --- | --- | --- | --- |
| Obscenity (images) | Freepik EVA-02 (four severity levels) | Falconsai ViT, AdamCodd ViT, SigLIP 2 probe | 1,500–2,000 hard negatives (hands, wearables, bathroom devices, baby monitors), ~2,000* typical device photos; no explicit positives are stored |
| Image classification | SigLIP 2 zero-shot, then a linear probe | Fine-tune only if the probe plateaus | ~300* per class × 7 classes ≈ 2,100 images |
| Toxicity | Detoxify `unbiased` | Detoxify `original` | Civil Comments (public), ~1,000* benign device sentences, a few hundred* toxic ones |
| Style rules | Vale with our rules | LanguageTool, as a separate track | Two test sentences per rule, plus a ~500*-unit sample of real text |
| Text classifier heads (opinion, marketing, complaint, vague, anecdote, field) | Heads on mmBERT-small | ModernBERT-base; zero-shot NLI | ~300* positives and ~300* near misses per head ≈ 3,000 units, much of it pre-labeled by source |
| Adds value / contradicts structured data | `nli-deberta-v3-base` | `nli-deberta-v3-xsmall`, `roberta-large-mnli` | At least 200 fact-and-unit pairs per rule ≈ 400 |
| English only | GlotLID v4 | fastText lid.176, lingua-py | English device text (from sources), ~100* units in each of ~8 likely languages |
| Personal data and secrets | gitleaks and detect-secrets rules, plus GLiNER-PII for names | Gravitee BERT-small, spaCy | ~300* positives across types (mostly written by us), ~1,000* negatives heavy with brand names |
| Whole pipeline | `/v1/contributions/check` with all rules | | 1,000–1,500 good contributions, a few hundred* bad ones |
| Regression | The guideline examples | | The ~80 examples already in the guidelines |
| Cost | Every surviving variant, fp32 and int8 | | No labels; needs the target CPU |

Text sets overlap: units from whole contributions double as unit-level data. Source-labeled pools need only a 200-item spot check each.

### Data to make it work, and data to benchmark it

Every scoring model also needs **thresholds**, and choosing them needs labeled examples. Those come from the `dev` split of the benchmark sets, so they need no extra collection.

| Area | To make it work | To benchmark it |
| --- | --- | --- |
| Obscenity | Nothing; pretrained. Thresholds from `dev` | The full set above |
| Image classification | Zero-shot: nothing. Linear probe: 16–256 labels per class (~100–1,800 total) | A separate held-out ~100* per class |
| Toxicity | Nothing; pretrained. Thresholds from `dev` | Benign and toxic sets |
| Style rules (Vale) | Rules written by hand, no examples | Test sentences and the precision sample |
| Text classifier heads | 8–128 positives per head; the learning curve shows how many are enough. Zero-shot NLI works with none, less well | Held-out units per head |
| NLI | Nothing; pretrained. Sentence templates for the structured data are written by hand | ~400 fact-and-unit pairs |
| Language ID | Nothing; pretrained | English and non-English sets |
| Personal data | Patterns written by hand; NER pretrained | Positives and brand-heavy negatives |
| Whole pipeline | Nothing | Good and bad contributions |

Only the image probe and the text heads need training data. So would the optional SigLIP obscenity probe or any later fine-tune. Training data can never double as test data, so wherever a model is trained, the benchmark keeps its own held-out set on top. Of the ~3,000 text-head units above, about half are for training and half for testing.

## How a candidate is judged

Every variant (one model at one precision) passes through three gates in order. Deterministic text checks and Vale rules skip gate 1 and go straight to gate 2. A variant that fails a gate is not measured further.

1. **Parity:** the served ONNX variant produces the same answers as the original PyTorch model.
2. **Quality:** it meets the task's quality floor on our evaluation set, at the operating point the policy needs.
3. **Cost:** it fits the capacity and memory budget on the target CPU. Checks run on submit, so latency is only gated by a loose ceiling.

Among the variants that pass all three, the choice goes to the cheapest one. Ties are broken by licence clarity, then by fewer moving parts (for example, a shared SigLIP backbone over a dedicated model).

**All quality numbers are measured through the `serve` API,** never in a notebook, so the server's decoding, preprocessing and tokenization are what gets tested. **Every export or quantization step re-runs gates 1 and 2,** as the shortlist requires.

## Gate 1: parity

For each variant, the `eval` image runs the original PyTorch model and the served variant on the dev split of the task's dataset, and compares the results.

| Check | fp32 pass | int8 / bf16 |
| --- | --- | --- |
| Max absolute score difference | ≤ 1e-3 | Reported, no fixed limit |
| Top-1 or threshold-decision agreement | ≥ 99.9% | Reported; quality is decided by gate 2 |
| Embedding cosine similarity (embedding models) | ≥ 0.999 | ≥ 0.99 as a sanity floor |

An fp32 parity failure is a bug in export or preprocessing, not a property of the model, and is fixed before anything else is measured. The usual causes are interpolation, alpha handling, normalization constants, and tokenizer truncation or padding.

## Gate 2: quality

### Datasets

No labeled sets exist yet, so building them is the first piece of work. Common rules for every task:

- **Manifests, not media, in git.** Each item has a stable id, source, licence, sha256, labels, labeler and split. Media lives on the access-restricted `data/` volume.
- **Frozen splits.** `dev` is used for prompt iteration, probe fitting and threshold tuning. `test` is touched only for the final run of each round, and is never used to pick thresholds. When a source has several items from the same device or author, they stay in the same split.
- **Two labelers on the hard cases.** Every hard-negative and borderline item gets two independent labels. Agreement (Cohen's κ) is reported, and disagreements are adjudicated before the set is frozen.
- **Versioned.** A dataset version is the hash of its manifest. Every report states it.

**Sample sizes follow from the error rates we need to prove.** To show that legitimate content is wrongly rejected or blocked below a target rate, with 95% confidence (Wilson upper bound):

| Target false-reject rate | Neutral items needed | Errors allowed in that sample |
| --- | --- | --- |
| ≤ 1% | 1,000 | up to 4 |
| ≤ 0.5% | 1,500 | up to 2 |
| ≤ 0.5% | 2,000 | up to 4 |

These counts are for the hard-negative subset on its own, not the whole neutral set. Easy neutrals hide the errors that matter.

### Obscenity

**Data**

- *Neutral, typical:* device photos and packshots as submitters upload them.
- *Neutral, hard negatives:* skin or people next to devices. Examples are hands holding devices, wearables, bathroom devices such as scales and shower controls, bedroom devices such as sleep sensors, baby monitors showing their camera view, and lifestyle marketing shots. Also swimwear, lingerie and cosmetics, as the shortlist lists.
- *Positives:* none are collected or stored. Explicit material isn't kept for evaluation, so the benchmark measures what it can check safely: false rejects and review-queue load on neutral images. Detection of explicit images rests on the model's published results.

**Metrics**

The policy has three outcomes:

| Outcome | Rule | What the threshold trades |
| --- | --- | --- |
| Allow | P(low or higher) < `t_review` | |
| Review | Otherwise, unless rejected | `t_review`: missed medium and high images against neutral images sent to review |
| Reject | P(high) ≥ `t_reject` | `t_reject`: high images reaching reviewers against legitimate images rejected |

Rejection exists to keep extremely crude images away from reviewers, so `t_reject` is deliberately conservative. Anything it isn't sure about goes to review.

- **Gate:** the false-reject rate on neutral images, hard negatives counted on their own, meets its floor.
- **Reported:**
  - **False-reject rate:** for neutral and hard-negative images separately, each with a Wilson interval.
  - **Review-queue rate:** on neutral and on hard negatives. Hard negatives cost reviewer time rather than contributor trust, so this is reported but not gated.
- **Binary models** (Falconsai, AdamCodd) output one score, so they can't tell high from medium; they use it for both thresholds.

### Image classification

**Data:** a few hundred images per class from the taxonomy in the spec, including an explicit `unrelated` class drawn from diverse out-of-distribution images.

**Metrics**

- Macro-F1 and accuracy per class, against the per-class floor.
- **Learning curve:** zero-shot, then a linear probe at 16, 64 and 256 labels per class (5-fold CV on `dev`, final on `test`). This shows whether more labeling or a fine-tune is the better next investment.
- **Calibration:** expected calibration error. Also selective accuracy against coverage, which sets the confidence below which a reviewer sees "unsure".
- **Prompt sets for zero-shot** are versioned like datasets, and the best set is chosen on `dev` only.
- A second resolution (384) is tried only if confusion is concentrated in fine-detail pairs such as packaging vs packshot or label vs internals.

### Toxicity

**Data**

- *Public:* the Civil Comments test set with identity labels. Content is English only for now.
- *Ours:* device-database-shaped text, scored per unit as the pipeline does (sentence, step or bullet). Benign items include technical terms that trip filters, such as "kill switch", "dead zone", "master/slave", "execute" and "blow out", plus product names, harmless swearing in reviews ("this damn thing finally works") and identity terms used neutrally. The guideline's complaint and opinion examples are benign for toxicity: they belong to other rules. Toxic items come from labeled real examples where available.

**Metrics**

- AUROC and AUPRC per head.
- False-block rate on our benign set at each head's threshold, split by category (technical terms, swearing, identity terms).
- **Bias:** subgroup AUC, BPSN and BNSP on Civil Comments identity subgroups, the metrics Detoxify `unbiased` was trained against.
- **Thresholds are per head,** tuned on `dev` for the `abuse_spam` rule.

### Text checks

These checks implement the rules in section 6 of the guidelines (see the spec's [Text checks](spec.md#text-checks)). They are measured at two levels:
- **Per rule,** which is what tunes thresholds.
- **Per contribution,** which is what a contributor experiences.

**Data**

Each unit (a name, sentence, step or bullet) is labeled with the field it belongs in and every rule it breaks, so one unit can carry several labels. Units are also grouped into whole contributions, together with the device's structured data, for the contribution-level metrics. The pools:

| Pool | Source | Labels | Role |
| --- | --- | --- | --- |
| Guideline suite | Every example in sections 4 and C.3 of the guidelines, about 80 | Verdict and reason as written there | Regression suite. Too small and too hand-picked to be a test set |
| Source text | Zigbee2MQTT and Z-Wave JS descriptions, pairing and notes; blakadder page titles; matterdatabase notes and CSA descriptions where reachable; Zigbee2MQTT meta descriptions | Mostly given by where the text comes from (see below) | Bulk positives and negatives per rule and per field |
| Written hard cases | The team, per rule, in the style of the guideline's (*) examples | Double-labeled | Near misses each rule must not fire on, and edge cases it must catch |
| Real contributions | Preview Edition submissions, once they exist | Double-labeled | Takes over as the test set as it grows |

Source-text labels come from where the text comes from:
- blakadder titles are bad names.
- CSA descriptions are marketing copy.
- Zigbee2MQTT pairing notes are getting-going text, and its troubleshooting notes are quirks.
- Zigbee2MQTT meta descriptions are boilerplate.

These labels are noisy. A sample of 200 per pool is double-labeled to measure the noise, and the rate is reported next to every metric that uses the pool. Sources whose operator approval is still pending (blakadder, per guideline C.1) are used only once it is given.

Written hard cases target the false positives the guideline worries about most:
- safety warnings in capitals
- hedged quirks ("usually", "some routers"), which the `vague` head must not fire on
- technical terms ("kill switch", "dead zone")
- brand and product names that look like people ("Philips", "Roomba", "Moes") and documented default addresses, which `pii` must not fire on
- English units heavy with brand names, model numbers and German or Nordic product names ("TRÅDFRI"), which `english` must not fire on
- owner reports with a stated basis, which aren't anecdotes
- hedged text that names a version and a source

**Metrics per rule**

- Precision and recall per unit, through `/v1/contributions/check` with only that rule enabled.
- Every rule is reported at two operating points, since warn or reject is set per rule. Proposed floors:
  - **Warn floor:** precision of at least 80%.
  - **Reject floor:** precision of at least 95%.
- **Evidence breakdown:** for each hit, which components fired. This shows whether, for example, the `opinion` head adds recall over Vale's words-to-watch list or only adds noise.

**Metrics per component**

| Component | Metrics |
| --- | --- |
| Rule heads | AUPRC per head. Learning curve at 8, 16, 32, 64 and 128 positives. mmBERT-small against ModernBERT-base. Zero-shot NLI (`nli-deberta-v3-base`, `zeroshot-v2.0-c`) as the day-one baseline |
| Field classifier | Macro-F1 and confusion matrix. The confusions that matter are summary ↔ quirk and quirk ↔ structured |
| Vale rules | A golden test file per rule (a sentence that must trigger it and one that must not), run in CI. Precision per rule on a sample of real text, marked by two reviewers. A rule below the warn floor is fixed or removed |
| Deterministic checks | Unit tests from the guideline suite. False-positive rate on the good-text pools (Zigbee2MQTT and Z-Wave JS notes, the guideline's Good examples) |
| NLI | Per rule (`adds_value`, `consistency`) on hand-labeled pairs of structured fact and unit, at least 200 per rule. Contradiction recall on implicit and numeric pairs is reported on its own. fp32 against int8 near the thresholds |
| Language ID | False "not English" rate on English device text, the decision metric. Detection rate on non-English units, per language. The share of units passed for too few letters |
| PII | Precision and recall per entity type. False positives on brand and product names, and on documented defaults (default access-point names and addresses), reported separately |
| Unit splitting | Agreement with hand-split units on 200 contributions. Code blocks must never be split |

**Metrics per contribution**

- **False-reject rate on good contributions:** the share of contributions labeled good that end up with `outcome: reject`. This is the decision metric for the whole pipeline. Errors add up across rules. Nineteen rules at 1% false positives each would bounce about one good contribution in six. So the per-rule floors are tightened until this number meets its target, and the sample-size table above applies to it.
- **Catch rate on bad contributions:** the share labeled bad that end up with a hit on at least one of the rules they break.
- **Guideline suite reproduction:** the share of guideline examples whose verdict the pipeline reproduces. Any example that passed before and fails now fails the run.

### Text: LanguageTool (separate track)

LanguageTool is not in the serving stack. This track decides whether it should be.

- **Setup:** a pinned LanguageTool server image under the `languagetool` compose profile, without n-gram data. The harness calls it directly. A second run with the English n-gram data is added only if the first run is close to the bar, since that data is about 15 GB.
- **Data:** the same device-database text sample used for Vale's precision check, with a custom dictionary of device terms (brands, model numbers, protocol names such as Zigbee, Z-Wave and Matter), so domain vocabulary doesn't drown the results.
- **Metrics:**
  - Precision per LanguageTool rule category, marked by the same two reviewers as for Vale.
  - **Added value:** genuine issues LanguageTool finds that Vale doesn't, per 1,000 words.
  - Alerts overlapping a Vale alert on the same span.
  - Cost: p95 latency per text, container memory, image size.
- **Decision:** it is added as a sidecar only if its added value is material and its precision on the categories we'd enable meets the same 90% bar as Vale's rules.

**Decision point for heads:** if frozen-body heads plateau below the floor, that is the evidence needed to bring full SetFit fine-tuning into scope.

## Gate 3: cost

### Environment capture

Every cost run records the following, and the report states it on the first line:

- CPU model, physical core count, and flags (`avx2`, `avx512f`, `avx512_vnni`, `avx512_bf16`, `amx_tile`, `amx_bf16`, `amx_int8`).
- Container CPU set, memory limit and kernel.
- ONNX Runtime version and session options: intra-op and inter-op threads, and the graph optimization level.

Checks run when a contribution is submitted, so the budget is generous. Cost decides between variants that pass gate 2, and catches anything far out of line, but it isn't the main constraint. Numbers from different hosts are never mixed in one comparison. Until the production CPU is known, runs on whatever x86 host is available are labeled provisional, and the gate is signed off only on the target CPU.

### Measurements

| Measurement | How | Reported |
| --- | --- | --- |
| Model cost | In-process ONNX Runtime, no HTTP, batch 1 and 8, after warm-up | p50, p95 |
| Endpoint latency | Open-loop (fixed arrival rate) load from `eval` against `serve`, batch 1, using representative inputs. Images at real upload sizes, since decoding a 12 MP JPEG is part of the cost. Contributions at real lengths | p50, p95, p99, and the server's breakdown into decode, preprocess and inference. Gated only by a loose ceiling, set once the target CPU is known |
| Capacity | Arrival rate stepped up until p95 passes the ceiling | Sustainable contributions per minute and images per minute, per core |
| Isolation | Saturating image load alongside a steady contribution load | Contribution p95 with the image load against without it, reported |
| Memory | RSS before and after loading each variant, plus peak under load | MB per variant, total |
| Startup | Container start to `/readyz` | Seconds |
| Image size | Compressed `serve` image | MB |

Open-loop load is required. A closed-loop client (send, wait, send) slows down with the server and hides queueing delay (coordinated omission). The load generator lives in the `eval` image, so runs are reproducible from one command.

### Precision variants

Each model that passes gate 2 in fp32 is re-run through gates 1–3 as:

- int8 dynamic (text models)
- int8 dynamic and int8 static with a calibration subset of `dev` (vision models)
- bf16, only where the CPU flags support it

A quantized variant is accepted only if its gate 2 results stay inside the fp32 result's confidence interval on the decision metrics. The point estimate isn't enough.

## Reports

Each run writes `reports/<date>-<task>-<run-id>/` containing:

- `results.json`: raw per-item scores, so thresholds can be re-derived without re-running.
- `metrics.json`: everything above, with confidence intervals.
- `report.md`: a summary table per gate, the environment, the dataset version, model hashes and the harness git commit.

A per-task scoreboard is generated from all runs, with one row per variant: its gate status, its decision metrics and its cost.
