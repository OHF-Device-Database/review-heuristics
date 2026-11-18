# Contribution checks stack: spec

Companion documents: [Configuration](configuration.md), [Model shortlist](model-shortlist.md), [Benchmark plan](benchmark-plan.md), [Text contribution guidelines](text-contribution-guidelines.md)

## Purpose

The stack runs the checks the device database's contribution review bot applies before a human sees a contribution:

- **Images:** obscenity and image classification.
- **Text:** the writing rules in section 6 of the [text contribution guidelines](text-contribution-guidelines.md), applied to the four text fields (name, summary, getting going, quirks).

It serves every shortlisted candidate behind one REST API, so they can be compared on the same data, the same preprocessing and the same hardware. It also serves the assembled pipeline the bot calls. Which models run, and how their scores become decisions, is configuration in [`config/`](../config), not code: see [Configuration](configuration.md).

It is two images:

| Image | Contains | Job |
| --- | --- | --- |
| `serve` | ONNX Runtime, FastAPI, Hugging Face `tokenizers`, Pillow, NumPy, the Vale binary, a fastText runtime, gitleaks rules and detect-secrets, the deterministic checks | Loads ONNX models from a registry, runs the checks, and serves the API. No PyTorch. |
| `eval` | PyTorch (CPU), Transformers, Optimum, timm, scikit-learn, the benchmark harness | Exports and quantizes models, checks parity, runs the quality and cost benchmarks against `serve`, and writes reports. |

Training of model weights is out of scope. The one exception is that the `eval` image fits logistic-regression heads on frozen embeddings: the image linear probe and the text classifier heads. Measuring those candidates requires it, it takes seconds on CPU, and each head is a small artifact that `serve` loads like any other model. Full SetFit (contrastive fine-tuning of the body) counts as training, and is only considered if frozen-body heads fall short.

## Non-goals

- GPU serving.
- Authentication, rate limiting per client, multi-tenant isolation. The service runs behind the submission service, not on the open internet.
- Rewrites and suggestions. The guideline defers "Suggest" to a later step, so there is no generative model in the stack.
- Verifying device facts. The guideline rules this out for the bot, because the bot can't own the device. Checks compare text against structured data and ingested sources, never against the world.

## Repository layout

```
config/                     # what runs and how scores become decisions; see configuration.md
  models.json               # every model variant the server can load
  checks.json               # rules: model, fields, thresholds, warn or reject, message
  image-classes.json        # the image taxonomy as zero-shot prompts
  language.json             # words masked before language identification
  facts.json                # structured data rendered as sentences for NLI
  schemas/                  # a JSON Schema for each file
benchmarks/
  cases/                    # known-good and known-bad contributions, with expected outcomes
  scores/                   # cached model scores, so config changes compare without a server
checks/ddb_checks/          # the rules engine, config loading and the benchmark (no ML libraries)
serve/                      # the model server: ONNX Runtime, FastAPI, the contribution check
eval/                       # export, parity, cost and quality harness (PyTorch)
  datasets/                 # manifests and the images this repo may publish
deploy/                     # an example Portainer stack
compose.yaml, Makefile      # make build / export / serve / parity / validate / compare / scores
pyproject.toml, uv.lock     # one uv workspace for checks, serve and eval
```

Model weights, reports and offensive or explicit test data are not committed. Weights and reports live in Docker volumes: `models` (written by `eval`, read-only in `serve`) and `reports`. The configuration is built into both images. Neither image needs a host directory shared with the Docker VM; reports are copied out with `docker cp`.

## Model registry

`config/models.json` lists every variant the server can load. A variant is one model at one precision. Its id, `<task>/<name>@<precision>`, also fixes where its files live: `<models>/<task>/<name>/<precision>/`.

```json
{
  "id": "toxicity/detoxify-unbiased@int8",
  "source": "unitary/unbiased-toxic-roberta@36295dd80b422dc49f40052021430dae76241adc",
  "license": "Apache-2.0",
  "labels": ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"],
  "runtime": { "intra_op_threads": 4, "max_concurrency": 1, "max_queue": 32, "batch_size": 32 }
}
```

`eval export` writes the variant's files (`model.onnx`, or `vision.onnx` and `text.onnx` for SigLIP 2, or a fastText `.ftz`) next to a `meta.json`. Where a repo publishes its own ONNX or model file, the registry entry names it as `artifact` and export copies it instead of exporting. The metadata holds everything taken from the source model rather than chosen by us:
- the full label list and the output activation, from the model config
- the token limit and padding token, for text models
- preprocessing (size, interpolation, crop, normalization), for image models, from `preprocessor_config.json` or the timm data config
- each file's SHA-256 (`files`), and one `sha256` for the variant as a whole: the file's own hash for a single-file variant, otherwise a hash over the sorted file list
- the tool versions used

Text models also get `tokenizer.json`. An int8 variant is quantized from the fp32 export of the same model.

- **Source revision is pinned** to a Hugging Face commit, never a branch. When a repo has only `.bin` weights, Transformers starts a background request asking the Hub to convert it to safetensors. The harness disables that (`DISABLE_SAFETENSORS_CONVERSION`), so nothing is fetched from, or triggered on, a ref other than the pinned one.
- **Every file's SHA-256 is checked at load** against `meta.json`. Every response carries the variant id and hash, so a report row can always be traced to the exact weights.
- **`labels` in the registry** picks which of the model's outputs are exposed. Detoxify `unbiased` also has identity-subgroup heads, which were training auxiliaries rather than toxicity signals.
- **`SERVE_MODELS`** (a comma-separated list or glob of ids) selects which variants load. Quality runs can load many. Cost runs load only the variant under test, so other sessions don't compete for cores.

## Initial variant matrix

| Task | Variant ids (fp32 and int8 each, unless noted) | Export path |
| --- | --- | --- |
| Obscenity | `freepik-eva02`, `falconsai-vit`, `adamcodd-vit` | timm / Transformers → `torch.onnx.export`; AdamCodd's published ONNX checked against our own export |
| Obscenity (shared backbone) | `siglip2-probe` | SigLIP 2 image embedding + logistic-regression head |
| Image embeddings | `siglip2-base-256` (224 and 384 later only if fine detail matters) | Vision and text towers exported separately |
| Toxicity | `detoxify-unbiased`, `detoxify-original` | Underlying HF checkpoints (`unitary/unbiased-toxic-roberta`, `unitary/toxic-bert`) via Optimum |
| Text embeddings | `mmbert-small`, `modernbert-base` | Optimum, mean pooling in the server |
| Text classifier heads | One set of heads per text-embedding variant (see [Text checks](#text-checks)) | Logistic regression fitted in `eval` |
| NLI | `nli-deberta-v3-base` (official fp32; int8 fails parity, see the shortlist), `nli-deberta-v3-xsmall`, `zeroshot-v2.0-c` (zero-shot only), `roberta-large-mnli` (fp32 accuracy ceiling) | Official ONNX from the `cross-encoder` repos; Optimum for the others |
| Language ID | `glotlid-v4` (the official quantised `model_v4.ftz`), `fasttext-lid176`, `lingua`, `py3langid` | No export. fastText models run on the fastText runtime; lingua and py3langid are Python packages |
| Personal names (NER) | `gliner-pii-base`, `gravitee-bert-small-pii`, `spacy-en-core-web-sm` | Official ONNX for both GLiNER and Gravitee; spaCy runs natively |
| Style rules | `vale` | No export. LanguageTool is benchmarked separately, outside the API |

Content is English only for now, so Detoxify `multilingual` and `textdetox` are deferred until other languages are needed. mmBERT-small stays in, because choosing it now avoids a backbone swap later.

Checks run when a contribution is submitted, not while the contributor types, so latency isn't the constraint (see the benchmark plan). Larger variants, such as ModernBERT-large or SigLIP 2 at 384, aren't ruled out by speed. They are added only if a base variant misses its quality floor.

Vision int8 is tried as both dynamic and static (calibrated) quantization, because dynamic quantization often costs vision transformers more accuracy. bf16 is added only on hosts whose CPU flags support it (see the benchmark plan).

## Text checks

### Units

Text is checked per unit, so every hit points at something the contributor can fix:

- **Name:** the whole field is one unit.
- **Summary:** each sentence is a unit.
- **Getting going:** each step is a unit.
- **Quirks:** each bullet is a unit.

Splitting text into units is part of the pipeline, and it is tested like any other component. Code blocks are kept whole, never split into sentences.

### Classifier heads

These are logistic-regression heads on frozen sentence embeddings. They come in two groups.

**Rule heads.** Each one is a binary head, and they follow the guideline's recurring "Bad" verdicts:

| Head | Fires on | Guideline examples |
| --- | --- | --- |
| `opinion` | Judgment, recommendation, ranking | "Great little relay … highly recommended", "Terrible battery life, avoid" |
| `marketing` | Vendor marketing copy | The Roomba Combo j9+ vendor paragraph |
| `complaint` | Complaints about vendor, support or software | "support was useless, iRobot should fix their API" |
| `vague` | Claims with no condition, version or detail | "Sometimes slow", "follow the instructions" |
| `anecdote` | Personal story without a reproducible step | "Took me three evenings …" |

**Field classifier.** This is one multi-class head that says which field a unit belongs in: `name`, `summary`, `getting_going`, `quirk`, `structured` (it restates or belongs in a structured field), or `none`. A unit classified into another field gets a "wrong field" hit, as in the guideline's "correct facts, wrong place" verdicts.

### Rules

Every rule from section 6 of the guideline is in scope, plus two that the guideline names elsewhere (complaints in 5.4, boilerplate in C.4). Each rule combines one or more components:

| Rule id | Guideline rule | Components |
| --- | --- | --- |
| `name.form` | Name adds nothing beyond the manufacturer's own name | Deterministic: trademark symbols, plus terms matching the category, protocol and brand vocabularies, checked against this device's own brand and category. `opinion` head for non-names ("Best Zigbee plug 2026!!") |
| `name.model_number` | Name carries a model number when the device has one | Deterministic: compared with the identifiers field |
| `name.duplicate` | Duplication check (4.1) | Identifier match, then fuzzy string match, then text-embedding similarity against existing names |
| `adds_value` | Text adds something beyond structured data | NLI: the device's structured facts entail the unit. The field classifier's `structured` label is the baseline |
| `factual` | Factual, no opinion or recommendation | Vale words-to-watch lists, plus the `opinion` head |
| `no_marketing` | No vendor marketing copy | `marketing` head, plus near-duplicate match against known vendor text from ingested sources |
| `no_complaints` | Complaints never go in (5.4) | `complaint` head |
| `specific` | Specific, not vague | Vale vague-terms list, plus the `vague` head |
| `uncertainty` | Uncertainty marked | Vale: claims phrased as questions. `anecdote` head. Unmarked uncertainty can't be detected without verifying facts |
| `length_form` | Length and form per field | Deterministic: length and form per guideline section 3. Field classifier for units in the wrong field |
| `generic_steps` | No duplicate of generic Home Assistant steps | Text-embedding similarity to the default getting-going text, plus Vale phrases |
| `code` | Code and configuration blocks | Deterministic: fenced blocks, YAML and JSON that parse, MQTT topics and payloads |
| `pii` | No personal data or secrets | Mask the device's own data (brand, identifiers, documented defaults) first. Then patterns and checksums: email, gitleaks and detect-secrets rules, Wi-Fi credentials, IPs with an ownership cue, serials with a cue, MAC, IMEI, EUI-64, pairing codes. NER for personal names |
| `commerce` | No prices or shop links | Deterministic: shop and affiliate URL lists, affiliate parameters, currency patterns |
| `links` | Links only to documentation and sources | Deterministic: domain allowlist |
| `english` | English only | Script check, then language ID on masked text (no digits, brand or multi-capital tokens). Units with too few remaining letters pass. Skipped for the name field |
| `abuse_spam` | Not abusive, not spam | Detoxify per unit, plus spam heuristics (link density, repetition) |
| `consistency` | Consistent with ingested sources and structured data | NLI contradiction against each structured fact, and against ingested text for the same device |
| `boilerplate` | Boilerplate from sources (C.4) | Deterministic: near-duplicate match against sentences that repeat across many entries of one source |

Structured facts are rendered to short English sentences before NLI ("This device works locally, without an internet connection."). The templates live in `config/facts.json`.

**Safety sentences are exempt from style rules.** A unit that warns about mains voltage or flashing hazards keeps its capitals and exclamation marks (guideline C.2). It is detected by a Vale rule, and `factual` and `specific` don't fire on it.

### Configuration

Each rule's configuration lives in `config/checks.json`: its type, the model it uses, the fields it applies to, the score at which it warns or rejects, the message shown to the contributor, and the guideline section it links to. Whether a rule warns or rejects is set per rule. The rules engine (`checks/ddb_checks`) is shared by the server and the benchmark, so the benchmark measures exactly what production runs. [Configuration](configuration.md) explains the files and how to test a change.

Rule types available now: `toxicity`, `language`, `nli_contradiction`, `nli_entailment`, `image_class` and `obscenity`. The other rules in the table above need components that aren't built yet: Vale, the classifier heads and the deterministic checks.

## API

FastAPI generates the OpenAPI 3.1 schema, served at `/openapi.json` with docs at `/docs`. The schema file is also committed and checked in CI, so a contract change shows up in review.

### Conventions

- **All component routes take a batch.** Each item has a caller-supplied `id` that is echoed back. Maximum batch size is per route (default 32 images, 128 texts).
- **`model` selects the variant** by registry id. If the task has exactly one loaded variant it can be omitted.
- **Every response carries** `model`, `model_sha256` and, for the whole request, `timings_ms` (`decode` for images, `preprocess`, `inference`, `total`), so the benchmark can split model cost from server overhead.
- **Images are sent as bytes,** as base64 in JSON (multipart parts are planned). The server never fetches URLs. Input limits are 20 MB per image and 40 megapixels after decode, to guard against decompression bombs.
- **Errors** use RFC 9457 problem details. Status codes:
  - `413` over limits;
  - `422` for invalid or undecodable input, or a missing `model` when several variants serve the task;
  - `404` for an unknown model;
  - `503` with `Retry-After` when that model's queue is full.

### Routes

| Method and path | Purpose | Returns per item |
| --- | --- | --- |
| `GET /healthz` | Process is up | |
| `GET /readyz` | All selected variants are loaded and warmed up | |
| `GET /v1/models` | Loaded variants with registry metadata | |
| `GET /playground` | A page for trying every endpoint by hand. Development builds only; production images leave it out | |
| `POST /v1/contributions/check` | The bot's pipeline: every text rule over one contribution | Outcome, plus hits with rule, field, span, action, message and evidence |
| `POST /v1/obscenity` | Score images | `scores` per model label; `at_least` per level (cumulative, for example `at_least.medium` = P(medium) + P(high)) for graded models; `nsfw` = 1 − P(neutral or normal), comparable across all models |
| `POST /v1/images/embeddings` | SigLIP 2 image embeddings | L2-normalised vector |
| `POST /v1/images/classify` | Zero-shot (prompts in the request) or a named probe head | Probability per class |
| `POST /v1/texts/embeddings` | SigLIP 2 text or mmBERT/ModernBERT embeddings | L2-normalised vector |
| `POST /v1/texts/classify` | Rule heads and the field classifier | Probability per head and per field |
| `POST /v1/toxicity` | Score texts | Score per head (toxicity, severe toxicity, obscene, threat, insult, identity attack, sexual explicit for `unbiased`) |
| `POST /v1/style/rules` | Vale | Alerts: `rule`, `severity`, `message`, `span` (character offsets), `suggestions` |
| `POST /v1/texts/nli` | Premise and hypothesis pairs, or zero-shot labels with a hypothesis template | Entailment, neutral and contradiction scores (entailment only for two-way models) |
| `POST /v1/texts/language` | Language ID, with masking applied | Top languages with probabilities, P(English), letters scored after masking |
| `POST /v1/texts/pii` | Patterns and NER, with an allowlist from the request | Entities: `type`, `span`, `score`, `detector` |

### Contribution check

The request carries the device's structured data as context, the text fields as submitted, and any images. `config_version` is optional; when given, the server refuses with `409` unless it runs that configuration version.

```http
POST /v1/contributions/check
Content-Type: application/json

{"config_version": "0.1.0",
 "device": {"brand": "Google", "category": "smart display", "connectivity": "cloud",
            "protocols": ["Wi-Fi", "Thread"]},
 "fields": {"name": "Nest Hub (2nd gen)",
            "summary": "Smart display with a touchscreen that connects over Wi-Fi.",
            "getting_going": null,
            "quirks": ["The Nest Hub acts as a Zigbee coordinator for your lights."]},
 "images": [{"id": "front", "data": "<base64>"}]}
```

```json
{"outcome": "warn",
 "config": {"version": "0.1.0", "sha256": "…"},
 "rules": ["abuse_spam", "english", "consistency", "adds_value", "image_category", "obscenity"],
 "hits": [{"rule": "consistency", "field": "quirks", "unit": 0, "item": 0,
           "text": "The Nest Hub acts as a Zigbee coordinator for your lights.", "span": [0, 58],
           "action": "warn",
           "message": "This seems to contradict the device's structured data. …",
           "guideline": "docs/text-contribution-guidelines.md#rules",
           "evidence": [{"model": "nli/nli-deberta-v3-base@fp32", "signal": "contradiction",
                         "score": 0.98, "threshold": 0.9,
                         "detail": "This device communicates over Wi-Fi."}]}],
 "timings_ms": {"total": 412.0}}
```

- **`outcome`** is `reject` if any hit's action is `reject`, `warn` if there are only warnings, and `accept` if there are no hits.
- **`evidence`** lists every score that crossed a threshold, with the fact or language behind it. The guideline treats a false positive as a bug in the rule, and fixing one needs to know which component produced it.

### Zero-shot image classification

Zero-shot classification takes the class prompts in the request, so prompt sets can be iterated without a redeploy. The server caches text embeddings by prompt hash.

```json
{"model": "image-embed/siglip2-base-256@fp32",
 "classes": {"packshot": ["a product photo on a plain background"],
             "in_situ": ["a smart home device installed in a room"],
             "unrelated": ["a photo unrelated to electronics"]},
 "images": [{"id": "…", "data": "…"}]}
```

## Serving internals

- **One process, one ONNX Runtime session per variant.** Each session has its own intra-op thread pool sized by `intra_op_threads`.
- **Memory arena per variant.** ONNX Runtime keeps the memory a run allocated for reuse, so a variant's footprint grows to its largest batch and stays there. The registry's `runtime.memory_arena: false` turns that off; the image models run that way. Measured on Nov 5, 2026: obscenity requests of 1, 8 and 16 photos grew the server by 2.8 GB with the arena and 0.6 GB without, for 2–5% more inference time.
- **Per-model bounded executor.** Each variant gets a thread pool of `max_concurrency` workers and a queue of `max_queue`. ONNX Runtime releases the GIL during `run`, so variants execute in parallel, and an image burst fills the image queue without delaying text requests. The thread budget is checked at startup, and a warning is logged if the sum of `max_concurrency × intra_op_threads` exceeds the container's CPU allowance.
- **The contribution pipeline embeds each unit once.** It runs the independent components for a contribution concurrently and merges their hits.
- **Server-side batching:** off at first. Requests are batched only within what the client sends. Dynamic micro-batching is added only if the throughput benchmark shows it is worth it.
- **Image preprocessing must match the reference exactly.** That covers EXIF orientation, mode conversion (transparent PNGs, common for packshots, are composited onto white), interpolation, crop and normalization. The parity check in the benchmark plan is what enforces this.
- **Tokenization** uses each model's `tokenizer.json` through `tokenizers`, with the model's own max length and truncation. SigLIP 2 text needs lowercase input padded to `max_length=64`, which is how it was trained, otherwise zero-shot scores degrade.
- **Vale** runs as a subprocess with JSON output, one invocation per request batch.
- **Warm-up:** each session runs one dummy batch before `/readyz` reports ready, so the first measured request is not a cold one.
- **Logging and metrics:** structured JSON logs, and a Prometheus `/metrics` endpoint with per-variant latency histograms, queue depth, rejections and hits per rule.

## Images

- **`serve`:** `python:3.12-slim` base. The Vale release binary is copied in with its checksum verified. All dependencies are pinned with hashes (`uv` lockfile). The image runs as a non-root user with a read-only root filesystem. Models are mounted read-only at `/models` during evaluation and baked into a versioned image for production.
- **`eval`:** CPU-only PyTorch wheels and the same Python version. All three packages share one `uv` workspace lockfile.
  - **Built:** `export` (including int8 quantization); `parity`, which checks a served variant against its original for toxicity, NLI, SigLIP 2 and obscenity; `bench`, which runs parity and cost against a served toxicity variant; and `quality` for toxicity, which scores Civil Comments and the device-text set through the served variant. Each writes a report.
  - **Planned:** `probe`, which fits heads, and `quality` for the other tasks.
- **Compose:** `serve` and `eval`. The CPU set for `serve` is pinned (`cpuset`), so cost runs are repeatable and match the target core budget.

## Decisions

- **Content language:** English only for now. Multilingual toxicity models are deferred.
- **Image taxonomy:** packshot, in situ, packaging, label or nameplate, internals or PCB, screenshot or app UI, unrelated.
- **Obscenity:** images with any sign of sexual content go to a reviewer; images very likely to be explicit are rejected, so reviewers don't see them. No explicit images are stored for evaluation.
- **Style guide:** the [text contribution guidelines](text-contribution-guidelines.md). Their section 6 rules define the text checks, and their section 4 examples seed the benchmark.
- **LanguageTool** is not part of the serving stack.
- **Latency:** checks run on submit, so cost is judged on capacity and memory; latency is only gated loosely.
- **Warn or reject** is set per rule in `config/checks.json`. A reviewer sees every contribution with warnings.

**Licences to settle before a model ships:** the Falconsai MIT/Apache-2.0 conflict; share-alike training data (SNLI, FEVER) behind the NLI models; GlotLID's "Apache-2.0 plus notices" terms; CC BY-SA 3.0 on fastText lid.176 weights; Knowledgator's undisclosed GLiNER training data.
