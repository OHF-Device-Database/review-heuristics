# Model shortlist

Oct 15, 2026

## Summary

Every candidate below is an encoder of roughly 90–310M parameters that can be exported to ONNX and run on x86 CPU cores. Accuracy figures come from the model authors' own benchmarks and need confirming on our evaluation sets before any choice is final.

| Task | First candidate | Baselines to compare | Why this one first |
| --- | --- | --- | --- |
| Obscenity | Freepik/nsfw\_image\_detector | Falconsai/nsfw\_image\_detection, AdamCodd/vit-base-nsfw-detector | Four severity levels and much higher reported accuracy on borderline ("low"/"medium") content |
| Image classification | SigLIP 2 base, zero-shot, then a linear probe | A fine-tuned head only if the probe plateaus | Works with zero labels today; a linear probe trains in minutes on CPU |
| Toxicity | Detoxify `unbiased` (English) | Detoxify `original`, `multilingual` | Trained specifically to reduce identity-term bias, which matters for automatic blocking |
| Style, mechanical rules | Vale | LanguageTool for grammar | Deterministic, explainable, single Go binary |
| Style, judgment-based aspects | mmBERT-small (SetFit-style classifier, then span tagging) | ModernBERT-base (English only) | Multilingual from the start, so no backbone swap later |
| Text against structured data (NLI) | cross-encoder/nli-deberta-v3-base | MoritzLaurer deberta-v3-base-zeroshot-v2.0-c (zero-shot only), nli-deberta-v3-xsmall, roberta-large-mnli | The only strong three-way NLI model trained on licence-clean data; official int8 ONNX |
| Language ID | GlotLID v4, quantised | fastText lid.176, lingua-py | Widest margin between English and non-English on a probe of device text; permissive weights |
| Personal data: names | knowledgator/gliner-pii-base-v1.0 | gravitee-io/bert-small-pii-detection, spaCy en_core_web_sm | Labels are chosen per request, so brand and product labels can compete with "person"; ONNX in the repo |
| Personal data: secrets and identifiers | gitleaks rules plus detect-secrets | | Maintained, permissively licensed rule sets; deterministic |

One design choice is worth testing early: SigLIP 2 image embeddings could feed both the product classifier and a second obscenity classifier. One image pass would then serve two heads. It only replaces the dedicated obscenity model if it matches that model's false-block rate.

The NLI model can do double duty too. Besides checking text against structured data, it classifies zero-shot ("This text is a recommendation."), which gives the text classifier heads a baseline before any labels exist.

Licences were checked against the model cards, repositories and dataset cards in November 2026. GlotLID v4 is used rather than v3: it ships an official quantised file and, per its README, is trained on short spans and calibrated, which suits device text better.

## Obscenity

Freepik's model reports the best accuracy on borderline content, which is where product imagery causes trouble. It is also the heaviest to run on CPU, so the lighter ViT models stay in the comparison.

| Model | Base and input size | Output | License | Reported results |
| --- | --- | --- | --- | --- |
| [Freepik/nsfw\_image\_detector](https://huggingface.co/Freepik/nsfw_image_detector) | EVA-02 base, 448 px | neutral / low / medium / high | MIT | Their benchmark: 99.5% high, 97.0% medium, 98.3% low, 99.9% neutral |
| [Falconsai/nsfw\_image\_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | ViT-B/16, 224 px | normal / nsfw | Card lists both MIT and Apache-2.0 | 98.0% eval accuracy; 78.5% medium and 31.3% low in Freepik's benchmark |
| [AdamCodd/vit-base-nsfw-detector](https://www.mixpeek.com/model/AdamCodd/vit-base-nsfw-detector) | ViT-B/16, 384 px | normal / nsfw | Apache-2.0 | 91.7% medium and 89.7% low in Freepik's benchmark; ONNX weights published |
| [Image-Guard-2.0](https://huggingface.co/blog/prithivMLmods/image-guard-models) | SigLIP 2 base, 224 px | Several SFW and explicit classes | Not stated | No figures in text |

- **Freepik's comparison is their own benchmark,** on their own manually labeled set. It shows the direction, but our hard negatives (swimwear, lingerie as a product, cosmetics on skin) decide the result.
- **CPU cost:** at 448 px, EVA-02 processes about five times as many image patches as a 224 px ViT (1,024 vs 196). Latency on our target cores needs measuring before it is accepted. Freepik's card only reports GPU timings and recommends bf16, which is fast only on x86 CPUs with bf16 support.
- **Falconsai license:** the card metadata says MIT while the page header says Apache-2.0. Both are permissive, but the conflict should be resolved before shipping.
- **Training data is undisclosed or synthetic** for all of these. Fine-tuning on our own hard negatives is the likely path to the target false-block rate, whichever base wins.

## Image classification

Start with [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256) base (Apache-2.0, about 0.4B parameters for image and text encoders together). It gives a zero-shot result today and frozen embeddings for a linear probe once labels exist.

1. **Zero-shot (no labels):** text prompts per class ("a packshot of a product on a plain background", "a person using a product" and so on) scored against the image embedding. This gives a baseline on day one and a way to pre-fill labels for reviewers.
2. **Linear probe (a few hundred labels):** logistic regression on frozen image embeddings. It trains in minutes on CPU, and only the image encoder runs at inference.
3. **Fine-tuning:** only if the probe plateaus below the per-class accuracy floor.

- **Size options:** SigLIP 2 comes in several resolutions (224, 256, 384, 512) and larger variants. Base at 224 or 256 is the cheapest to start with; higher resolution only if fine detail decides classes, such as packaging vs packshot.
- **"Unrelated" class:** keep it as an explicit probe class trained on diverse out-of-distribution images, plus a low-confidence fallback. The probe's calibrated probability decides when a reviewer sees "unsure".
- **Shared backbone:** these embeddings could also train an obscenity probe. One image pass would serve both tasks, as long as it meets the obscenity false-block rate on its own.

## Toxicity

Use Detoxify `unbiased` for English now. Choose the multilingual model when the language list is known, because the candidates cover different languages.

| Model | Base | Languages | License | Notes |
| --- | --- | --- | --- | --- |
| [Detoxify `unbiased`](https://pypi.org/project/detoxify) | roberta-base | English | Apache-2.0 | Trained on Civil Comments with identity labels to reduce bias; 7 output heads including identity attack |
| Detoxify `original` | bert-base-uncased | English | Apache-2.0 | Wikipedia comments; known to flag identity terms and harmless swearing |
| Detoxify `multilingual` | xlm-roberta-base | en, fr, es, it, pt, tr, ru | Apache-2.0 | Per-language AUC 89–97%; the authors say to use it only on these 7 languages |
| [textdetox/xlmr-large-toxicity-classifier](https://huggingface.co/textdetox/xlmr-large-toxicity-classifier) | xlm-roberta-large | en, ru, uk, de, es, ar, am, hi, zh | OpenRAIL++ | F1 0.87 overall, 0.97 English, 0.73 Chinese; binary output; a v2 exists |

- **Detoxify's own warning:** text with swearing or insults tends to score as toxic regardless of intent, including humorous or self-deprecating text. That is the main source of false blocks to test for.
- **Per-head thresholds:** `unbiased` returns toxicity, severe toxicity, obscene, threat, insult, identity attack and sexual explicit scores. Each head is a separate blocking decision with its own threshold.
- **CPU cost of the large model:** xlm-roberta-large is about three times the compute of a base model. It is acceptable for short text but needs a latency check.
- **OpenRAIL++** carries use-based restrictions, unlike Apache-2.0 or MIT. It needs a license review before shipping.
- **Fallback path:** if neither multilingual model covers the needed languages, fine-tune mmBERT-small (see Style enforcement) on the [textdetox multilingual dataset](https://huggingface.co/textdetox/xlmr-large-toxicity-classifier) plus our own labeled data.

## Style enforcement

Vale handles the mechanical rules. A small encoder model handles tone, formality, directness, transitions and POV/tense consistency.

### Rules

| Tool | What it covers | License | Runtime |
| --- | --- | --- | --- |
| [Vale](https://cdn.jsdelivr.net/gh/vale-cli/vale@v3/README.md) | Our own YAML rules: term lists, regex, part-of-speech patterns, readability formulas, scripts; rules can carry fixes | MIT | Single Go binary, no runtime |
| [LanguageTool](https://www.credativ.de/en/blog/credativ-inside/languagetool-powerful-language-checking-on-your-own-network/) | General grammar and spelling, many languages | LGPL-2.1 (core) | Java server; optional n-gram data is about 15 GB for English |

- Vale covers most of punctuation and mechanics, contractions, capitalization, hyphenation, and sentence and paragraph length, each as a named rule with an exact span.
- LanguageTool is optional. It adds general grammar checks at the cost of a Java runtime in the image. Its n-gram data is optional, and leaving it out keeps the image much smaller.

### Model

| Backbone | Size | Languages | License | Role |
| --- | --- | --- | --- | --- |
| [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | 140M (42M outside the embeddings) | 1,800+ | MIT | Recommended: classifier now, span tagger later, multilingual without a swap |
| [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base/blob/refs%2Fpr%2F85/README.md) | 149M | English | Apache-2.0 | English-only alternative if mmBERT underperforms on English |

- **Start with a SetFit-style classifier** per judgment aspect (tone, formality, directness). [SetFit](https://huggingface.co/blog/setfit) reports results competitive with full fine-tuning from 8–16 labeled examples per class, and it trains on CPU in minutes. That fits starting from zero examples.
- **Move to span tagging** (token classification) once reviewers have produced span-level labels, so flags point at the exact words.
- **mmBERT's compute is small:** most of its 140M parameters are its 256k-token vocabulary embeddings, which are lookups. The transformer layers are 42M parameters.
- **One backbone for both text tasks** is possible: mmBERT-small can also host the multilingual toxicity fallback, which keeps one tokenizer and runtime path in the image.

## Text against structured data (NLI)

An NLI model reads a premise and a claim, and says whether the premise implies the claim, contradicts it, or neither. The device's structured data is rendered as short sentences and used as the premise. Each contributed sentence or bullet is the claim. If the structured data implies the sentence, the sentence adds nothing (`adds_value`). If it contradicts the sentence, the sentence fails `consistency`.

| Model | Base and size | Labels | Licence | Training data | Reported results | ONNX |
| --- | --- | --- | --- | --- | --- | --- |
| [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | DeBERTa-v3-base, 184M | contradiction / entailment / neutral | Apache-2.0 | MNLI, SNLI | MNLI-mm 90.0, SNLI 92.4 | Official fp32, and int8 for AVX2, AVX-512 and VNNI (244 MB) |
| [cross-encoder/nli-deberta-v3-xsmall](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) | DeBERTa-v3-xsmall, 71M | Three-way | Apache-2.0 | MNLI, SNLI | MNLI-mm 87.8, SNLI 91.6 | Official, int8 87 MB |
| [MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c) | DeBERTa-v3-base, 184M | entailment / not_entailment | MIT | Synthetic data (Apache-2.0), MNLI, FEVER | Mean macro-F1 0.619 over 28 zero-shot tasks (bart-large-mnli: 0.497) | None; we export it |
| [FacebookAI/roberta-large-mnli](https://huggingface.co/FacebookAI/roberta-large-mnli) | RoBERTa-large, 356M | Three-way | MIT | MNLI | MNLI 90.2 | None; RoBERTa exports cleanly. An accuracy ceiling, not a serving candidate |

- **Training-data licences decided this list.** Most stronger NLI models were trained on ANLI (CC BY-NC 4.0), either directly, through a teacher model or through their initial weights. That excludes:
  - MoritzLaurer's `mnli-fever-anli` models;
  - his `zeroshot-v2.0` models without `-c`, including the ModernBERT ones;
  - `tasksource/ModernBERT-*-nli`;
  - dleemiller's ModernCE, EttinX and finecat models.
- **Share-alike data:** SNLI is CC BY-SA 4.0, and FEVER is tagged CC BY-SA 3.0 and GPL-3.0. Whether share-alike terms carry over to model weights is unsettled. The risk is low, but it belongs in the licence review.
- **Two-way models can't detect contradiction.** The `zeroshot-v2.0-c` model is therefore only a candidate for zero-shot classification.
- **Hard contradictions:** the licence-clean models score about 0.34–0.43 on ANLI (third-party figures). Plain contradictions like the Nest Hub example in the guidelines should be caught. Implicit or numeric ones may not be, so `consistency` starts as a warning.
- **"Neutral" covers both new information and off-topic text,** so only the entailment and contradiction scores drive decisions.
- **One fact per premise.** A long fact sheet can overflow the 512-token limit. Checking each fact separately and taking the strongest contradiction is more robust.
- **DeBERTa-v3 int8 fails parity.** On the 19-pair parity set, every dynamic int8 version changed the top class for 26–58% of pairs. That includes the repo's own `quint8_avx2` and `qint8_arm64` files and our own exports (QInt8, QUInt8, per-channel). `nli-deberta-v3-base` is served in fp32 until static (calibrated) quantization is tried.
- **Label order differs between repos.** It is read from each model's `config.json`.

## Language ID

| Tool | Size | Licence (code / weights) | Evidence | Notes |
| --- | --- | --- | --- | --- |
| [GlotLID v4](https://huggingface.co/cis-lmu/glotlid) on a fastText runtime | 783 MB, 101 MB as the official quantised `model_v4.ftz` | MIT / Apache-2.0 plus notices | v3 on the probe: 19 of 20 English lines correct; all non-English lines scored 0.00 for English. Not yet re-run on v4 | v4 is trained on short spans and calibrated, which the README says makes it markedly better on short text than v3 |
| [fastText lid.176.ftz](https://fasttext.cc/docs/en/language-identification.html) | 0.9 MB | MIT / CC BY-SA 3.0 | Probe: 20 of 20 English, no non-English called English | Share-alike on the weights; frozen since 2019 |
| [lingua-py](https://github.com/pemistahl/lingua-py) | 170 MB wheel | Apache-2.0 | Its own tables, English: sentences 99%, word pairs 88%, single words 55%. Probe: 17 of 20 | English lines heavy with brand names score low |
| [py3langid](https://github.com/adbar/py3langid) | 4.4 MB | BSD-3-Clause | Probe: 16 of 20 | Small fallback |

- **The probe** was 20 short technical English lines and 11 lines in eight other languages. It shows direction, not accuracy. Calibration happens on real device-database text.
- **Excluded:**
  - OpenLID (GPL-3.0 weights);
  - NLLB LID-218 (CC BY-NC 4.0);
  - papluca/xlm-roberta-base-language-detection (non-commercial training data);
  - heliport (GPL-3.0);
  - langdetect, langid.py and gcld3, which are unmaintained or have no Python 3.12 wheels.
- **The fastText runtime** comes from community wheels (`fasttext-turbo` or `fasttext-predict`, both MIT), because upstream fastText was archived in March 2024. The wheel is pinned and checksummed.
- **Decision rule, to be calibrated:**
  1. A unit mostly in a non-Latin script is not English, with no model needed.
  2. Mask before scoring: tokens with digits, tokens with several capitals, mid-sentence capitalised words, and the brand and protocol vocabulary.
  3. If fewer than about 15–20 letters remain, pass the unit and log it.
  4. Block only when P(English) is below 0.05 and another language scores at least 0.5.
  5. Warn between 0.05 and about 0.3.
  6. Also score the whole field, not just units.
  On the probe, masking took brand-heavy English lines from 0.37–0.54 to 1.00.

## Personal data and secrets

Most of this is patterns, not models. A model is only needed for people's names.

- **Mask the device's own data first:** brand, model names, identifiers, and documented defaults such as a default access-point name or IP address.
- **Patterns:**
  - email addresses;
  - secrets from the gitleaks rule set (222 rules, MIT) and detect-secrets plugins (Apache-2.0), including JWTs, which covers Home Assistant long-lived tokens;
  - Wi-Fi credentials introduced by a keyword;
  - IP addresses, flagged when private with an ownership cue ("my server at …") or when public and outside documentation ranges;
  - serial numbers introduced by a cue;
  - MAC addresses, IMEIs (Luhn), non-placeholder EUI-64s and UUIDs;
  - smart-home pairing secrets (Matter setup codes, HomeKit codes, Tuya local keys).
- **NER for names** runs on the masked text only:

| Model | Base and size | Licence | Training data | Reported results | Notes |
| --- | --- | --- | --- | --- | --- |
| [knowledgator/gliner-pii-base-v1.0](https://huggingface.co/knowledgator/gliner-pii-base-v1.0) | DeBERTa-v3-small (GLiNER span mode), int8 ONNX 197 MB | Apache-2.0 | Not disclosed | F1 81.0 on the publisher's synthetic benchmark | Labels per request, so "brand" and "product" can compete with "person" |
| [gravitee-io/bert-small-pii-detection](https://huggingface.co/gravitee-io/bert-small-pii-detection) | BERT-small, 28.5M, int8 ONNX 29 MB | Apache-2.0 | Every source permissively licensed | F1 0.87 on its own eval split | Fixed 25 labels; plain token classification |
| [spaCy en_core_web_sm 3.8](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0) | CNN, 13 MB | MIT; OntoNotes licensed commercially by Explosion | OntoNotes 5 | PERSON F1 0.88 | Baseline: what a general NER does with brand names |

- **The `gliner` package requires PyTorch,** even on its ONNX path. Serving needs our own pre- and post-processing around ONNX Runtime, parity-tested against the package in `eval`.
- **Knowledgator doesn't disclose its training data.** If that blocks it, the Gravitee model has a verified permissive chain throughout.
- **Product names mistaken for people** are a documented failure for at least one GLiNER PII model. The negative set needs to be heavy with brand and product names.
- **Second wave:** [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) (Apache-2.0) labels *private* persons specifically, which fits the brand-name problem. But it is a 1.5B mixture-of-experts model (50M active), and running it on Python ONNX Runtime is unverified.
- **Presidio Analyzer** (MIT) can host the recognizers and the allowlist masking. Its core needs spaCy, not PyTorch.
- **gitleaks patterns are Go RE2.** How cleanly they port to Python regex needs a test pass.
- **Excluded:**
  - Piiranha (CC BY-NC-ND 4.0);
  - ai4privacy models and datasets (non-commercial or conflicting terms);
  - dslim/bert-base-NER (CoNLL-2003 text is research-only);
  - nvidia/gliner-PII (custom NVIDIA licence, needs legal review);
  - trufflehog (AGPL-3.0);
  - secrets-patterns-db (CC BY-SA 4.0);
  - scrubadub (unmaintained);
  - spaCy `en_core_web_trf` (pulls in PyTorch).

## CPU serving

Export every model to ONNX and serve them all from one ONNX Runtime process. PyTorch is then only needed in the training image, not the serving image.

- **Export and quantize** with [Hugging Face Optimum](https://huggingface.co/docs/optimum/v1.27.0/quicktour): ONNX export, then dynamic int8 quantization for the text encoders. Image models are also benchmarked in fp32 and int8, because quantization can hurt vision transformers more.
- **Benchmark on the real server CPU.** int8 and bf16 speedups depend on which x86 instruction set extensions the cores have (for example AVX-512 VNNI, AMX). Benchmarks on a laptop or a different server generation don't carry over.
- **Re-run the full evaluation after every export or quantization step,** as the testing plan requires. The quantized model is the one being tested.
- **Separate thread pools per model,** so a burst of image requests doesn't starve text requests. Latency budgets are tested per endpoint.
- **Model-size budget:** the dedicated models are each roughly 90–310M parameters. All of them together fit comfortably in a few GB of RAM, and int8 reduces that further.

## Sources

- [Freepik/nsfw\_image\_detector model card](https://huggingface.co/Freepik/nsfw_image_detector/blob/main/README.md)
- [Falconsai/nsfw\_image\_detection model card](https://huggingface.co/Falconsai/nsfw_image_detection/blame/93173218e15260d257912a3357b1ea34cdda5b68/README.md)
- [AdamCodd/vit-base-nsfw-detector overview](https://www.mixpeek.com/model/AdamCodd/vit-base-nsfw-detector)
- [Image-Guard-2.0 article](https://huggingface.co/blog/prithivMLmods/image-guard-models)
- [SigLIP 2 base model card](https://huggingface.co/google/siglip2-base-patch16-256)
- [Detoxify on PyPI](https://pypi.org/project/detoxify)
- [textdetox/xlmr-large-toxicity-classifier model card](https://huggingface.co/textdetox/xlmr-large-toxicity-classifier)
- [mmBERT announcement](https://huggingface.co/blog/mmbert) and [mmBERT-small model card](https://huggingface.co/jhu-clsp/mmBERT-small)
- [ModernBERT-base model card](https://huggingface.co/answerdotai/ModernBERT-base/blob/refs%2Fpr%2F85/README.md)
- [SetFit announcement](https://huggingface.co/blog/setfit)
- [Vale README](https://cdn.jsdelivr.net/gh/vale-cli/vale@v3/README.md)
- [Self-hosting LanguageTool (credativ)](https://www.credativ.de/en/blog/credativ-inside/languagetool-powerful-language-checking-on-your-own-network/)
- [Hugging Face Optimum quick tour](https://huggingface.co/docs/optimum/v1.27.0/quicktour)
- NLI: [cross-encoder NLI models](https://huggingface.co/cross-encoder/nli-deberta-v3-base), [MoritzLaurer zero-shot v2.0 paper](https://arxiv.org/abs/2312.17543) and [dataset licence list](https://github.com/MoritzLaurer/zeroshot-classifier/blob/7f82e4ab88d7aa82a4776f161b368cc9fa778001/v1_human_data/datasets_overview.csv), [ANLI licence](https://huggingface.co/datasets/facebook/anli), [Open NLI leaderboard](https://huggingface.co/spaces/dleemiller/open-nli-leaderboard)
- Language ID: [GlotLID repository](https://github.com/cisnlp/GlotLID) and [paper](https://arxiv.org/abs/2310.16248), [lingua-py accuracy tables](https://github.com/pemistahl/lingua-py/tree/main/tables), [fastText-turbo](https://github.com/CommonGlot/fastText-turbo), [OpenLID licence](https://github.com/laurieburchell/open-lid-dataset)
- Personal data: [Presidio](https://github.com/data-privacy-stack/presidio), [GLiNER ONNX export](https://github.com/urchade/GLiNER/blob/main/docs/convert_to_onnx.md), [gitleaks](https://github.com/gitleaks/gitleaks), [detect-secrets](https://github.com/Yelp/detect-secrets), [OpenAI Privacy Filter model card](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf)
