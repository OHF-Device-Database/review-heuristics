# Configuration

Everything that decides how contributions are checked lives in [`config/`](../config) as JSON. The model server applies it to every contribution, and production uses a tagged release of it. Improving the checks usually means editing one of these files and showing, with the benchmark, that the change helps.

| File | What it controls | Typical change |
| --- | --- | --- |
| [`checks.json`](../config/checks.json) | The rules: which model each uses, which fields it reads, the score at which it warns or rejects, and the message and guideline link the contributor sees | Tune a threshold, reword a message, move a rule from reject to warn |
| [`image-classes.json`](../config/image-classes.json) | The image taxonomy, as text prompts for zero-shot classification | Reword or add prompts for a class |
| [`language.json`](../config/language.json) | Words removed before language identification, such as protocol names | Add a term that keeps making English look foreign |
| [`facts.json`](../config/facts.json) | How structured device data becomes short sentences for the NLI rules | Reword a template the model misreads |
| [`models.json`](../config/models.json) | Every model variant the server can load, pinned to a source revision | Add or replace a model (maintainers; needs a rebuild) |

Each file has a JSON Schema in [`config/schemas/`](../config/schemas). Editors that understand `$schema`, such as VS Code, check the file and show the description of each field as you type. JSON has no comments, so explanations go in `description` and `notes` fields.

## How a rule works

A rule turns scores into an action for each unit of text: the name, each sentence of the summary, each getting-going step, each quirk. Image rules act on each image.

```json
{
  "id": "abuse_spam",
  "type": "toxicity",
  "model": "toxicity/detoxify-unbiased@int8",
  "fields": ["name", "summary", "getting_going", "quirks"],
  "thresholds": {
    "insult": { "warn": 0.7 },
    "threat": { "warn": 0.5, "reject": 0.9 }
  },
  "message": "This reads as abusive. Describe the device and its behaviour without insults or attacks.",
  "guideline": "docs/text-contribution-guidelines.md#rules"
}
```

- A unit whose score reaches `reject` is rejected; one that reaches `warn` is flagged. Leave out `reject` and the rule only ever warns.
- A contribution's outcome is `reject` if any unit is rejected, `warn` if any is flagged, and `accept` otherwise. A reviewer sees every contribution with warnings.
- Each hit in the result carries the scores behind it, so a wrong decision can be traced to the threshold that caused it.

| Type | Signals you can set thresholds on |
| --- | --- |
| `toxicity` | The model's heads, as listed under `labels` in `models.json` |
| `language` | Its own settings: `warn_below`, `reject_below` and `reject_needs_other_at_least` on the probability of `language`; units with fewer than `min_letters` letters left after masking are skipped |
| `nli_contradiction` | `contradiction`: the strongest against any one fact from `facts.json` |
| `nli_entailment` | `entailment`: the strongest from any one fact; fires when the structured data already says the same |
| `image_class` | A class from `image-classes.json`, on its zero-shot probability |
| `obscenity` | `nsfw`, `at_least_low`, `at_least_medium`, `at_least_high`, or a model label |

## Testing a change

```bash
make validate
```

This checks every file against its schema and against the others; for example, that each rule's model is in `models.json` and each threshold names a signal that model produces.

```bash
make compare
```

This runs the benchmark cases in [`benchmarks/cases/`](../benchmarks/cases) through the configuration on your branch and the one on `main`, and prints what changed:

```
| Rule | Good, wrongly rejected | Good, wrongly warned | Bad, caught |
| --- | --- | --- | --- |
| **All rules** | 0/399 | 31/399 → 29/399 | 20/22 |
| abuse_spam | 0/162 | 26/162 → 24/162 | 0/0 |

**Changed decisions: 2**

| Case | Rule | Decision | Expected |
| --- | --- | --- | --- |
| device-text/swearing/19 | abuse_spam | warn → pass | pass |
```

- **The hard rule:** a change may not reject more good cases than before. `make compare` exits with an error when it does. Everything else is for reviewers to weigh.
- **Changed decisions** lists every case that flipped, which is usually more telling than the totals.
- **No model server needed** for changes to thresholds, actions, fields, messages or links. Model scores are cached in [`benchmarks/scores/`](../benchmarks/scores), and the comparison takes under a second. It needs only the small `ddb-checks` package: `uv` installs it on first run, without PyTorch or ONNX Runtime.
- **A model server is needed** when a change alters what the models see: a different model, image prompts, mask words or fact templates. Run one (see the [README](../README.md)), then `make compare COMPARE_ARGS=--serve-url=http://localhost:8000`, and `make scores` to update the cache in your branch.

`BASE=<ref>` compares against something other than `origin/main`. CI runs `make validate` and `make compare` on every pull request, shows the comparison in the run's summary, and fails the check on the hard rule or on missing scores.

## Benchmark cases

Each line of a file in [`benchmarks/cases/`](../benchmarks/cases) is one contribution and the outcome each rule should have on it:

```json
{"id": "nest/quirk-zigbee-coordinator",
 "contribution": {
   "device": {"brand": "Google", "category": "smart display", "protocols": ["Wi-Fi", "Thread"]},
   "fields": {"quirks": ["The Nest Hub acts as a Zigbee coordinator for your lights."]}},
 "expect": {"consistency": "warn"},
 "why": "The guideline's contradiction example: the Nest Hub has no Zigbee radio."}
```

- `expect` maps rule ids to `pass`, `warn` or `reject`. A case is only judged on the rules it lists, so it doesn't need updating when an unrelated rule is added.
- Images are referenced by a `path` relative to the repository root, and must be licensed for this repository; see the README in each image folder.
- Every false positive reported on a real contribution should become a case. That is how the benchmark grows.
- Offensive text and explicit images are never committed. Offensive cases can be kept locally in `benchmarks/private/cases/`, which git ignores; run them with `--cases benchmarks/private`.
- Adding cases changes the inputs, so run `make scores` against a model server and commit the updated `benchmarks/scores/`.

## Releasing a configuration

1. Bump `version` in `checks.json`. The server reports it, and a request can insist on it with `config_version`.
2. Merge, then tag the release (`config-v0.2.0`). Production builds the model server from the tag, so a merged change goes live only when it is tagged and deployed.
3. A change to `models.json` also needs the new variants exported and checked (`make export`, `make parity`) before the image is built.
