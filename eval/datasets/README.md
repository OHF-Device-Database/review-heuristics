# Dataset manifests

Each manifest is JSON Lines, one item per line, with an `id` and either `text` or a `premise` and `hypothesis`.
A dataset's version is the SHA-256 of its manifest file; every report records it.

| Manifest | Used by | Contents |
| --- | --- | --- |
| `parity-text-v0.jsonl` | Text parity and cost runs | The examples from sections 4 and C.3 of the [text contribution guidelines](../../docs/text-contribution-guidelines.md) and technical terms that trip filters, so a wide range of scores is exercised. It has no labels and is not a quality benchmark. |
| `device-text-v0.jsonl` | Toxicity `quality` runs | 160 benign device-database-shaped texts, each with a `category`: technical terms that trip filters, mild swearing in otherwise useful text, neutral identity terms, complaints and plain device text. Lines taken from the guidelines name their section in `source`. `labeler: draft` marks labels nobody has reviewed yet. Offensive examples are kept out of this repository. |
| `parity-nli-v0.jsonl` | NLI parity runs | Premise and hypothesis pairs: short device facts as premises, and the guideline's example sentences as hypotheses, including contradictions. One synthetic premise runs over the token limit, so truncation is exercised. No labels; not a quality benchmark. |
| `images-ha-green-v0.jsonl` | Trying the image endpoints; a first look at zero-shot classification | 15 official Home Assistant Green photos and renders in `images/ha-green/` (CC BY-NC-SA 4.0, see its README), each with a draft label from the image taxonomy and a note on borderline calls. One product only, so it can't measure accuracy. |
| `images-cast-pro-1-v0.jsonl` | Trying the image endpoints; a first look at zero-shot classification | 8 Apollo Automation CAST PRO-1 product photos in `images/cast-pro-1/` (used with permission, see its README): in situ, port close-ups and both sides of the board, each with a draft label. One product only, so it can't measure accuracy. |

Civil Comments is not committed. `quality` downloads Jigsaw's `test_private_expanded.csv` (97,320 comments with toxicity and identity labels, CC0) from a pinned Hub mirror into the `hf-cache` volume, and checks it against a pinned SHA-256 (`eval/ddb_eval/civil_comments.py`).

Image parity runs use a synthetic set generated in code (`eval/ddb_eval/images.py`): JPEG, PNG, GIF and WebP files covering transparency, palettes, EXIF rotation, CMYK, greyscale and very small and large sizes. Parity compares served and original models on identical inputs, so the images don't need realistic content.
