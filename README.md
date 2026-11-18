# Device Database contribution checks

The checks the Open Home Foundation Device Database runs on every contribution before a person reviews it. The checks cover abusive text, text that isn't English, text that contradicts or only repeats the device's structured data, images that don't show the device, and explicit images.

This repository has three parts:

- **[`config/`](config)**: what the checks do. Plain JSON that sets which models run and when each rule warns or rejects. Production uses a tagged release of it, so improving the checks is usually a pull request against these files. Start with [Configuration](docs/configuration.md).
- **[`benchmarks/`](benchmarks)**: known-good and known-bad contributions with the outcome each rule should have, so every change shows what it fixes and what it breaks.
- **The model server** (`serve/`) and its tooling (`checks/`, `eval/`): a CPU-only service that runs the models and applies the configuration, and a harness that exports and checks the models.

## Improving the checks

You don't need Docker or a model server for most changes. You need [`uv`](https://docs.astral.sh/uv/), and `make`.

1. Edit a file in `config/`, for example a threshold in `checks.json`.
2. Check it:

   ```bash
   make validate
   ```

3. See what it changes, compared with `main`:

   ```bash
   make compare
   ```

4. Open a pull request. CI runs the same comparison and shows it in the run's summary.

[Configuration](docs/configuration.md) explains each file, how rules decide, how to add benchmark cases, and which changes need a model server.

## Running the model server

Needs Docker with Compose.

```bash
make build
make export-all
make serve MODEL='*'
```

`export-all` downloads every model in `config/models.json` at its pinned revision and converts it to ONNX. Expect a few GB and several minutes. The server then listens on `http://localhost:8000`, with API docs at `/docs` and, in development builds, a page for trying every endpoint at `/playground`.

The main endpoint is `POST /v1/contributions/check`. It takes a contribution's structured data, text fields and images, and returns `accept`, `warn` or `reject`, with every rule hit and the scores behind it. The [spec](docs/spec.md) describes it and the per-model endpoints.

To check that each served model still matches the original it was converted from:

```bash
make parity MODEL=nli/nli-deberta-v3-base@fp32
```

`make scores` refreshes the benchmark's cached scores from the running server. You need it after changing models, image prompts, mask words, fact templates or benchmark cases.

### Deploying

`make bundle` builds an image with the exported models inside, saved to `build/` for `docker load`. [`deploy/portainer-stack.yaml`](deploy/portainer-stack.yaml) is an example stack for it. The API has no authentication, so run it behind the submission service or on a trusted network.

## Developing

`checks/`, `serve/` and `eval/` are one `uv` workspace with a shared lockfile:

| Package | What it is | Heavy dependencies |
| --- | --- | --- |
| `checks/` (`ddb_checks`) | Loads and validates the configuration, splits contributions into units, applies the rules, and runs the benchmark | None |
| `serve/` (`ddb_serve`) | The model server | ONNX Runtime, fastText, Pillow |
| `eval/` (`ddb_eval`) | Model export, parity, cost and quality reports | PyTorch, Transformers, timm |

```bash
make lint
make test
```

To add a model: add it to `config/models.json` with its source pinned to a commit (`repo@revision`). If its task is new, add an exporter (`_EXPORTERS` in `eval/ddb_eval/export.py`) and a loader (`_LOADERS` in `serve/ddb_serve/runtime.py`). Then run `make export VARIANTS=<id>` and `make parity MODEL=<id>`.

The [benchmark plan](docs/benchmark-plan.md) describes how candidate models are compared, and the [model shortlist](docs/model-shortlist.md) lists them.

## Licence

Apache-2.0; see [LICENSE](LICENSE). The images in `eval/datasets/images/` are licensed separately; see the README in each folder.
