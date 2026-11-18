# The model server, the harness and the configuration benchmark. See README.md.

COMPOSE := docker compose
VARIANTS ?= toxicity/detoxify-unbiased@fp32 toxicity/detoxify-unbiased@int8
MODEL ?= toxicity/detoxify-unbiased@fp32
BENCH_ARGS ?=
QUALITY_ARGS ?=
TOXICITY_VARIANTS := toxicity/detoxify-unbiased@fp32 toxicity/detoxify-unbiased@int8 \
	toxicity/detoxify-original@fp32
# Every variant in config/models.json; `make export-all` exports them before a bundle.
ALL_VARIANTS := $(TOXICITY_VARIANTS) nli/nli-deberta-v3-base@fp32 language/glotlid-v4@ftz \
	image-embed/siglip2-base-256@fp32 obscenity/freepik-eva02@fp32
EVAL_CONTAINER := ddb-checks-eval
BUNDLE_PLATFORM ?= linux/amd64
BUNDLE_IMAGE ?= ddb-checks/serve-bundle:0.1.0
# Production bundles leave the playground out; true includes it for a test deployment.
BUNDLE_PLAYGROUND ?= false
BUNDLE_TAR := build/serve-bundle-0.1.0-$(subst /,-,$(BUNDLE_PLATFORM)).tar.gz
MODELS_CONTAINER := ddb-checks-models

# Run one eval command, then copy the reports volume out, so no host directory has to be
# shared with the Docker VM. The eval command's exit status is kept.
define run_eval
@docker rm --force $(EVAL_CONTAINER) >/dev/null 2>&1 || true
$(COMPOSE) run --name $(EVAL_CONTAINER) eval $(1); status=$$?; mkdir -p reports && docker cp $(EVAL_CONTAINER):/reports/. reports/; docker rm $(EVAL_CONTAINER) >/dev/null; exit $$status
endef

.PHONY: toxicity-quality build export export-all serve parity bench quality bundle down lint test \
	validate compare scores

# Toxicity quality on Civil Comments and the device-text set, for every toxicity variant.
# Detoxify original has to pass parity before its quality numbers mean anything.
toxicity-quality: build
	$(MAKE) export VARIANTS="$(TOXICITY_VARIANTS)"
	$(MAKE) bench MODEL=toxicity/detoxify-original@fp32
	$(foreach variant,$(TOXICITY_VARIANTS),$(MAKE) quality MODEL=$(variant) &&) true

build:
	$(COMPOSE) --profile eval build

export:
	$(COMPOSE) run --rm eval export $(VARIANTS)

export-all:
	$(MAKE) export VARIANTS="$(ALL_VARIANTS)"

# Restart serve with only $(MODEL) loaded, so nothing else competes for its cores.
serve:
	SERVE_MODELS=$(MODEL) $(COMPOSE) up --detach --wait --force-recreate serve

parity: serve
	$(call run_eval,parity $(MODEL))

bench: serve
	$(call run_eval,bench $(MODEL) $(BENCH_ARGS))

quality: serve
	$(call run_eval,quality $(MODEL) $(QUALITY_ARGS))

# A serve image with every exported model inside, saved as a tar for `docker load` or a
# Portainer image import. Export the models first.
bundle:
	rm -rf build/models && mkdir -p build
	@docker rm --force $(MODELS_CONTAINER) >/dev/null 2>&1 || true
	docker create --name $(MODELS_CONTAINER) -v ddb-checks_models:/models ddb-checks/serve:dev >/dev/null
	docker cp $(MODELS_CONTAINER):/models build/models; status=$$?; docker rm $(MODELS_CONTAINER) >/dev/null; exit $$status
	docker buildx build --platform $(BUNDLE_PLATFORM) -f serve/Dockerfile --target bundle \
		--build-arg PLAYGROUND=$(BUNDLE_PLAYGROUND) \
		--build-context models=build/models -t $(BUNDLE_IMAGE) --load .
	docker save $(BUNDLE_IMAGE) | gzip > $(BUNDLE_TAR)
	@ls -lh $(BUNDLE_TAR)

down:
	$(COMPOSE) down

lint:
	uvx ruff check .
	uvx ruff format --check .

test:
	cd checks && uv run --package ddb-checks pytest -q
	cd serve && uv run --package ddb-serve pytest -q
	cd eval && uv run --package ddb-eval pytest -q

# Configuration and benchmark. These need only the light ddb-checks package, not Docker.
CHECKS := uv run --package ddb-checks python -m ddb_checks
BASE ?= origin/main
SERVE_URL ?= http://localhost:8000

# Check every config file against its schema and the others.
validate:
	uv run --package ddb-checks python -c "from pathlib import Path; from ddb_checks import config; \
	c = config.load(Path('config')); print(f'config {c.version} is valid: {len(c.rules)} rules')"

# Compare config/ with $(BASE) on the benchmark cases. Offline for threshold changes; set
# COMPARE_ARGS=--serve-url=... when models, prompts, mask words or fact templates change.
compare:
	$(CHECKS) compare --base $(BASE) $(COMPARE_ARGS)

# Refresh the benchmark's score cache from a model server (make serve MODEL='*' first).
scores:
	$(CHECKS) scores --serve-url $(SERVE_URL)
