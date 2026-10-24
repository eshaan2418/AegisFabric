# AegisFabric — top-level entry points.
#   make test             unit tests (host) + Rust tests
#   make test-lab         integration tests inside the privileged Linux lab container
#   make ci               lint + test + test-lab (what CI runs)
#   make reproduce-quick  < 1 h subset of every experiment, regenerates figures
#   make reproduce-full   full experiment campaign (several hours)
#   make paper            build paper/main.pdf

SHELL := /bin/bash
export PATH := $(HOME)/.cargo/bin:/opt/homebrew/opt/openjdk/bin:$(PATH)
UV ?= uv
PY := $(UV) run python
LAB_IMAGE ?= aegis-lab:latest
GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell test -n "$$(git status --porcelain --untracked-files=no 2>/dev/null)" && echo 1 || echo 0)
LAB_RUN := docker run --rm --privileged --cgroupns=host -e AEGIS_GIT_COMMIT=$(GIT_COMMIT) -e AEGIS_GIT_DIRTY=$(GIT_DIRTY) \
	-v $(CURDIR):/work -v aegis-cargo-target:/work/ckpt/target $(LAB_IMAGE)

.PHONY: setup lab-image lint test test-py test-rust test-formal test-lab ci lab-shell probe data \
	reproduce-quick reproduce-full paper audit clean

setup:
	$(UV) sync --all-extras

lab-image:
	docker build -f docker/lab.Dockerfile --secret id=uvindex,env=UV_INDEX_URL -t $(LAB_IMAGE) .

probe:
	$(PY) -m bench.probe_env --image $(LAB_IMAGE)

data:
	$(PY) -m data.synth.finance --out data/generated/corpus --n 2000 --seed 0

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	@if [ -f ckpt/Cargo.toml ]; then cd ckpt && cargo fmt --check && cargo clippy -q --all-targets -- -D warnings; fi
	$(PY) -m bench.secret_scan

test: test-py test-rust

test-py:
	$(UV) run pytest -q

test-rust:
	@if [ -f ckpt/Cargo.toml ]; then cd ckpt && cargo test -q --release; fi

test-formal:
	bash formal/get_tools.sh && $(UV) run pytest -q -m slow -o addopts="" tests/test_formal.py

test-lab:
	$(LAB_RUN) bash -lc 'pytest -q -m "netlab" -o addopts="" tests'

ci: lint test test-formal test-lab

lab-shell:
	docker run --rm -it --privileged --cgroupns=host -v $(CURDIR):/work $(LAB_IMAGE) bash

reproduce-quick:
	bash bench/reproduce.sh quick

reproduce-full:
	bash bench/reproduce.sh full

paper:
	$(PY) -m bench.make_figures
	cd paper && tectonic -X compile main.tex

audit:
	$(PY) -m bench.audit_numbers

clean:
	rm -rf .pytest_cache .ruff_cache
