.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf src/__pycache__ data/output .mypy_cache llm_sdk/llm_sdk/__pycache__

lint:
	uv run mypy src --follow-imports=silent \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--warn-return-any
	uv run flake8 src

lint-strict:
	uv run mypy src --follow-imports=silent --strict
	uv run flake8 src
