# The checks CI runs. `make` runs all of them; each also runs on its own.
RUN := uv run --locked

.PHONY: ci lint typecheck test

ci: lint typecheck test

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck:
	$(RUN) ty check

test:
	$(RUN) pytest -q
