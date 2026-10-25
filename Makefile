# The checks CI runs. `make` runs all of them; each also runs on its own.
RUN := uv run --locked

.PHONY: ci lint typecheck test docs

ci: lint typecheck test docs

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck:
	$(RUN) ty check

test:
	$(RUN) pytest -q

# The documentation, as Read the Docs builds it: any warning fails it.
docs:
	$(RUN) sphinx-build -W --keep-going -q -b html docs docs/_build/html
