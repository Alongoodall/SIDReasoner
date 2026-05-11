.PHONY: help sync install-dev data precommit lint format clean

UV ?= uv
export UV_CACHE_DIR ?= .cache/uv
DATA_FILE_ID ?= 1S92QvaEs7aAhCwWOUovyb4y5P-pXwUhH
DATA_ARCHIVE ?= data/sidreasoner_amazon_dataset.zip
DATA_DIR ?= data/Amazon
DATA_TMP ?= data/.extract_tmp

help:
	@echo "Targets:"
	@echo "  make sync         Create/update the uv environment"
	@echo "  make install-dev  Sync dev tools and install pre-commit hooks"
	@echo "  make data         Download and extract the Amazon dataset"
	@echo "  make lint         Run ruff checks"
	@echo "  make format       Format Python files with ruff"
	@echo "  make precommit    Run all pre-commit hooks"

sync:
	$(UV) sync

install-dev:
	$(UV) sync --group dev
	$(UV) run pre-commit install

data:
	mkdir -p data
	if [ ! -f "$(DATA_ARCHIVE)" ]; then \
		$(UV) tool run --isolated --from gdown gdown "$(DATA_FILE_ID)" -O "$(DATA_ARCHIVE)"; \
	fi
	rm -rf "$(DATA_TMP)"
	mkdir -p "$(DATA_TMP)"
	if unzip -tq "$(DATA_ARCHIVE)" >/dev/null 2>&1; then \
		unzip -q "$(DATA_ARCHIVE)" -d "$(DATA_TMP)"; \
	else \
		tar -xf "$(DATA_ARCHIVE)" -C "$(DATA_TMP)"; \
	fi
	rm -rf "$(DATA_DIR)"
	if [ -d "$(DATA_TMP)/Amazon" ]; then \
		mv "$(DATA_TMP)/Amazon" "$(DATA_DIR)"; \
	elif [ "$$(find "$(DATA_TMP)" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ]; then \
		mv "$$(find "$(DATA_TMP)" -mindepth 1 -maxdepth 1 -type d)" "$(DATA_DIR)"; \
	else \
		mkdir -p "$(DATA_DIR)"; \
		cp -R "$(DATA_TMP)/." "$(DATA_DIR)/"; \
	fi
	rm -rf "$(DATA_TMP)"
	@echo "Dataset is available at $(DATA_DIR)"

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

precommit:
	$(UV) run pre-commit run --all-files

clean:
	rm -rf .ruff_cache .pytest_cache "$(DATA_TMP)"
