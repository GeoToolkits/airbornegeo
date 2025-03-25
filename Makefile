PROJECT=airbornegeo
VERSION := $(shell grep -m 1 'version =' pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)

print-%  : ; @echo $* = $($*)
####
####
# install commands
####
####

create:
	mamba env create --file environment.yml

install:
	pip install --no-deps -e .

remove:
	mamba env remove --name $(PROJECT)

pip_install:
	pip install $(PROJECT)[all]==$(VERSION)

conda_install:
	mamba create --name $(PROJECT) --yes --force --channel conda-forge $(PROJECT)=$(VERSION) pytest pytest-cov ipykernel

####
####
# test commands
####
####

test: test_coverage test_numba

test_coverage:
	NUMBA_DISABLE_JIT=1 pytest

test_numba:
	NUMBA_DISABLE_JIT=0 pytest -rP -m use_numba

####
####
# style commands
####
####

check:
	pre-commit run --all-files

pylint:
	pylint $(PROJECT)

style: check pylint

mypy:
	mypy src/$(PROJECT)