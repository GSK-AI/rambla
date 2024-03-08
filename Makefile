define PROJECT_HELP_MSG

Usage:
	make help          		show this message


	make init
		creates/recreates conda environment, sets up pre-commit hooks, installs base project package in develop mode
	make clean_env
		only clears the conda cache
	make conda_create
		creates new conda environment following repo naming
	make conda_delete
		deletes conda environment

endef
export PROJECT_HELP_MSG

.PHONY: help
help:
	echo "$$PROJECT_HELP_MSG"

PKG=rambla
ENV_FILE=environment.yml
ENV_TAG=env

.PHONY: init
init: conda_delete conda_create init_install

.PHONY: clean_env
clean_env:
	conda clean --all --yes

.PHONY: conda_create 
conda_create:
	@echo "Creating conda env $(PKG)_$(ENV_TAG) from ${ENV_FILE}"
	conda env create -n $(PKG)_$(ENV_TAG) -f ${ENV_FILE}

.PHONY: conda_delete
conda_delete:
	@echo "Deleting conda env $(PKG)_$(ENV_TAG)"
	conda env remove -n $(PKG)_$(ENV_TAG)

.PHONY: init_install
init_install:
	@echo "Installing pre-commit and ${PKG} package"
	source $$(conda info --base)/etc/profile.d/conda.sh; conda activate $(PKG)_$(ENV_TAG); pre-commit install; pip install -e .
