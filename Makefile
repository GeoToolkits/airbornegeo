PROJECT=airbornegeo

####
####
# install commands
####
####

create:
	mamba env create --file environment.yml --name $(PROJECT)

install:
	pip install --no-deps -e .

update:
	mamba env update --file environment.yml --name $(PROJECT) --prune

remove:
	mamba env remove --name $(PROJECT)
