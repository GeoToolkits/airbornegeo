# Install

## Install Python

Before installing _airbornegeo_, ensure you have Python 3.12 or greater downloaded.
If you don't, we recommend setting up Python with Miniforge.
See the install instructions [here](https://github.com/conda-forge/miniforge).

## Install _airbornegeo_ locally

There are 3 main ways to install `airbornegeo`. We show them here in order of simplest to hardest.

### Conda / Mamba

```{note}
`conda` and `mamba` are interchangeable
```

The easiest way to install this package and it's dependencies is with conda or mamba into a new virtual environment:

    conda create --name airbornegeo --yes --force airbornegeo --channel conda-forge

Activate the environment:

    conda activate airbornegeo

### Pip

Instead, you can use pip to install `airbornegeo`, but first you maybe need to install a few dependencies first with conda.
This is because a few dependencies rely on C packages, which can only be install with conda/mamba and not with pip.

Create a new virtual environment:

```
conda create --name airbornegeo --yes --force pygmt geopandas --channel conda-forge
```

activate the environment and use `pip` to install `airbornegeo`:

```
conda activate airbornegeo
pip install airbornegeo
```

### Development version

You can use pip, with the above created environment, to install the latest source from GitHub:

    pip install git+https://github.com/mdtanker/airbornegeo.git

Or you can clone the git repository and install:


    git clone https://github.com/mdtanker/airbornegeo.git
    cd airbornegeo
    pip install .

## Test your install

Run the following inside a Python interpreter:

```python
import airbornegeo

airbornegeo.__version__
```

This should tell you which version was installed.

To further test, you can clone the GitHub repository and run the suite of tests, see the [Contributors Guide](contributing.md).

A simpler method to ensure the basics are working would be to download any of the jupyter notebooks from the documentation and run them locally. On the documentation, each of the examples should have a drop down button in the top right corner to download the `.ipynb`.
