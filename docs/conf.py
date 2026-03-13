import importlib.metadata
from typing import Any

project = "airbornegeo"
copyright = "2025, Matt Tankersley"
author = "Matt Tankersley"
version = release = importlib.metadata.version("airbornegeo")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    # "sphinx.ext.viewcode",
    "nbsphinx",
    # githubpages just adds a .nojekyll file
    "sphinx.ext.githubpages",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    #
    # Runtime deps
    #
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
    ("py:class", "optional"),
    ("py:class", "optuna.trial"),
    ("py:class", "optuna.study"),
    ("py:class", "optuna.storages.BaseStorage"),
    ("py:class", "plotly.graph_objects.Figure"),
    ("py:class", "Ellipsis"),
]

always_document_param_types = True
# add_module_names = False
# add_function_parentheses = False


nbsphinx_execute = "auto"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png2x'}",
    "--InlineBackend.rc=figure.dpi=96",
]

nbsphinx_kernel_name = "python3"

# HTML output configuration
# -----------------------------------------------------------------------------
# html_title = f'{project} <span class="project-version">{version}</span>'
html_title = f"{project}"
html_last_updated_fmt = "%b %d, %Y"
html_copy_source = True
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = False
html_theme = "sphinx_book_theme"
html_theme_options: dict[str, Any] = {
    "repository_url": "https://github.com/mdtanker/airbornegeo",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "home_page_in_toc": False,
    "version_selector": True,
}
