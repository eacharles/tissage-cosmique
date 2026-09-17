"""Sphinx configuration for tissage-cosmique documentation."""

project = "tissage-cosmique"
copyright = "2026, LSST Dark Energy Science Collaboration (DESC)"
author = "Eric Charles"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
]

# AutoAPI — auto-generate API reference from source
autoapi_type = "python"
autoapi_dirs = ["../src/tissage_cosmique"]
autoapi_ignore = ["*/__pycache__/*", "*/_version.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = True

# Napoleon — NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx — cross-reference external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

# General
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_static_path = ["_static"]
