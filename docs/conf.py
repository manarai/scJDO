import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = 'scJDO'
author = 'Tommy W. Terooatea, David Redd, and contributors'
copyright = '2026, scJDO developers'
release = '0.3.0'
version = '0.3.0'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

myst_enable_extensions = [
    'amsmath',
    'dollarmath',
    'colon_fence',
    'deflist',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme = 'pydata_sphinx_theme'
html_title = 'scJDO'
html_show_sourcelink = True
html_theme_options = {
    'show_toc_level': 2,
    'navigation_depth': 4,
    'collapse_navigation': False,
    'navbar_align': 'left',
    'github_url': 'https://github.com/manarai/scJDO',
    'logo': {
        'text': 'scJDO',
    },
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/manarai/scJDO',
            'icon': 'fa-brands fa-github',
        },
    ],
    'switcher': {
        'json_url': '_static/switcher.json',
        'version_match': 'stable',
    },
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
    'anndata': ('https://anndata.readthedocs.io/en/latest/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'inherited-members': False,
}
autosummary_generate = True
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_mock_imports = [
    'torch', 'scanpy', 'scvelo', 'cellrank', 'anndata', 'numba', 'scipy',
    'sklearn', 'sklearn.decomposition', 'sklearn.neighbors', 'sklearn.preprocessing',
    'matplotlib', 'matplotlib.pyplot', 'seaborn', 'h5py',
    'tqdm', 'networkx', 'decoupler',
]

nitpicky = False
