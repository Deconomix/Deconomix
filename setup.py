from setuptools import setup, find_packages

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
  name = "deconomix",
  version = "1.0.0",
  author = "Malte Mensching-Buhr, Thomas Sterr, Dennis Voelkl, Franziska Goertler, Michael Altenbuchinger",
  author_email = "michael.altenbuchinger@bioinf.med.uni-goettingen.de",
  description = "Provides methods for cellular composition, hidden background and gene regulation estimation of omics bulk mixtures.",
  long_description = long_description,
  long_description_content_type = "text/markdown",
  packages = find_packages(),
  py_modules = [
    "methods",
    "utils"
  ],
  install_requires = [
    "numpy>=1.23.0",
    "scikit-learn",
    "qpsolvers>=4.3.2",
    "qpsolvers[quadprog]",
    "qpsolvers[scs]",
    "qpsolvers[proxqp]",
    "qpsolvers[clarabel]",
    "proxsuite",
    "matplotlib",
    "pandas>=2.2.2",
    "pyarrow",
    "seaborn",
    "tabulate",
    "scipy",
    "scs",
    "tqdm>=4.66.4",
    "torch>=2.3.1"
  ]
)
