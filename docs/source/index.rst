Deconomix Documentation
==================
The `Deconomix` package provides functions for building cell-type deconvolution pipelines using different models based on the (A)DTD algorithm.
It is split into four Python modules, where `deconomix.models` contains the Deconomix class. It is a wrapper for all models introduced in the article and offers a straightforward application to your data. The modules `deconomix.utils` and `deconomix.methods` provide the underlying utilities and algorithms, respectively and are aimed at advanced users. The module `deconomix.hps` contains the hyperparameter search function usable to find  a suitable hyperparameter for the Deconomix+h,r model.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   models
   utils
   methods
   hps

Installation
------------
You can install the package using the official PyPI repositories:

.. code-block:: bash

   pip install deconomix

Alternatively, clone the repository and install from the directory:

.. code-block:: bash
   
   pip install git+https://github.com/Deconomix/Deconomix.git
