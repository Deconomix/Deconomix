Deconomix Documentation
==================
The `deconomix` package provides functions for (Adaptive) Digital Tissue Deconvolution.
It is split into two collections of functions, `deconomix.utils` and `deconomix.methods`, where the first provides useful utilities 
around the usual deconvolution workflow and the latter provides the actual models and training routines.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   utils
   methods

Modules
------------
In the article, we divided the functionalities of Deconomix semantically into three different modules, which are realizable with the Python functions we provide in the package. However, the functions have a broad spectrum of use-cases overall, which we want to keep for advanced users. Therefore we want to give a summary on how to use the modules described in the article here:

Module 1: Learning Gene Weights:
.. code-block:: bash
   X_ref, Y_train, C_train = deconomix.utils.simulate_data(scRNA_df, n_mixtures=10000)
   module1 = deconomix.methods.DTD(X_ref, Y_train, C_train)
   module1.run()
   gene_weights = module1.gamma

Module 2: Cellular Composition (with hidden background) 
.. code-block:: bash
   module2 = deconomix.methods.ADTD(X_ref, test_bulks_df, gene_weights, C_static = False, Delta_static = True)
   module2.run()
   Cellular_Contributions   = module2.C_est
   Hidden_Contributions     = module2.c_est
   Hidden_Consensus_Profile = module2.x_est

Module 3: Gene Regulation
.. code-block:: bash
   # Search hyperparameter with deconomix.methods.HPS() or determine otherwise, e.g. 1e-6
   module3 = deconomix.methods.ADTD(X_ref, test_bulks_df, gene_weights, C_static=True, Delta_static = true, lambda2 = 1e-6)
   # This also updates the cellular contributions slightly.
   Cellular_Contributions = module3.C_est
   Hidden_Contributions   = module3.c_est
   Hidden_Consensus_Profile = module3.x_est
   Gene_Regulation_Factors = module3.Delta_est


Installation
------------
You can install the package using the official PyPI repositories:

.. code-block:: bash

   pip install deconomix

Alternatively, clone the repository and install from the directory:

.. code-block:: bash
   
   pip install git+https://github.com/Deconomix/Deconomix.git
