Methods
=======
The `deconomix.methods` submodule provides the models for DTD and ADTD and the respective methods to train them.

DTD Class
---------

.. autoclass:: deconomix.methods.DTD
   :members:
   :exclude-members: Model, Model.forward

ADTD Class
----------

.. autoclass:: deconomix.methods.ADTD
   :members:

Hyperparameter Search
---------------------

Finding suitable hyperparameters is not trivial. We provided an extensive gridsearch in the example section of the repository (`gridsearch.py`, `Dockerfile`). As a low-cost alternative, you can use this function to estimate a good `lambda_2` for the case of `lambda_1` to infinity:

.. automodule:: deconomix.methods
   :members:
   :exclude-members: DTD, ADTD
   
