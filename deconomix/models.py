# Wrapper for Deconomix models, using algorithms from the methods module
# and utilities from the utils module
import numpy as np
import pandas as pd
from deconomix.methods import DTD, ADTD
from deconomix.utils import simulate_data, calculate_estimated_composition, plot_corr, calculate_corr
import warnings

class Deconomix:
    """Wrapper class for Deconomix algorithms, providing methods to fit, predict, and analyze
    cell type deconvolution models using single-cell and bulk gene expression data.
    Includes the naive, Deconomix, Deconomix+h, Deconomix+h,r models described in the article.

    The Deconomix class integrates core deconvolution algorithms (e.g., DTD, ADTD) and utilities
    for simulating artificial bulks, optimizing gene weights, and estimating cell compositions.

    Attributes
    ----------
    verbose : bool
        Whether to print verbose output during model fitting.
    X_ref : np.ndarray or None
        Reference matrix of single cell data used for training.
    Y_train : np.ndarray or None
        Matrix of simulated bulk mixtures.
    C_train : np.ndarray or None
        Ground truth cell compositions for the mixtures.
    gamma : np.ndarray or None
        Optimized gene weights obtained from the DTD algorithm.
    gamma_naive : np.ndarray or None
        Optionally, gene weights before optimization.
    C_est : np.ndarray or None
        Estimated cell type compositions.
    c_est : np.ndarray or None
        Optionally, other forms of estimated compositions.
    x_est : np.ndarray or None
        Optionally, estimated expressions.
    Delta_est : np.ndarray or None
        Optionally, estimated Delta from model predictions.
    """

    def __init__(self, verbose : bool = False):
        self.verbose = verbose
        self.X_ref = None
        self.Y_train = None
        self.C_train = None
        self.gamma = None
        self.gamma_naive = None
        self.C_est = None
        self.c_est = None
        self.x_est = None
        self.Delta_est = None

    def fit(self,
            sc_df : pd.DataFrame,
            n_mixtures : int = 1000):
        """
        Performs the gene weight optimization learned on single cell data needed for all Deconomix models.

        Parameters
        ----------
        sc_df : pd.DataFrame
            Data frame containing the single cell data with gene labels as rows and cell type labels as columns.
        n_mixtures : int (optional)
            Number of artificial bulks to simulate. Default is 1000.
        """

        self.gene_labels = sc_df.index
        self.celltype_labels = sc_df.columns.unique()
        self.sc_df = sc_df.values

        # Simulating Artificial Bulks
        if self.verbose:
            print("Simulating artificial bulks...")
        self.X_ref, self.Y_train, self.C_train = simulate_data(scRNA_df = sc_df,
                                                               n_mixtures = n_mixtures,
                                                               n_cells_in_mix = 100)
        if self.verbose:
            print("Artificial bulks simulated.")

        # Optimizing gene weights via DTD algorithm
        if self.verbose:
            print("Optimizing gene weights via DTD algorithm...")
        algo_DTD = DTD(X_mat = self.X_ref,
                       Y_mat = self.Y_train,
                       C_mat = self.C_train)
        algo_DTD.run(iterations=1000, plot=self.verbose)
        self.gamma = algo_DTD.gamma
        if self.verbose:
            print("Gene weights optimized.")

    def validate_fit(self,
                     test_sc_df : pd.DataFrame,
                     n_mixtures = 1000,
                     plot = True,
                     hidden_ct=None):
        """
        Validates the fit of Deconomix's gene weights by simulating test bulks from a separate single cell data set and predicting known cell type compositions with a Deconomix model or known and hidden cell type compositions with a Deconomix+h model.

        Parameters
        ----------
        test_sc_df : pd.DataFrame
            Data frame containing test single cell data with gene labels as rows and cell type labels as columns.
        n_mixtures : int (optional)
            Number of artificial bulks to simulate. Default is 1000.
        plot : bool (optional)
            Whether to plot the correlation between the true and estimated cell type compositions. Default is True.
        hidden_ct : str (optional)
            The cell type that is hidden in the training set. Default is None. If given, the Deconomix+h model is used to predict known and hidden cell type compositions.
        """
        if self.verbose:
            print("Simulating test bulks...")
        _, Y_test, C_test = simulate_data(scRNA_df = test_sc_df,
                                                               n_mixtures = n_mixtures,
                                                               n_cells_in_mix = 100)
        if self.verbose:
            print("Test bulks simulated.")
        
        C_test_est = calculate_estimated_composition(self.X_ref, Y_test, self.gamma)

        if hidden_ct == None:

            if plot:
                plot_corr(C_test, C_test_est,
                        title='Performance on test data\n model: Deconomix',
                        color = '#5e81ac')
            print(calculate_corr(C_test, C_test_est))

        elif hidden_ct is not None:
            # predict with Deconomix+h model
            algo_ADTD = ADTD(X_mat = self.X_ref,
                             Y_mat = Y_test,
                             gamma = self.gamma,
                             C_static = True,
                             Delta_static = True,
                             max_iterations = 1000)
            algo_ADTD.run(verbose = self.verbose)
            C_test_est = algo_ADTD.C_est
            c_test_est = algo_ADTD.c_est
            if plot:
                plot_corr(C_test, C_test_est,
                        title='Performance on test data\n model: Deconomix+h\n(cell type: ' + hidden_ct + ' hidden in training set)',
                        color = '#5e81ac',
                        hidden_ct = hidden_ct,
                        c_est = c_test_est)
            print(calculate_corr(C_test, C_test_est, hidden_ct = hidden_ct, c_est = c_test_est))

        
        

        
    

    def predict(self,
                bulk_df : pd.DataFrame,
                model : str = None,
                lambda1: float = np.inf,
                lambda2: float = None,
                max_iterations : int = 1000):
        """
        Predicts cell type compositions for a given bulk data set using a specified Deconomix model.

        Parameters
        ----------
        bulk_df : pd.DataFrame
            Data frame containing the bulk data with gene labels as rows and sample labels as columns.
        model : str (optional)
            The model to use for prediction. Default is None. Must be one of ['Naive', 'Deconomix', 'Deconomix+h', 'Deconomix+h,r', 'Custom'].
        lambda1 : float (optional)
            Regularization parameter only used for custom models. Default is np.inf.
        lambda2 : float (optional)
            Regularization parameter used for the cell-type-specific gene regulation in the Deconomix+h,r model. Default is None.
        max_iterations : int (optional)
            The maximum number of iterations for the ADTD algorithm used in the Deconomix+h, Deconomix+h,r and custom models. Default is 1000.
        """
        
        # Input Handling

        
        if model not in ["Naive", "Deconomix", "Deconomix+h", "Deconomix+h,r", "Custom"]:
            raise ValueError(f"Invalid model: {model}. Must be one of ['Naive', 'Deconomix', 'Deconomix+h', 'Deconomix+h,r', 'Custom'].")

        if lambda1 is not np.inf:
            raise Warning("You defined a custom lambda1, which will not be used in the available models, except the Custom one. Proceed only if truly intended.")

        if lambda2 is None and model == "Deconomix+h,r":
            raise ValueError("lambda2 must be specified for Deconomix+h,r model. Use the HPS function on your bulk data to find lambda2.")

        # Other Warnings
        if self.gamma is None:
            raise RuntimeError("The fit() function must be run before predict().")
        if self.C_est is not None:
            warnings.warn("You have already applied a model to the data. Predicting with a new model will overwrite estimates C_est, c_est, x_est and Delta_est.")
        if not bulk_df.index.equals(self.X_ref.index):
            raise ValueError("Gene labels (index) must be identical between X_ref and bulk_df.")


        # Extract sample ids from bulk_df
        sample_ids = bulk_df.columns

        # Caculate gamma naive
        self.gamma_naive = pd.Series(np.ones((self.X_ref.shape[0])))
        self.gamma_naive.index = self.X_ref.index

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        if model == "Naive":
            self.C_est = calculate_estimated_composition(self.X_ref, bulk_df, self.gamma_naive)
            self.c_est = None
            self.x_est = None
            self.Delta_est = None
            if self.verbose:
                print("Calculating C_est without optimized gene weights.")

        elif model == "Deconomix":
            self.C_est = calculate_estimated_composition(self.X_ref, bulk_df, self.gamma)
            self.c_est = None
            self.x_est = None
            self.Delta_est = None
            if self.verbose:
                print("Calculating C_est with optimized gene weights.")

        elif model == "Deconomix+h":
            if self.verbose:
                print("Applying ADTD algorithm to infer hidden background contributions.")
            algo_ADTD = ADTD(X_mat = self.X_ref,
                             Y_mat = bulk_df,
                             gamma = self.gamma,
                             C_static = True,
                             Delta_static = True,
                             max_iterations = max_iterations)

            algo_ADTD.run(verbose = self.verbose)
            self.C_est = algo_ADTD.C_est
            self.c_est = algo_ADTD.c_est
            self.x_est = algo_ADTD.x_est
            self.Delta_est = None

        elif model == "Deconomix+h,r":
            if self.verbose:
                print("Applying ADTD algorithm to infer hidden background contributions and cell-type-specific gene regulation.")
            algo_ADTD = ADTD(X_mat = self.X_ref,
                             Y_mat = bulk_df,
                             gamma = self.gamma,
                             C_static = True,
                             Delta_static = False,
                             lambda2 = self.lambda2,
                             max_iterations = max_iterations)
            algo_ADTD.run(verbose = self.verbose)
            self.C_est = algo_ADTD.C_est
            self.c_est = algo_ADTD.c_est
            self.x_est = algo_ADTD.x_est
            self.Delta_est = algo_ADTD.Delta_est

        # Add sample ids to the estimates if they are not None
        if self.C_est is not None:
            self.C_est.columns = sample_ids
        if self.c_est is not None:
            self.c_est.columns = sample_ids
        if self.x_est is not None:
            self.x_est.columns = ["hidden profile"]
