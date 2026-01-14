#### Imports ####################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from qpsolvers import solve_qp
import scipy.sparse as sp
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.ticker as ticker

# Suppress the specific UserWarning
#warnings.filterwarnings("ignore", category=UserWarning)
import scipy.sparse as sp
import sys


# Tqdm based on session:
def in_notebook():
    try:
        # Check if the Jupyter notebook environment exists
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (maybe an IDE)
    except NameError:
        return False  # Probably standard Python interpreter

# Use the result of the check to determine which tqdm to import
if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

#### DTD #########################################################################################

class DTD:
    """
    Class that implements Digital Tissue Deconvolution via Loss-function Learning
    
    Attributes
    ----------
    X_mat : pd.DataFrame
        Single cell reference matrix X containing (average) profiles as columns per celltype. 
        Shape: genes x celltypes
    Y_mat : pd.DataFrame
        Y matrix containing the generated artificial bulk profiles.
        Shape: genes x n_mixtures
    C_mat : pd.DataFrame
        C matrix containing the relative composition of the bulk mixtures in Y.
        Shape: cell types x n_mixtures
    """

    # Setup
    def __init__(self,
                 X_mat : pd.DataFrame,
                 Y_mat : pd.DataFrame,
                 C_mat : pd.DataFrame):
        
        # Input Handling

        # Check if the number of genes match between X_ref and Y_mat
        if X_mat.shape[0] != Y_mat.shape[0]:
            raise ValueError(f"Number of genes (rows) must match between X_mat and Y_mat. "
                            f"Found {X_mat.shape[0]} in X_mat and {Y_mat.shape[0]} in Y_mat.")

        # Check if the number of celltypes in X_ref matches the number of celltypes in C_mat
        if X_mat.shape[1] != C_mat.shape[0]:
            raise ValueError(f"Number of celltypes (columns) in X_mat must match the number of celltypes (rows) in C_mat. "
                            f"Found {X_mat.shape[1]} in X_mat and {C_mat.shape[0]} in C_mat.")

        # Check if the number of mixtures in Y_mat matches the number of mixtures in C_mat
        if Y_mat.shape[1] != C_mat.shape[1]:
            raise ValueError(f"Number of mixtures (columns) must match between Y_mat and C_mat. "
                            f"Found {Y_mat.shape[1]} in Y_mat and {C_mat.shape[1]} in C_mat.")
        
        # Check if the gene labels (index) are consistent between X_mat and Y_mat
        if not X_mat.index.equals(Y_mat.index):
            raise ValueError("Gene labels (index) must be identical between X_mat and Y_mat.")
        
        # Check if the cell type labels are consistent between X_mat and C_mat
        if not X_mat.columns.equals(C_mat.index):
            raise ValueError("Cell type labels must be identical between X_mat (columns) and C_mat (rows).")

        # Initialize training results to None
        self.gamma = None
        self.losses = None
            
        # Extract labels and amount of genes
        self.genes = X_mat.index
        self.p = len(self.genes)
        self.celltypes = X_mat.columns

        # Convert Inputs to Torch Tensors
        self.X, self.Y, self.C = DTD.XYCtoTorch(X_mat, Y_mat, C_mat)

    # Define static help functions for DTD algorithm
    @staticmethod
    def XYCtoTorch(X_mat, Y_mat, C_mat):
        X_torch = torch.Tensor(X_mat.to_numpy())
        Y_torch = torch.Tensor(Y_mat.to_numpy())
        C_torch = torch.Tensor(C_mat.to_numpy())
        return X_torch, Y_torch, C_torch

    @staticmethod
    def C_est(X, Y, gamma):
        Gamma = torch.diag(gamma)
        C_e = torch.linalg.inv(X.T @ Gamma @ X) @ X.T @ Gamma @ Y
        C_e[C_e < 0] = 0
        return (C_e)

    @staticmethod
    def DTD_loss(C, C_e):
        C_mean = torch.reshape(torch.mean(C, axis=1), (C.shape[0], 1))
        C_e_mean = torch.reshape(torch.mean(C_e, axis=1), (C.shape[0], 1))
        r_num = torch.sum((C - C_mean) * (C_e - C_e_mean), axis=1)
        r_den = torch.sqrt(torch.sum((C - C_mean) ** 2, axis=1) *
                           torch.sum((C_e - C_e_mean) ** 2, axis=1))
        r = r_num / r_den
        # total loss , individual loss
        return - torch.sum(r), r
    
    @staticmethod
    def Cosine_loss(C, C_e):
        loss = 0
        for i in range(C.shape[0]):
            r_num = C[i, :] @ C_e[i, :]
            r_den = torch.sqrt(torch.sum(C[i, :]**2)) * torch.sqrt(torch.sum(C_e[i, :]**2))
            loss += r_num / r_den

        C_mean = torch.reshape(torch.mean(C, axis=1), (C.shape[0], 1))
        C_e_mean = torch.reshape(torch.mean(C_e, axis=1), (C.shape[0], 1))
        r_num = torch.sum((C - C_mean) * (C_e - C_e_mean), axis=1)
        r_den = torch.sqrt(torch.sum((C - C_mean) ** 2, axis=1) *
                           torch.sum((C_e - C_e_mean) ** 2, axis=1))
        r = r_num / r_den
        return -loss, r

    # Define custom PyTorch model for gradient optimization
    class Model(nn.Module):

        def __init__(self, p):
            super().__init__()
            # Set seed for reproducibility
            torch.manual_seed(42)
            # initialization of weights with random numbers
            g = torch.distributions.Uniform(0.001, 0.1).sample((p,))
            # conversion to torch parameters
            self.weights = nn.Parameter(g)

        def forward(self, X, Y):
            g = self.weights
            return DTD.C_est(X, Y, g ** 2)

    def get_loss_function(self, func):
        pick = None
        if func == 'pearson':
            pick = self.DTD_loss
        if func == 'cosine':
            pick = self.Cosine_loss
        return pick

    # Define the training function
    def run(self, iterations=1000, plot=False, path_plot=None, func='pearson'):
        """
        Function that executes the training of a DTD model and saves the results in the model attributes.
        
        Parameters
        ----------
        iterations : int
            How many training steps should be conducted.
        plot : bool
            Whether to plot the development of loss-values during the training.
        
        
        Updates Attributes
        ------------------
        gamma: pd.DataFrame
            gene weights resulting form the training.
            Shape: genes x 1
        losses: list
            list of loss per training step.
        """

        # Initializing internal model and parameters
        m = self.Model(self.p)
        opt = torch.optim.Adam(m.parameters(), lr=0.001)
        losses = []
        mean_corr = []
        all_corr  = []

        # Training loop
        with tqdm(total=iterations) as pbar:
            for i in range(iterations):
                preds = m(self.X, self.Y)
                loss_func = self.get_loss_function(func)
                loss, all_r = loss_func(self.C, preds)
                #print(all_r)
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.detach().numpy())
                all_corr.append(all_r.detach().numpy())
                mean_corr.append(all_r.detach().numpy().mean())

                pbar.set_description('i = %i, loss = %1.6e' % (i + 1, losses[-1]))
                pbar.update(1)

        # Format returns
        gamma = m.weights.detach().numpy() ** 2
        gamma = len(gamma) * gamma / np.sum(gamma)
        gamma = pd.DataFrame((gamma), index=self.genes, columns=["gene weights"])

        # Plotting functionality
        all_corr = np.array(all_corr)
        if plot:
            plt.figure()
            #plt.axhline(y=1, alpha=0.3, c='gray')
            #plt.title("Training Progress Overview")
            plt.ylabel("corr($\hat{C}$,$C$)")
            plt.xlabel("Iterations")
            for i in range(self.X.shape[1]):
                plt.plot(all_corr[:,i],color = f'C{i}', label = self.celltypes[i], alpha=0.6)
            plt.plot(mean_corr, color="black", label="AVG", ls = "dotted", lw=3)

            plt.legend(loc='lower right', labelspacing = 0.2)
            if path_plot:
                plt.savefig(path_plot)
            plt.show()

        # Save results to model instance
        self.gamma = gamma
        self.mean_corr = mean_corr
        self.all_corr = all_corr
        self.losses = losses



#### ADTD ########################################################################################

class ADTD:
    """
    Class that implements the Adaptive Digital Tissue Deconvolution algorithm.
    
    Attributes
    ----------
    X_mat : pd.DataFrame
        Single cell reference matrix X containing (average) profiles as columns per celltype. 
        Shape: genes x celltypes
    Y_mat : pd.DataFrame
        Y matrix containing bulk profiles.
        Shape: genes x n_mixtures
    gamma : pd.DataFrame
        gene weights.
        Shape: genes x 1
    max_iterations : int
        Maximum amount of optimization steps.
    lambda1 : float
        Hyperparameter for cellular composition estimation.
    lambda2 : float
        Hyperparameter for reference profile adaption (gene regulation).
    eps : float
        Stopping criterion based on error.
    C_static : bool
        Whether cellular composition shall be optimized.
    Delta_static : bool
        Whether reference profile adaption shall be optimized.
    """

    def __init__(self,
                 X_mat: pd.DataFrame,
                 Y_mat: pd.DataFrame,
                 gamma: pd.DataFrame,
                 lambda1: float = 1.0,
                 lambda2: float = 1.0,
                 max_iterations: int = 200,
                 eps: float = 1e-8,
                 C_static: bool = False,
                 Delta_static: bool = False,
                 gamma_offset: bool = True,
                 delta_stepsize: int = 1):
        
        # Input Handling
        # Check if the number of genes (rows) matches between X_mat and Y_mat
        if X_mat.shape[0] != Y_mat.shape[0]:
            raise ValueError(f"Number of genes (rows) must match between X_mat and Y_mat. "
                            f"Found {X_mat.shape[0]} in X_mat and {Y_mat.shape[0]} in Y_mat.")

        # Check if the number of genes (rows) matches between X_mat and gamma
        if X_mat.shape[0] != gamma.shape[0]:
            raise ValueError(f"Number of genes (rows) must match between X_mat and gamma. "
                            f"Found {X_mat.shape[0]} in X_mat and {gamma.shape[0]} in gamma.")

        # Check if the gene labels (index) are consistent between X_mat and Y_mat
        if not X_mat.index.equals(Y_mat.index):
            raise ValueError("Gene labels (index) must be identical between X_mat and Y_mat.")

        # Check if the gene labels (index) are consistent between X_mat and gamma
        if not X_mat.index.equals(gamma.index):
            raise ValueError("Gene labels (index) must be identical between X_mat and gamma.")
        
        # Initialize results to None
        self.C_est = None
        self.c_est = None
        self.x_est = None
        self.Delta_est = None
        
        # Extract labels
        self.celltypes = X_mat.columns
        self.genes = X_mat.index

        # Convert Inputs to Numpy Arrays
        self.X = X_mat.to_numpy()
        self.Y = Y_mat.to_numpy()
        if gamma_offset:
            self.gamma = gamma.to_numpy().flatten() + 1. / self.X.shape[0]  # Offset to counter genes weighed 0
        else:
            self.gamma = gamma.to_numpy().flatten()


        # Normalize expression profiles
        self.X = self.X / np.sum(self.X, 0)
        self.Y = self.Y / np.sum(self.Y, 0)

        self.p, self.q, self.n = self.X.shape[0], self.X.shape[1], self.Y.shape[1]

        self.Gamma = np.diag(self.gamma)
        self.G = np.diag(np.sqrt(self.gamma))

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iterations = max_iterations
        self.eps = eps
        self.C_static = C_static
        self.Delta_static = Delta_static
        self.delta_stepsize = delta_stepsize

        self.Delta = np.ones((self.p, self.q))
        self.C0_init = None
        self.x = None
        self.C = np.zeros((self.q, self.n))  # initialize C
        self.c = np.random.rand(1, self.n)  # initialize random c (might be a better way)


    def setup(self):
        """
        Still needed due to reasonable estimate for x.
        Random initialization may be possible.
        """
        reg_nnls = LinearRegression(fit_intercept=False, positive=True)
        self.C0_init = reg_nnls.fit(self.G @ self.X, self.G @ self.Y).coef_.T
        x_base = np.mean(self.Y - self.X @ self.C0_init, 1).reshape((self.p, 1))
        x_base = np.maximum(x_base, 0)

        self.x = x_base / np.sum(x_base, 0)


    def update_C0(self):
        """
        Calculate C0 with with lambda1=0 and lambda2->inf. 
        Reasoning that lambda1->inf yields DTD solution no longer holds.
        """
        A = self.G @ self.Y  # (p x n) - matrix
        B = np.c_[(self.G @ self.X), (self.G @ self.x)]

        P = 2.0 * B.T @ B  # Positiv definit (q+u x q+u) matrix
        Q = -2.0 * B.T @ A  # (q+u x n) - matrix

        #equality constraints
        A_eq = np.c_[np.ones((1, self.q)), 1.]  # (1 x q+u) - matrix
        b = np.array([1.])

        #inequality constraints
        Gmat = -np.identity(self.q + 1)
        h = np.zeros((self.q + 1, 1))

        C_tilde = np.r_[self.C, self.c]


        for i in range(self.n):
            sol = solve_qp(P, Q[:, i], Gmat, h, A_eq, b, solver="quadprog")
            C_tilde[:, i] = sol

        self.C0_init = C_tilde[:self.q, :]
        # get first prediction for c as by product
        self.c = C_tilde[self.q:, :]

        self.C0_init[self.C0_init < 0] = 0
        self.c[self.c < 0] = 0


    def update_C(self):
        if self.C_static:
            self.C = self.C0_init
            self.c = np.ones((1, self.n)) - np.sum(self.C, 0)
            self.c[self.c < 0] = 0
            return None

        a1 = self.G @ self.Y  # (p x n) - matrix
        a2 = np.sqrt(self.lambda1) * self.C0_init  # (p x n) - matrix
        A = np.r_[a1, a2]  # (p+q x n) - matrix

        b1 = np.c_[(self.G @ (self.Delta * self.X)), (self.G @ self.x)]
        b2 = np.c_[np.sqrt(self.lambda1) * np.identity(self.q), np.zeros((self.q, 1))]  # (q x q+u) - matrix
        B = np.r_[b1, b2]  # (p+q x q+u) - matrix

        P = 2.0 * B.T @ B  # Positiv definit (q+u x q+u) matrix
        Q = -2.0 * B.T @ A  # (q+u x n) - matrix

        #equality constraints
        A_eq = np.c_[np.ones((1, self.q)), 1.]  # (1 x q+u) - matrix
        b = np.array([1.])

        #inequality constraints
        Gmat = -np.identity(self.q + 1)
        h = np.zeros((self.q + 1, 1))

        C_tilde = np.r_[self.C, self.c]

        for i in range(self.n):
            sol = solve_qp(P, Q[:, i], Gmat, h, A_eq, b, solver="quadprog")
            C_tilde[:, i] = sol

        self.C = C_tilde[:self.q, :]
        self.c = C_tilde[self.q:, :]

        self.C[self.C < 0] = 0
        self.c[self.c < 0] = 0
        #print(np.sum(self.C, axis=0), self.c)

    def update_Delta(self):
        if self.Delta_static:
            return None
            
        # Calculate statics
        Y_tilde = self.G @ (self.Y - self.x @ self.c)
        CCT = self.C @ self.C.T

        all_diags = []
        for i in range(self.q):
            row_diags = []
            for j in range(self.q):
                diag_value = CCT[i, j] # Extract diagonal element
                block = diag_value * self.Gamma # Create the diagonal matrix for this element
                row_diags.append(sp.csr_matrix(block))
            all_diags.append(row_diags)
        # Construct the block diagonal structure
        a1 = sp.bmat(all_diags, format='csr')

        a2 = sp.lil_matrix((self.p * self.q, self.p * self.q))
        for i in range(0, self.p * self.q, self.p):
            for j in range(i, self.p * self.q, self.p):
                a2[i:i + self.p, j:j + self.p] = sp.diags(self.X[:, i // self.p] * self.X[:, j // self.p])
                if i != j:
                    # If i != j, fill the symmetric block as well
                    a2[j:j + self.p, i:i + self.p] = a2[i:i + self.p, j:j + self.p]

        a2 = a2.tocsr()

        # Element-wise multiplication of a1 and a2
        a12 = a1.multiply(a2)
        
        # Create sparse identity matrix for a3
        a3 = self.lambda2 * sp.eye(self.p * self.q, format='csr')
        
        # Calculate P
        P = 2.0 * (a12 + a3)
        P = P.tocsc()  # Convert to CSC for solver compatibility
        
        # Calculate b1 
        kron_eye = sp.kron(sp.eye(self.q), self.G)
        X_flat = self.X.flatten('F')
        Y_tilde_C = self.C @ Y_tilde.T
        b1 = (Y_tilde_C.flatten() @ kron_eye @ sp.diags(X_flat))
        
        # Calculate Q
        Q = -2.0 * (b1 + self.lambda2 * np.ones(self.p * self.q))

        # Inequality constraints
        Gmat = -sp.eye(self.p * self.q, format='csc')
        h = np.zeros(self.p * self.q)
        
        # Equality constraints
        Amat = sp.lil_matrix((self.q, self.q * self.p))
        for i in range(self.q):
            Amat[i, i*self.p:(i+1)*self.p] = self.X[:, i]
        Amat = Amat.tocsc()
        b = np.ones(self.q)

        # Solve quadratic program
        sol = solve_qp(P, Q, Gmat, h, Amat, b, solver="clarabel")

        # Failsafe if clarabel yields None as solution
        if np.any(self.Delta) is None:
            sol = solve_qp(P, Q, Gmat, h, Amat, b, solver="scs")

        self.Delta = sol.reshape((self.p, self.q), order='F')


    def update_x(self):
        if np.all(self.c == 0):
            return np.ones((self.p, 1)) / self.p

        Z = self.Y - (self.Delta * self.X) @ self.C
        P = 2. * self.Gamma * np.sum(self.c ** 2)
        P = sp.csc_matrix(P)
        q = - 2. * (self.c @ Z.T @ self.Gamma).flatten()
        G = - np.identity(self.p)
        G = sp.csc_matrix(G)
        h = np.zeros(self.p)
        A = np.ones(self.p)
        A = sp.csc_matrix(A)
        b = np.array([1.])

        sol = solve_qp(P, q, G, h, A, b, solver="scs")
        self.x = sol.reshape((self.p, 1))
        self.x[self.x < 0] = 0


    def Loss_ADTD(self):
        ltemp = self.G @ (self.Y - (self.Delta * self.X) @ self.C - (self.x @ self.c))
        term1 = np.sum(ltemp ** 2)
        term2 = self.lambda1 * np.sum((self.C - self.C0_init) ** 2)
        term3 = self.lambda2 * np.sum((np.ones((self.p, self.q)) - self.Delta) ** 2)

        return term1 + term2 + term3, term1, term2 + term3
    
    def Loss_static(self):
        return np.sum((self.G @ (self.Y - self.X @ self.C - self.x @ self.c))**2)

    def run(self, verbose=True):
        """
        Fit an ADTD model

        Attributes
        ----------
        verbose : bool
            Wether progress bar should be shown during training (default) or not (silent mode).
        
        Updates Attributes
        ------------------
        C_est : pd.DataFrame
            Estimated contribution of the referenced cell types in X to the composition of mixtures in Y.
            Shape: referenced cell types x mixtures
        c_est : pd.DataFrame
            Estimated contribution of the hidden background to the composition of mixtures in Y.
            Shape: 1 x mixtures
        x_est : pd.DataFrame
            Estimated consensus hidden background profile.
            Shape: genes x 1
        Delta_est : pd.DataFrame
            Estimated element-wise adaption factors for the reference matrix X (gene regulation).
            Shape: genes x referenced cell types (same shape as X)
        """
        self.setup()
        self.update_C0()
        #if self.verbose:
        #    print(
        #        "start loss = %1.6e, start loss RSS = %1.6e, start loss bias = %1.6e"
        #        % self.Loss_ADTD()[0:]
        #    )

        loss = []
        
        # Case 1: C_static = True
        if self.C_static:
            with tqdm(total=self.max_iterations, disable = not verbose) as pbar:
                
                # Outer Optimization Loop
                # Set Error for first iteration
                loss_old = np.inf
                for i in range(self.max_iterations):

                    # Convergence Check and pbar
                    ltemp = self.Loss_ADTD()
                    loss_new = ltemp[0]
                    loss.append(loss_new)
                    #print(loss_old, loss_new)
                    err = loss_old - loss_new
                    if err <= self.eps:
                        pbar.set_description('i = %i, err = %1.2e - Convergence reached!' % (i, err))
                        break
                    else:
                        pbar.set_description('i = %i, err = %1.2e, loss = %1.6e, loss RSS = %1.6e, loss bias = %1.6e' % (i+1, err, ltemp[0], ltemp[1], ltemp[2]))
                        pbar.update(1)


                    # Initialization Loop (only on first iteration)
                    if i == 0:
                        # Set Static Error >> eps for first iteration of inner loop
                        old_static_loss = np.inf
                        new_static_loss = 0
                        for j in range(1000):
                            # Calculate Error
                            static_err = old_static_loss - new_static_loss
                            # If convergence reached, update Delta, else optimize further
                            if static_err <= self.eps:
                                self.update_C
                                self.update_x()
                                self.update_Delta()
                                loss_old = loss_new
                                break
                            else:
                                old_static_loss = self.Loss_static()
                                self.update_C0
                                self.C = self.C0_init
                                self.update_x()
                                new_static_loss = self.Loss_static()
                    else:
                        loss_old = loss_new
                        self.update_C()
                        self.update_x()
                        self.update_Delta()

        # Case 2: C_static = False
        else:
            with tqdm(total=self.max_iterations, disable = not verbose) as pbar:
                for i in range(self.max_iterations):
                    # Save old
                    C_copy = self.C.copy()
                    c_copy = self.c.copy()
                    x_copy = self.x.copy()
                    Delta_copy = self.Delta.copy()

                    loss_old = self.Loss_ADTD()[0]
                    loss.append(loss_old)

                    # Update all optimization variables
                    self.update_C()
                    self.update_x()
                    self.update_Delta()

                    # Calculate loss and new error
                    loss_new = self.Loss_ADTD()[0]
                    err = loss_old - loss_new

                    # Update pbar
                    ltemp = self.Loss_ADTD()
                    pbar.set_description('i= %i, err = %1.2e, loss = %1.6e, loss RSS = %1.6e, loss bias = %1.6e' %
                                        (i + 1, err, ltemp[0], ltemp[1], ltemp[2]))
                    pbar.update(1)

                    # Convergence criterion
                    if i > 1 and err < self.eps:
                        if err < 0:  #avoids wrong parameters if min is overshot
                            self.C = C_copy
                            self.c = c_copy
                            self.x = x_copy
                            self.Delta = Delta_copy           
                        pbar.set_description("i= %i, err = %1.2e, Convergence Reached!" % (i, err))
                        #pbar.n = pbar.total  # Move progress to 100%
                        pbar.update(0)  # Force refresh
                        break

        # Format returns properly
        C_est = pd.DataFrame(self.C, index=self.celltypes)
        c_est = pd.DataFrame(self.c, index=["hidden"])
        x_est = pd.DataFrame(self.x, index=self.genes)
        Delta_est = pd.DataFrame(self.Delta, index=self.genes, columns=self.celltypes)

        # Save results to instance
        self.C_est = C_est
        self.c_est = c_est
        self.x_est = x_est
        self.Delta_est = Delta_est