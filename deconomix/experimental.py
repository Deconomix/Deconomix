# Imports
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from deconomix.methods import ADTD
import multiprocessing as mp
from tqdm import tqdm
import datetime

class HPS2:
    def __init__(self,
                 X_ref : pd.DataFrame,
                 Y_test : pd.DataFrame,
                 gamma : pd.DataFrame,
                 k_folds: int = 5,
                 lambdas: Iterable = np.logspace(-10, 0, num=11)):

        # Initialize Attributes
        self.X_df = X_ref
        self.X_mat = X_ref.values
        self.Y_df = Y_test
        self.Y_mat = Y_test.values
        self.gamma = gamma
        self.G_mat = np.diag(np.sqrt(gamma.values.flatten()))
        self.k_folds = k_folds
        self.lambdas = lambdas
        self.validation_losses = pd.DataFrame(
            columns=["fold", "lambda", "loss"]
        )
        self.results = None

        # Force numerical sample ids
        self.Y_df.columns = range(self.Y_df.shape[1])

        # Prepare list of jobs (one for every (fold, lambda) combination)
        print("Preparing Job List")
        self.jobs = []
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold_num, (train_index, test_index) in enumerate(kf.split(self.Y_df)):
            for lmbda in lambdas:
                job = {
                    "fold": fold_num,
                    "train_ids": train_index,
                    "test_ids": test_index,
                    "lambda": lmbda
                }
                self.jobs.append(job)

        # Define a baseline model with C static and Delta static for a consensus C_est, c_est and x_est on all samples
        # This way we can evaluate Delta independently of the other estimates
        print("Preparing Baseline Model")
        model_baseline = ADTD(self.X_df, self.Y_df, self.gamma, C_static=True, Delta_static=True, max_iterations=1000)
        model_baseline.run()
        self.C_df_baseline = model_baseline.C_est
        # x_est and c_est are not affected by Delta, therefore omitted in later calculations (relative loss comparisons)
        

        
    def _generalization_loss(self, Delta, test_ids):
        Y_test_mat = self.Y_df.loc[:,test_ids].values
        C_test_mat = self.C_df_baseline.loc[:,test_ids].values
        recon_error = Y_test_mat - ((Delta.values * self.X_mat) @ C_test_mat)
        #print(recon_error.shape)
        #print(self.G_mat.shape)
        loss_raw = np.linalg.norm(recon_error, 'fro')**2
        loss_weighted = np.linalg.norm(self.G_mat @ recon_error, 'fro')**2
        return loss_raw, loss_weighted

        
                
    def _run_job(self, job):
        """
        Runs a single cross-validation job given by the job dict.
        The job dict should contain:
            - 'fold': Fold number
            - 'train_ids': Indices for training samples
            - 'test_ids': Indices for test samples
            - 'lambda': The regularization parameter

        Returns:
            A dict or tuple with results (e.g., fold, lambda, loss)
        """
        # Placeholder: The user should implement the logic, e.g.:
        # 1. Split train/test sets from self.X, self.Y using job['train_ids'] and job['test_ids']
        # 2. Fit a (ADTD) model on the training part using job['lambda']
        # 3. Evaluate loss on test set
        #
        # Example structure (replace with actual computations):
        fold = job['fold']
        lmbda = job['lambda']
        train_ids = job['train_ids']
        test_ids = job['test_ids']

        #print(f"Starting job: fold={fold}, lambda={lmbda}")
        # Train a model on the training data
        model = ADTD(self.X_df,
                     self.Y_df.loc[:,train_ids],
                     self.gamma,
                     max_iterations=1000,
                     C_static=True,
                     Delta_static=False,
                     lambda2=lmbda)
        model.run(verbose=False)

        # Evaluate Generalization Performance

        loss_raw, loss_weighted  = self._generalization_loss(model.Delta_est, test_ids)
        #print(f"Finished job: fold={fold}, lambda={lmbda}, loss={loss}")
        return {"fold": fold, "lambda": lmbda, "loss_raw": loss_raw, "loss_weighted": loss_weighted}


    def run(self, n_workers=10):
        # Assume self.jobs is a list of job dicts to process.
        # If not defined, user is expected to set up jobs before calling run().

        jobs = getattr(self, "jobs", None)
        if jobs is None:
            raise ValueError("No jobs found. Please assign a list of jobs to self.jobs before running.")

        # tqdm does not work well with mp.Pool.imap if the function is from a class due to pickling/tracking issues.
        # Fallback: collect results using imap_unordered and manual progress reporting.

        results = []
        with mp.Pool(processes=n_workers) as pool:
            for result in tqdm(pool.imap_unordered(self._run_job, jobs), total=len(jobs), desc="Running jobs"):
                results.append(result)

        # Store or return results as appropriate (here we set them as an attribute)
        self.results = results

        # Write results to a log file with a timestamp
        log_filename = f"deconomix_run_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            with open(log_filename, "w") as logfile:
                logfile.write(f"Run timestamp: {datetime.datetime.now().isoformat()}\n")
                logfile.write("Results:\n")
                for res in results:
                    logfile.write(str(res) + "\n")
        except Exception as e:
            print(f"Could not write log file: {e}")



