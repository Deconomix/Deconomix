# Importing Dependencies
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import DeconomiX
from tqdm import tqdm

# Loading Dataset and training a DTD Model
print("\nImporting Data:")
test, train = DeconomiX.utils.load_example()
X_ref, Y_mat, C_mat = DeconomiX.utils.simulate_data(train.drop("B", axis=1), n_mixtures = 1000, n_cells_in_mix = 100)
X_test , Y_test, C_test  = DeconomiX.utils.simulate_data(test, n_mixtures = 500, n_cells_in_mix = 100)
model_DTD = DeconomiX.methods.DTD(X_ref, Y_mat, C_mat)
print("\nTraining DTD Model:")
model_DTD.run(iterations = 500, plot = False)

# Conduct Gridsearch
print("\nGridsearch:")
grid = np.logspace(-10, 5, num=16)

losses = np.full((len(grid), len(grid)), np.nan)
C_est_mean = np.full((len(grid), len(grid)), np.nan)
c_est = np.full((len(grid), len(grid)), np.nan)

with tqdm(total=len(grid)**2) as pbar:
    i = 0
    j = 0
    for l1 in grid:
        for l2 in grid:
            pbar.set_description('lambda1=%1.2e, lambda2=%1.2e' % (l1,l2)) 
            #print(l1,l2)
            curr_model = DeconomiX.methods.ADTD(X_ref, Y_test, model_DTD.gamma,
                                       C_static=False, lambda1 = l1,
                                       Delta_static=False, lambda2 = l2,
                                       max_iterations=1000)
            curr_model.run(verbose=False)
            loss, loss_RSS, loss_Bias = curr_model.Loss_ADTD()
            corr = DeconomiX.utils.calculate_corr(C_test, curr_model.C_est,
                                        hidden_ct="B", c_est=curr_model.c_est)
            #print(loss, loss_RSS, loss_Bias)
            #print(corr)
            losses[i,j] = loss
            C_est_mean[i,j] = corr[corr.index != 'hidden'].mean()
            c_est[i,j] = corr['hidden']
            pbar.update(1)
            j +=1
            if j == len(grid):
                j = 0
        i += 1

# Save results
np.save("gridsearch_C.npy", C_est_mean)
np.save("gridsearch_c.npy", c_est)
np.save("gridsearch_loss.npy", losses)
