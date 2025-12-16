# %% ####################################################################################
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip
from tqdm import tqdm
import gzip
import shutil
from scipy.stats import spearmanr
from scipy.stats import fisher_exact
import warnings
import seaborn as sns
from scipy.stats import mannwhitneyu


# %% ####################################################################################
def simulate_data(
    scRNA_df : pd.DataFrame,
    n_mixtures : int,
    n_cells_in_mix : int,
    n_genes : int = None,
    seed : int = 1,
    coverage_threshold : int = 5):

    """
    Function that generates artificial bulk mixtures Y from a collection of single cell profiles and saves the ground truth of their composition C.
    Furthermore, it derives a reference matrix X by averaging over provided example profiles per cell type.

    Parameters
    ----------
    scRNA_df : pd.DataFrame
        A matrix containing labeled single cell RNA profiles from multiple cell types.
        Please provide multiple examples per cell type and label each sample with a column label stating the cell type.
        Shape: genes x n_samples
    n_mixtures : int
        How many artificial bulks should be created.
    n_cells_in_mix : int
        How many profiles from scRNA_df should be randomly mixed together in one artifical bulk.
    n_genes : int
        Filters genes from scRNA_df by variance across celltypes and restricts the gene space to the n_genes most variable genes.
    seed : int
        Random seed for the drawing process.
    coverage_threshold : int
        Minimum number of examples per cell type, if number of examples is below the threshold a warning will be thrown. (Default = 5).

    Returns
    -------
    X_ref : pd.DataFrame
        Single cell reference matrix X calculated by averaging over all examples of each cell type in scRNA_df.
        Shape: genes x celltypes
    Y_mat : pd.DataFrame
        Y matrix containing the generated artificial bulk profiles. Each column represents one artificial bulk, normalized by n_cells_in_mix
        Shape: genes x n_mixtures
    C_mat : pd.DataFrame
        Composition matrix containing the true cellular abundance in percent for each artificial mixture according to the drawing process.
        Shape: cell types x n_mixtures
    """

    # Input Handling
    # Check if n_genes is plausible
    if n_genes:
        if scRNA_df.shape[0] < n_genes:
            raise ValueError(f"Number of genes to select needs to be equal or lower than the amount of genes provided in scRNA_df. "
                             f"Found {scRNA_df.shape[0]} genes in scRNA_df but n_genes is {n_genes}")
    
    # Coverage check
    ct_unique = set(scRNA_df.columns)
    n_celltypes = len(ct_unique)
    n_examples = len(scRNA_df.columns)
    
    # Issue warnings for low coverage and check for single-example cell types
    celltype_counts = scRNA_df.columns.value_counts()
    for celltype, count in celltype_counts.items():
        if count == 1:
            warnings.warn(f"Warning: Only one example provided for cell type '{celltype}'. The mean will be the same as the single example.")
        elif count <= coverage_threshold:  # Assuming coverage_threshold is defined
            warnings.warn(f"Warning: Low coverage for cell type '{celltype}' ({count} examples).")

    # Gene filter function
    def gene_filter(scRNA_df, n_genes=1000):
        # Calculate average expression per cell type
        X_ref = scRNA_df.T.groupby(level=0, observed=False).mean().T

        # Calculate variance across cell types for each gene
        gene_variances = X_ref.var(axis=1)

        # Select the top n_genes based on variance
        selected_genes = gene_variances.nlargest(n_genes).index

        return selected_genes

    # Filter genes by variance across cell types if n_genes is specified
    if n_genes:
        gene_selection = gene_filter(scRNA_df, n_genes)
        scRNA_df = scRNA_df.loc[gene_selection, :]

    
    # Creating X_ref as average profiles of the unique cell types
    X_ref = scRNA_df.T.groupby(level=0, observed=False).mean().T

    # Generate random samples all at once via random index choice
    np.random.seed(seed)
    bulk_indices = np.random.choice(scRNA_df.columns.size, size=(n_mixtures, n_cells_in_mix), replace=True)

    # Create Y_mat and C_mat without a for loop
    Y_mat = pd.DataFrame(
        np.add.reduceat(scRNA_df.values[:, bulk_indices.flatten()], np.arange(0, n_cells_in_mix * n_mixtures, n_cells_in_mix), axis=1) / n_cells_in_mix,
        index=scRNA_df.index
    )

    # Create C_mat by counting occurrences of each cell type in the sampled indices
    C_mat = pd.DataFrame(0, index=scRNA_df.columns.unique(), columns=range(n_mixtures))

    for i in range(n_mixtures):
        sampled_celltypes = scRNA_df.columns[bulk_indices[i, :]]
        celltype_counts = sampled_celltypes.value_counts()
        C_mat.loc[celltype_counts.index, i] = celltype_counts.values

    C_mat = C_mat.div(n_cells_in_mix)
    C_mat = C_mat.reindex(X_ref.columns)


    return X_ref, Y_mat, C_mat


# %% ####################################################################################
def calculate_estimated_composition(
    X_mat : pd.DataFrame,
    Y_mat : pd.DataFrame,
    gamma : pd.DataFrame):
    r"""
    Function that calculates the estimated composition of Y based on a
    reference matrix X and the gene weight vector gamma generated by the DTD training.

    Parameters
    ----------
    X_mat : pd.DataFrame
        Single cell reference matrix X containing (average) profiles as columns per celltype. 
        Shape: genes x celltypes
    Y_mat : pd.DataFrame
        Y matrix containing the generated artificial bulk profiles.
        Shape: genes x n_mixtures
    gamma : pd.DataFrame
        DataFrame containing the weights for the genes featured in X_mat and Y_mat.
        Usually generated by Loss-function Learning for Digital Tissue Deconvolution.
        Shape: genes x 1
    
    Returns
    -------
    C_estimated : pd.DataFrame
        Matrix that contains estimated cellular abundances for the bulk profiles of the Y matrix.

    """

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
    
    # Calculations

    X = torch.from_numpy(X_mat.values.astype(np.double))
    Y = torch.from_numpy(Y_mat.values.astype(np.double))
    gamma = torch.from_numpy(gamma.values.flatten().astype(np.double))
    Gamma = torch.diag(gamma)
    C_e  = torch.linalg.inv(X.T @ Gamma @ X) @ X.T @ Gamma @ Y
    C_e[C_e < 0] = 0
    C_estimated = pd.DataFrame(C_e, index = X_mat.columns, columns = Y_mat.columns)
    return(C_estimated)



# %% ####################################################################################
def XYCtoTorch(X_mat, Y_mat, C_mat):
    X_torch = torch.Tensor(X_mat.to_numpy())
    Y_torch = torch.Tensor(Y_mat.to_numpy())
    C_torch = torch.Tensor(C_mat.to_numpy())
    return X_torch, Y_torch, C_torch


# %% ####################################################################################
def load_example():
    r"""
    Function providing an example test and training set for a DeconomiX workflow.
    Data is downloaded from NCBI (Tirosh et al., https://doi.org/10.1126/science.aad0501) once and cached for further usage.

    Returns
    -------
    tir_test : pd.DataFrame
        Matrix that contains multiple single cell RNA profiles for different cell types from a set of tumors.
        Column labels equals cell type. Shape: genes x n_samples
    tir_train : pd.DataFrame
        Matrix that contains multiple single cell RNA profiles for different cell types from a different set of tumors.
        Column labels equals cell type. Shape: genes x n_samples
    """
    

    # Internal functions:

    def download_and_extract_gz(url, extract_path):
        #creating data folder
        if not os.path.exists("example/data"):
            os.makedirs("example/data")
        # Download the file if it doesn't exist
        if not os.path.exists(extract_path):
            print(f"Downloading Tirosh data from {url}...")
            with urllib.request.urlopen(url) as response, open("example/data/downloaded_file.gz", 'wb') as out_file:
                content_length = int(response.headers.get('Content-Length'))
                with tqdm(total=content_length, unit='B', unit_scale=True) as pbar:
                    while True:
                        data = response.read(1024)
                        if not data:
                            break
                        out_file.write(data)
                        pbar.update(len(data))
        

            # Extract the file
            print("Extracting file...")
            with gzip.open("example/data/downloaded_file.gz", "rb") as f_in, open(extract_path, "wb") as f_out:
                f_out.write(f_in.read())

            # Remove the downloaded file
            os.remove("example/data/downloaded_file.gz")

            print(f"File extracted to {extract_path}")

        else:
            print("Raw data already downloaded")

    def importing_tirosh():
        # reading data
        #if os.path.exists("example/data/train.feather"):
        #    print("Loading already existing preprocessed data...")
        #    tir_train = pd.read_feather("example/data/train.feather")
        #    tir_test = pd.read_feather("example/data/test.feather")
        #    return tir_train, tir_test
        
        #else:

            tirosh = pd.read_csv("example/data/tirosh_raw.txt", sep="\t", index_col=0)
            #Drop weird gene 'MARCH1' which is featured TWICE in the dataset?!
            tirosh = tirosh.drop('MARCH1')
            # list of cell types to filter
            cell_types = [0,1,2,3,4,5,6]
            celltype_dict = {0: 'Malignant', 1: 'T', 2: 'B', 3: 'Macro', 4: 'Endo', 5: 'CAF', 6: 'NK'}

            # Transformation to tpm
            #Formula: 10*(2^Ei,j-1)
            def backtransformation(x):
                return 10*(2**x - 1)
            
            tirosh[3:].map(backtransformation)

            # Restricting gene space (OLD)
            #temp_var = tirosh.iloc[:,:].var(axis = 1)
            #temp_var[0:3] = None
            #tirosh['Variance'] = temp_var
            #tirosh.sort_values('Variance', ascending = False, na_position = 'first', inplace = True)
            #tirosh.drop('Variance', axis = 1, inplace = True)
            #tirosh = tirosh[0:1003]

            ##### Restricting gene space (NEW) ####
            # Step 1: Create Reference Matrix from whole dataset
            scRNA_df = tirosh.iloc[3:,:]
            scRNA_df.index = tirosh.index[3:]
            scRNA_df.columns = tirosh.iloc[2,:]
            #display(scRNA_df)
            ct_unique = set(scRNA_df.columns)
            X_ref = pd.DataFrame()
            for celltype in ct_unique:
                curr_avg = scRNA_df.loc[:, celltype].mean(axis=1)
                X_ref[celltype] = curr_avg
            

            # Step 2: Calculate Variance over average expression levels of genes between different celltypes
            temp_var = X_ref.iloc[:,1:].var(axis=1) # Ignore malignant cells here?!
            X_ref['Variance'] = temp_var
            X_ref.sort_values('Variance', ascending = False, na_position = 'first', inplace = True)
            #display(X_ref)
            genes_top1k = list(X_ref.index[0:1000])
            #print(len(genes_top1k))

            # Step 3: Choose the calculated top 1k genes in the full dataset
            head_index = list(tirosh.index[0:3])
            restricted_selection = head_index + genes_top1k
            tirosh = tirosh.loc[restricted_selection]
            #display(tirosh)


            
            #### Splitting Data by Tumor Ids #####
            tumor_ids = np.unique(tirosh.iloc[0].values)
            i_split = int(len(tumor_ids)/2)
            train_ids = tumor_ids[:i_split]
            test_ids = tumor_ids[i_split:]

            tir_train = tirosh.loc[:, tirosh.iloc[0].isin(train_ids)]
            tir_test  = tirosh.loc[:, tirosh.iloc[0].isin(test_ids)]

            def choose_celltypes(tir_raw, cell_types):
                # Chooses celltypes and drops non gene expression rows
                tir_filtered = tir_raw.loc[:, tir_raw.iloc[2].isin(cell_types)]
                tir_filtered = tir_filtered.iloc[2:]
                tir_filtered.columns = tir_filtered.iloc[0]
                tir_filtered = tir_filtered[1:]
                tir_filtered = tir_filtered.rename(columns=celltype_dict)
                tir_filtered.index.name = "Genes"
                tir_filtered.columns.name = "Celltypes"
                return tir_filtered

            # filtering the sets
            tir_train = choose_celltypes(tir_train, cell_types)
            tir_test  = choose_celltypes(tir_test, cell_types)

            return tir_train, tir_test
    

    # Actual Import
    # Tirosh URL
    url = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE72056' \
        '&format=file&file=GSE72056%5Fmelanoma%5Fsingle%5Fcell%5Frevised%5Fv2%2Etxt%2Egz'
    extract_path = os.path.join(os.getcwd(), "example/data/tirosh_raw.txt")
    download_and_extract_gz(url, extract_path)
    print("Importing Data to Python ...")

    if os.path.exists("example/data/train.pkl"):
        # Found preprocessed cached data, skip preprocessing and load from csv instead:
        print("Using cached, preprocessed data")
        tir_train = pd.read_pickle("example/data/train.pkl")
        tir_test = pd.read_pickle("example/data/test.pkl")
    else:
        tirosh = pd.read_csv("example/data/tirosh_raw.txt", sep="\t", index_col=0)
        print("Splitting and filtering data ...")
        tir_train, tir_test = importing_tirosh()
        # Drop malignant samples
        tir_train = tir_train.drop("Malignant", axis=1)
        tir_test = tir_test.drop("Malignant", axis=1)
        # save the preprocessed data
        tir_train.to_pickle("example/data/train.pkl")
        tir_test.to_pickle("example/data/test.pkl")
    print("Done")
    #drop last row (1001 instead of 1000 genes...)
    #tir_train.drop(tir_train.tail(1).index,inplace=True)
    #tir_test.drop(tir_test.tail(1).index,inplace=True)
    return tir_train, tir_test


def plot_corr(C_true, C_est, title="", color = "grey", hidden_ct=None, c_est=None, path=None):
    """
    Function that calculates and visualizes the correlation of estimated composition for multiple bulks with the ground truth.
    
    Parameters
    ----------
    C_true : pd.DataFrame
        True cellular compositions for all mixtures analyzed in C_est.
        Features all cell types, even if one cell type is hidden.
        Shape: n_cell_types x n_mixtures
    C_est : pd.DataFrame
        Estimated cellular compositions for multiple bulk mixtures.
        Features all non-hidden cell types from the ground truth.
        Shape: b_cell_types x n_mixtures
    color : String
        Sets the color of the correlation plots for the non-hidden cell types.
    hidden_ct : String (Optional)
        Needs to be set to the label of the cell type in C_true that is hidden, which means its not featured in C_est.
    c_est : pd.DataFrame (Optional)
        Estimated hidden background contributions for all mixtures featured in C_est (ADTD).
        Shape: 1 x n_mixtures
    path : String (Optional)
        Specify a path for saving the figure as pdf.
    

    Returns
    -------
    correlations : pd.DataFrame
        Spearman correlation values for C_true vs. C_est for each cell type.
        Shape: n_cell_types
    """

    # Inititialize pd.Series for correlation values
    correlations = pd.Series()

    # Calculate bounds & plots
    rows = int(np.ceil(len(C_true.index)/3))
    empty = rows*3 - len(C_true.index)
    i = 0
    j = 0
    fig, axes = plt.subplots(rows,3, figsize=(15,rows*5), squeeze=False)
    fig.suptitle(title)
    for celltype in C_est.index:
        r_val, _ = spearmanr(C_true.loc[celltype], C_est.loc[celltype])
        correlations[celltype] = r_val
        axes[j,i].set_title(celltype + " (R: %1.2f)" % (r_val))
        axes[j,i].scatter(C_true.loc[celltype], C_est.loc[celltype], alpha = 0.6, c=color)
        axes[j,i].set_xlabel("true abundance")
        axes[j,i].set_ylabel("estimated abundance")
        if i == 2:
            j += 1
            i = 0
        else:
            i +=1
    if hidden_ct:
        r_val, _ = spearmanr(C_true.loc[hidden_ct], c_est.loc["hidden"])
        correlations['hidden'] = r_val
        axes[j,i].set_title("Hidden (R: %1.2f)" % (r_val))
        axes[j,i].scatter(C_true.loc[hidden_ct], c_est.loc["hidden"], alpha = 0.6, c="black")
        axes[j,i].set_xlabel("true abundance")
        axes[j,i].set_ylabel("estimated abundance")

    #remove empty subplots
    if empty == 2:
        axes.flat[-1].set_visible(False)
        axes.flat[-2].set_visible(False)
    elif empty == 1:
        axes.flat[-1].set_visible(False)

    if path:
        plt.savefig(path + '.pdf')
    
    plt.show()

    return correlations

def calculate_corr(C_true, C_est, hidden_ct=None, c_est=None):
    """
    Function that calculates the correlation of estimated composition for multiple bulks with the ground truth.

    Parameters
    ----------
    C_true : pd.DataFrame
        True cellular compositions for all mixtures analyzed in C_est.
        Features all cell types, even if one cell type is hidden.
        Shape: n_cell_types x n_mixtures
    C_est : pd.DataFrame
        Estimated cellular compositions for multiple bulk mixtures.
        Features all non-hidden cell types from the ground truth.
        Shape: b_cell_types x n_mixtures
    hidden_ct : String (Optional)
        Needs to be set to the label of the cell type in C_true that is hidden, which means its not featured in C_est.
    c_est : pd.DataFrame (Optional)
        Estimated hidden background contributions for all mixtures featured in C_est (ADTD).
        Shape: 1 x n_mixtures

    Returns
    -------
    correlations : pd.DataFrame
        Spearman correlation values for C_true vs. C_est for each cell type.
        Shape: n_cell_types
    """

    # Inititialize pd.Series for correlation values
    correlations = pd.Series()

    # Calculate correlations for all non-hidden cell types
    for celltype in C_est.index:
        r_val, _ = spearmanr(C_true.loc[celltype], C_est.loc[celltype])
        correlations[celltype] = r_val

    # Calculate correlation for a hidden cell type
    if hidden_ct:
        r_val, _ = spearmanr(C_true.loc[hidden_ct], c_est.loc["hidden"])
        correlations['hidden'] = r_val
    
    return correlations


def fishers_exact_test(ranking,
                       geneset,
                       percentage=5.0,
                       verbose=True):
    """
    Function that performs Fisher's exact test to determine wether genes in a given set of genes are enriched
    in a top percentage of a ranked list of genes.

    Parameters
    ----------
    ranking : list
        Ranked list of genes, for instance the gene names of one column in a Delta matrix of an ADTD model after ordering it.
    geneset : list
        List of genes of interest, unordered, for instance a gene set corresponding to a specific pathway in one cell type.
    percentage : float
        Top percentage of the ranked list, that are considered 'highly' ranked, default is 5 percent, parameter of the Fisher's exact test.   
    verbose : bool
        Wether to print all results (default) or not.

    Returns
    -------
    results : dict
        Dictionary containing the statistical results of the test, keys "Odds Ratio" and "p-Value".
    contingency_df : pd.DataFrame
        Contingency table as pandas dataframe.
    """

    X = int(percentage * 1e-2 * len(ranking)) # Top X genes from rank_delta to compare

    # Step 1: Create sets of interest
    top_X_genes = set(ranking[:X])  # Top X genes
    regulated_genes_set = set(geneset)  # Convert to set for faster comparison

    # Step 2: Define the contingency table values
    a = len(top_X_genes & regulated_genes_set)  # Overlap between top X and regulated genes
    b = X - a  # Non-regulated genes in top X
    c = len(regulated_genes_set - top_X_genes)  # Regulated genes not in top X
    d = (len(ranking) - X) - c  # Non-regulated genes not in top X

    contingency_df = pd.DataFrame([[a, b], [c, d]], 
                                index=['In Top %i Genes' % (X), 'Not in Top %i Genes' % (X)], 
                                columns=['In Regulated Genes', 'Not in Regulated Genes'])

    # Step 3: Perform Fisher's Exact Test
    oddsratio, p_value = fisher_exact(contingency_df, alternative='greater')
    results = {"Odds Ratio":oddsratio,
            "p-Value": p_value}

    # Output
    if verbose:
        print("Contingency Table:")
        print(contingency_df)
        print("Test Statistics:")
        print(f"Odds Ratio: {oddsratio}, P-value: {p_value}\n")

    # Returns
    return results, contingency_df


def delta_enrichment(Delta, geneset, percentage=5):
    results = []
    for celltype in Delta.columns: 
        ranking = list(Delta.sort_values(by=celltype, ascending=False).index)
        X = int(percentage * 1e-2 * len(ranking)) # Top X genes from rank_delta to compare

        # Step 1: Create sets of interest
        top_X_genes = set(ranking[:X])  # Top X genes
        regulated_genes_set = set(geneset)  # Convert to set for faster comparison

        # Step 2: Define the contingency table values
        a = len(top_X_genes & regulated_genes_set)  # Overlap between top X and regulated genes
        b = X - a  # Non-regulated genes in top X
        c = len(regulated_genes_set - top_X_genes)  # Regulated genes not in top X
        d = (len(ranking) - X) - c  # Non-regulated genes not in top X

        contingency_df = pd.DataFrame([[a, b], [c, d]], 
                                    index=['In Top %i Genes' % (X), 'Not in Top %i Genes' % (X)], 
                                    columns=['In Regulated Genes', 'Not in Regulated Genes'])

        # Step 3: Perform Fisher's Exact Test
        oddsratio, p_value = fisher_exact(contingency_df, alternative='greater')

        results.append({
                    'cell_type': celltype,
                    'p_value': p_value,
                    'gene_count': a,
                    'es': -np.log(p_value)
                })
        
    results = pd.DataFrame(results)

    def get_enrichment_plot(data, number_celltypes):
        # Set global style parameters
        # plt.rcParams.update({
        #     'font.size': 12,                 # Global font size
        #     'axes.titlesize': 12,            # Title font size
        #     'axes.labelsize': 12,            # Axis label font size
        #     'xtick.labelsize': 12,           # X-tick label font size
        #     'ytick.labelsize': 12,           # Y-tick label font size
        #     'legend.fontsize': 12,           # Legend font size
        #     'figure.figsize': (4, 4),        # Figure size
        #     'axes.linewidth': 1.2,           # Line width for axes
        #     'grid.linewidth': 0.5,           # Line width for grid lines
        #     'grid.color': 'gray',            # Grid line color
        #     'legend.frameon': True,         # No frame for legend
        #     'axes.grid': False,               # Enable grid
        #     'grid.alpha': 0.5,
        #     'savefig.bbox' : 'tight',
        #     'savefig.pad_inches' : 0.05,
        #     'font.family' : 'serif',
        #     'mathtext.fontset' : 'dejavuserif'})

        data = data.nsmallest(number_celltypes, 'p_value')

        # # Set up the figure size
        # plt.rc('axes', axisbelow=True
        #        'font.size': 9)
        fig, ax = plt.subplots()

        scatter = sns.scatterplot(
            data=data,
            x='es', y='cell_type', size='gene_count', hue=data['p_value'],
            palette='RdYlBu', sizes=(10, 200), edgecolor='black', legend=False,
            ax=ax
        )
        ax.margins(y=0.2)

        #  colorbar (for p_value)
        norm = plt.Normalize(data['p_value'].min(), data['p_value'].max())
        sm = plt.cm.ScalarMappable(cmap="PuOr", norm=norm)
        sm.set_array([])

       
        cbar = fig.colorbar(sm, ax=ax, shrink=0.25, aspect=3, format="{x:.1e}", pad=0.125, spacing='uniform')
        cbar.ax.set_title("pvalue", pad=7)
        # cbar.ax.set_yticks([data['p_value'].min(), data['p_value'].max()])
        cbar.ax.invert_yaxis()
        # Set ticks for the colorbar
        ticks = np.linspace(data['p_value'].min(),  data['p_value'].max(), num=3)  # You can customize the number of ticks here
        cbar.ax.set_yticks(ticks)

        # # Optionally, set the tick labels if you want to customize them further
        # cbar.ax.set_ticklabels([f'{tick:.2f}' for tick in ticks])

        # legend (for gene_count)
        sizes = [10, 100, 200]  # Correspond to size range of points
        labels = [f'{int(size)}' for size in
                np.linspace(data['gene_count'].min(), data['gene_count'].max(), len(sizes))]

        for size, label in zip(sizes, labels):
            ax.scatter([], [], s=size, c='k', alpha=0.5, label=label)

        ax.legend(title="Genes in Top 50", loc="upper center", bbox_to_anchor=(0.5, 1.25), framealpha=0., columnspacing=0.8, ncol=3, handletextpad=0.3)

        # labels and title
        ax.set_xlabel('ES')
        ax.set_ylabel('')
        # ax.set_title(f'Enrichment Analysis')

        plt.grid(alpha=0.2)
        # Add extra space at the top and bottom
        plt.tight_layout()
        plt.subplots_adjust(top=0.825, bottom=0.15)
        # plt.savefig("example/output/delta_enrichment.pdf")
        # plt.savefig("example/output/delta_enrichment.png")
        plt.show()

    get_enrichment_plot(results, len(Delta.columns))




def gene_set_enrichment(Deltas, gene_set):
    """
    Perform gene set enrichment analysis using Mann-Whitney U tests.

    This function evaluates whether a specified gene set (e.g., a family of genes) 
    shows a statistically significant tendency to have higher or lower values 
    in the given data, compared to other genes. The analysis is conducted 
    across multiple datasets, each represented as a dataframe in `Deltas`.

    Parameters:
    ----------
    Deltas : list of pandas.DataFrame
        A list of dataframes where each dataframe contains gene expression 
        or delta values. Rows represent genes, and columns represent different 
        cell types or conditions.
    gene_set : str or list
        A string specifying the indicator of a gene family or list with gene names.
        Genes in each dataframe that contain this string will be considered part
        of the target gene set.

    Returns:
    -------
    all_results : list of pandas.DataFrame
        A list of dataframes, one for each dataframe in `Deltas`. Each result 
        dataframe contains the following columns:
        - `celltypes`: The cell type or condition.
        - `p_values`: P-values from the Mann-Whitney U test comparing the 
          target gene set against all other genes.
        - `direction`: Indicates whether the target gene set shows higher 
          ("up") or lower ("down") values compared to the other genes.
    """

    all_results = []

    for delta in Deltas:   
        results = {     
        "celltypes" : [],
        "p_values" : [],
        "direction" : []
        }

        not_wanted_genes = ["RPL", "RPS", "MT-"]
        for n in not_wanted_genes:
            drop_genes = [str(gene).upper() for gene in delta.index if n.upper() in gene.upper()]
            delta = delta.drop(index=drop_genes)
        

        
        for i in range(len(delta.columns)):
            gene_family = []
            not_family = []
            # Prepare ranked gene list
            current_celltype = delta.iloc[:, i]
            ranked_vector = (1 - current_celltype).abs().sort_values()

            if isinstance(gene_set, str):
                gene_family = [gene for gene in delta.index if gene_set in gene]
                not_family = [gene for gene in delta.index if gene_set not in gene]


            gene_set_ranks = []
            other_ranks = []
            for j, gene in enumerate(ranked_vector.index):
                if gene in gene_family:
                    gene_set_ranks.append(j)
                else:
                    other_ranks.append(j)
            

            statistics, p_value = mannwhitneyu(current_celltype.loc[gene_family], current_celltype.loc[not_family])

            
            results["celltypes"].append(delta.columns[i])
            results["p_values"].append(np.round(p_value, 3))
            if delta.loc[gene_family, delta.columns[i]].mean() < delta.loc[not_family, delta.columns[i]].mean():
                results["direction"].append("down")
            else:
                results["direction"].append("up")
            
        df = pd.DataFrame.from_dict(results)
        df.index = df.iloc[:,0]
        all_results.append(df)

    return all_results
        

            



    
