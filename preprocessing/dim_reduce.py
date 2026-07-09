import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Squashes many columns into a few summary dimensions that capture most of
# the pattern in the data, so it's easy to plot; also reports which
# original columns matter most for each summary dimension.
def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(X)
    
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)]  # name each output column PC1, PC2, ... one per summary dimension
    )
    
    loadings_df = pd.DataFrame(
        pca.components_,
        columns=X.columns,
        index=[f'PC{i+1}' for i in range(n_components)]  # label each row PC1, PC2, ... one per summary dimension
    )
    
    print("\n--- PCA Explained Variance Ratio ---")
    print(pca.explained_variance_ratio_)
    
    return pca_df, loadings_df


# Builds a 2D map where similar people are placed near each other, which
# is useful for spotting visual clusters that PCA might miss.
def apply_tsne(X, n_components=2, perplexity=30.0):

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(X)
    
    tsne_df = pd.DataFrame(
        tsne_result,
        columns=[f'tSNE{i+1}' for i in range(n_components)]  # name each output column tSNE1, tSNE2, ... one per map dimension
    )
    return tsne_df
