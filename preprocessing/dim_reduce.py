import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(X)
    
    pca_df = pd.DataFrame(
        pca_result, 
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    loadings_df = pd.DataFrame(
        pca.components_,
        columns=X.columns,
        index=[f'PC{i+1}' for i in range(n_components)]
    )
    
    print("\n--- PCA Explained Variance Ratio ---")
    print(pca.explained_variance_ratio_)
    
    return pca_df, loadings_df


def apply_tsne(X, n_components=2, perplexity=30.0):

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(X)
    
    tsne_df = pd.DataFrame(
        tsne_result, 
        columns=[f'tSNE{i+1}' for i in range(n_components)]
    )
    return tsne_df
