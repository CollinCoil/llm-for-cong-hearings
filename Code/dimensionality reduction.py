from sklearn.manifold import TSNE
import numpy as np
from umap import UMAP

def create_tsne_embeddings(dataframe, perplexity = 50):
    tsne_embeddings = TSNE(n_componsnts = 2, perplexity = perplexity).fit_transform(dataframe)
    return tsne_embeddings

def create_umap_embeddings(dataframe):
    mapping = UMAP(
        n_neighbors=5,
        min_dist=0.05,
        n_components=2,
        metric="minkowski"
    )
    umap_embedding = mapping.fit_transform(dataframe)
    return umap_embedding

