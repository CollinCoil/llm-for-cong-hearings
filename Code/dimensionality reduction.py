"""
This file contains two functions to reduce dimensionality of embeddings. The embeddings created by generate_embeddings.py are high-dimensional, 
which makes it impossible to visualize. These functions transform the high-dimenional embeddings into two-dimensional embeddings. t-SNE is a nonlinear
dimensionality reduction technique that attempts to presserve local structures. UMAP is a different nonlinear dimensionality reduction that assumes data 
is distributed on a Riemann manifold, models the manifold with a topological structure, and searches for a low-dimensional projection with the closest 
topological structure. 
"""

from sklearn.manifold import TSNE
import numpy as np
from umap import UMAP

def create_tsne_embeddings(original_embeddings, perplexity:int = 50):
    """
    Performs t-SNE on a numpy array of embeddings. 

    Args: 
        original_embeddings: a numpy array
        perplexity: an integer to govern how to balance the attention between local and global phenomena in the data
    
    """
    tsne_embeddings = TSNE(n_components = 2, perplexity = perplexity).fit_transform(original_embeddings)
    return tsne_embeddings

def create_umap_embeddings(original_embeddings):
    """
    Performs UMAP on a numpy array of embeddings. 

    Args: 
        original_embeddings: a numpy array
    """
    mapping = UMAP(
        n_neighbors=5,
        min_dist=0.05,
        n_components=2,
        metric="euclidean"
    )
    umap_embedding = mapping.fit_transform(original_embeddings)
    return umap_embedding

