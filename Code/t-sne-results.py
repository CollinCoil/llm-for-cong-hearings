# TODO: Perform t-SNE on paragraphs related to Obamacare on the W&LHIC to see how the embeddings are distributed in space
# TODO: Use these t-SNE embeddings to see how witness statements compare to LHIC embeddings

from sklearn.manifold import TSNE

def create_tsne_embeddings(dataframe, perplexity = 50):
    tsne_embeddings = TSNE(n_componsnts = 2, perplexity = perplexity).fit_transform(dataframe)
    return tsne_embeddings