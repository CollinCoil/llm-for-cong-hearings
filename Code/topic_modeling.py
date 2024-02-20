# TODO: Use several techniques to generate topics for each file in the W&LHIC
# TODO: Look into KeyBERT and LDA
# Output: 
    # (1) List of keywords for every document in W&LHIC
        # This will allow us to make silly little word clouds for topics most often discussed by witnesses against the three branches

from keybert import KeyBERT
from sklearn.decomposition import LatentDirichletAllocation

def extract_keywords(document_list, n_keywords):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(document_list, keyphrase_ngram_range = (1,2), stop_words = "english",
                                         top_n = n_keywords, diversity = 0.5)
    
    return keywords


def lda(document_list, n_keywords):
    model = LatentDirichletAllocation(n_components = n_keywords, learning_method="online", random_state=2024)
    keywords = model.fit_transform(document_list)
    return keywords

documents = ["How often to evaluate perplexity. Only used in fit method. set it to 0 or negative number to not evaluate perplexity in training at all. Evaluating perplexity can help you check convergence in training process, but it will also increase total training time. Evaluating perplexity in every iteration might increase training time up to two-fold.", "In natural language processing, latent Dirichlet allocation (LDA) is a Bayesian network (and, therefore, a generative statistical model) for modeling automatically extracted topics in textual corpora. The LDA is an example of a Bayesian topic model. In this, observations (e.g., words) are collected into documents, and each word's presence is attributable to one of the document's topics. Each document will contain a small number of topics."]
extract_keywords(documents, 10)
lda(documents, 10)