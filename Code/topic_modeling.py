'''
This file contains code used for extended extracting the topics of the corpus. It does this through two different techniques: (1) KeyBERT and (2) Latent Dirichlet Allocation (LDA).
Keybert is a more modern, sophisticated tool, and LDA is a classic tool for topic modeling. This python file takes the json file containing the corpus and returns
the keywords for each file. You can choose to use both KeyBERT and LDA or either one. The output is a csv file with the keywords that align with the indices of each document in 
the corpus. 

'''


from keybert import KeyBERT
import numpy as np
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')  
nltk.download('omw-1.4')  
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Code to extract keywords using KeyBert
def keybert_keywords(document_list, n_keywords):
    kw_model = KeyBERT()
    keybert_results = kw_model.extract_keywords(document_list, keyphrase_ngram_range = (1,2), stop_words = "english",
                                         top_n = n_keywords, diversity = 0.5, use_mmr=True)
    keywords = list()
    for document in keybert_results:
        document_keywords = list()
        for word, value in document:
            document_keywords.append(word)

        keywords.append(document_keywords)
    return keywords


# Code to perform LDA is adapted from https://www.datacamp.com/tutorial/what-is-topic-modeling with some changes. 

# Remove stopwords, punctuation, and normalize the corpus
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def create_lda_inputs(document_list):
    clean_documents = [clean(doc).split() for doc in document_list]
    dictionary = corpora.dictionary.Dictionary(clean_documents)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_documents]
    return dictionary, doc_term_matrix

def lda_keywords(document_list, n_keywords):
    dictionary, document_term_matrix = create_lda_inputs(document_list)
    lda_model = LdaModel(document_term_matrix, id2word = dictionary, num_topics = n_keywords, random_state = 2024)
    keywords = list()
    for doc in document_term_matrix:
        topics = lda_model[doc] # Gets topics for each document
        document_keywords = list()
        for topic, prob in topics: 
            topic_list = lda_model.show_topic(topic, topn=n_keywords)
            for word, value in topic_list:
                document_keywords.append(word)

        keywords.append(document_keywords)
            
    return keywords


# Function to easily generate keywords using both Keybert and LDA
def generate_keywords(document_list, n_keybert=1, n_lda=1):
    if n_keybert == 0 and n_lda == 0:
        raise ValueError("At least one of n_keybert and n_lda needs to be greater than 0.")
    if n_keybert > 0:
        keybert_results = np.array(keybert_keywords(document_list, n_keybert))
    if n_lda > 0:
        lda_results = np.array(lda_keywords(document_list, n_lda))
    if n_keybert > 0 and n_lda > 0:
        keywords = np.hstack((keybert_results, lda_results))
        return keywords
    elif n_keybert == 0:
        return lda_results
    else:
        return keybert_results
    
# Function to save keyword dataframe
def save_keywords(keywords, file_path_name):
    np.savetxt(file_path_name, keywords, delimiter=",")


keywords = generate_keywords("Data/file_path.npy", 5, 5) # TODO: Add this numpy array or json file with the corpus here
save_keywords(keywords, "Data/corpus_keywords.csv")