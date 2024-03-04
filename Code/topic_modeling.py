'''
This file contains code used for extended extracting the topics of the corpus. It does this using KeyBERT.
Keybert is a more modern, sophisticated tool for keyword extraction. This python file takes the json file containing the corpus and returns
the keywords for each file. The output is a csv file with the keywords that align with the indices of each document in 
the corpus. 

'''


from keybert import KeyBERT
import json
import numpy as np


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
    return np.array(keywords)

    
# Function to save keyword dataframe
def save_keywords(keywords, file_path_name):
    np.savetxt(file_path_name, keywords, delimiter=",", fmt="%s")

# Function to read in json lines file if that is data format
def read_jsonl(file_path_name):
    texts = list()
    ids = list()
    with open(file_path_name, "r") as jsonfile:
        for line in jsonfile:
            data = json.loads(line)
            text = data.get("speech") # May need to change this string to whatever the text is saved as in the jsonl file
            texts.append(text)
            id = data.get("speech_id")
            ids.append(str(id))
            

    text_array = np.array(texts)
    return text_array, ids

# Function to add speech id to keywords for easier tracking
def add_ids_to_keywords(ids, keywords):
    keywords_with_id = np.empty((keywords.shape[0], keywords.shape[1] + 1), dtype=object)
    keywords_with_id[:, 1:] = keywords
    for index, id in enumerate(ids):
        keywords_with_id[index, 0] = id
    return keywords_with_id

congresses = ["110", "111", "112", "113", "114"]

for term in congresses: 
    text, ids = read_jsonl("Data/relevant_speeches_%s.jsonl" % term) # Replace this read_jsonl function call with something to open other data type if necessary
    keywords = keybert_keywords(text, 3, 3)
    keywords_with_id = add_ids_to_keywords(ids, keywords)
    output_file_name = "Data/congressional_speech_%s_keywords.csv" % term
    save_keywords(keywords_with_id, output_file_name)