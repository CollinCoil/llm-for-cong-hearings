"""
This file contains code used for extended extracting the topics of the corpus. It does this using KeyBERT.
Keybert is a more modern, sophisticated tool for keyword extraction. This python file takes the json file or directory containing the corpus and returns
the keywords for each file. The output is a csv file with the keywords that align with the indices of each document in 
the corpus. 
"""


from keybert import KeyBERT
import json
import numpy as np
import os


# Code to extract keywords using KeyBert
def keybert_keywords(document_list, n_keywords:int):
    """
    A function to determine keywords for a document based on the document's primary topic. 

    Args: 
        document_list: a list, numpy array, pandas dataframe column or similar containing documents for which keywords should be extracted
        n_keywords: the number of keywords to be extracted. 
    """
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
def save_keywords(keywords, file_path_name: str, encoding = None):
    """
    Helper function to save the keywords in a dataframe. 

    Args: 
        keywords: a numpy array of keywords
        file_path_name: the location and name for the output file
        encoding: the text encoding for the keywords
    """
    np.savetxt(file_path_name, keywords, delimiter=",", fmt="%s", encoding = encoding)

# Function to read in json lines file if that is data format
def read_jsonl(file_path_name:str):
    """
    Helper function to read a json lines file containing documents for keyword extraction. 

    Args: 
        file_path_name: the name of the jsone lines file
    """
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

# Function to read text files from directory
def read_directory(directory:str):
    """
    Helper function to read all files from a directory

    Args: 
        directory: string of the directory location
    """
    texts = list()
    file_names = list()
    for filename in os.listdir(directory):
        filename = directory + filename
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
            if text != "":
                text = " ".join(text.split("\n"))
                texts.append(text)
                file_names.append(filename)
    text_array = np.array(texts)
    return text_array, file_names
    

# Function to add speech id to keywords for easier tracking
def add_ids_to_keywords(ids, keywords):
    """
    Helper function to connect keywords from congressional speeches with the ID of the speech
    """
    keywords_with_id = np.empty((keywords.shape[0], keywords.shape[1] + 1), dtype=object)
    keywords_with_id[:, 1:] = keywords
    for index, id in enumerate(ids):
        keywords_with_id[index, 0] = id
    return keywords_with_id


def get_topics(text_type:str):
    """
    Helper function to determine the type of text the keywords are being extracted from, then extract and save the keywords. 

    Args: 
        text_type: a string stating what documents are having keywords extracted
    """
    if text_type == "floor speech":
        # Extracting keywords from congressional speeches
        congresses = ["110", "111", "112", "113", "114" ]    
        for term in congresses: 
            text, ids = read_jsonl("Data/relevant_speeches_%s.jsonl" % term) # Replace this read_jsonl function call with something to open other data type if necessary
            keywords = keybert_keywords(text, 10)
            keywords_with_id = add_ids_to_keywords(ids, keywords)
            output_file_name = "Data/congressional_speech_%s_keywords.csv" % term
            save_keywords(keywords_with_id, output_file_name)

    if text_type == "witness testimony":
        text, names = read_directory("Data/Witness Statement Txt/")
        names_column = np.array(names)[:, np.newaxis]

        keywords = keybert_keywords(text, 10)
        keywords_with_names = np.hstack((names_column, keywords))
        output_file_name = "Data/witness_statement_keywords.csv"
        save_keywords(keywords_with_names, output_file_name, encoding = "utf-8")


get_topics("witness testimony")