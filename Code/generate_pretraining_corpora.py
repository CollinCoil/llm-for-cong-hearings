'''
This code accepts two inputs:
    (1) the json file created by the query_s2orc.py file, and 
    (2) the json file containing text extracted from the LHIC that will not be searched for comparison with witness statements. 

The goal is to generate two corpora and output csv files that will be used in model_training.py for extended pre-training and fine tuning of the sentence transformers model. 
'''
import json
import numpy as np
import re

# TODO: Generate two corpora:
    # (2) Whatever texts are discarded in the LHIC
        # Use a zero-shot classification model to identify paragraphs in the discarded texts that are related to given topics, and use them to create paragraph pairings
            # Paragraphs with similar topics will be used to make pairs of similar paragraphs, paragraphs with different topics will be used to make pairs of different topics


def generate_pretraining_corpus(data_path:str):
    abstracts = list()
    with open(data_path, "r") as jsonfile:
        for line in jsonfile:
            data = json.loads(line)
            abstract = data.get("abstract")
            
            if abstract is not None:
                line_skip = "\n"
                abstract = re.sub(line_skip, " ", abstract)
                abstract = abstract.strip()
                abstracts.append(abstract)

    abstract_array = np.array(abstracts)
    np.save("Data/abstracts.npy", abstract_array)








def generate_finetuning_corpus(data_path:str, output_path:str):
    pass







generate_pretraining_corpus("Data/papers.jsonl")