'''
The goal is to generate two corpora and output csv files that will be used in model_training.py for extended pre-training and fine-tuning of the sentence transformers model. 
'''
import json
import numpy as np
import os
from random import sample, choice
import re

# This code accepts two inputs:
#     (1) the json file created by the query_s2orc.py file, and 
#     (2) the json file containing text extracted from the LHIC that will not be searched for comparison with witness statements. 
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


#############################################################################################
# Will need to clean up following code for bugs
#############################################################################################



# Look into using billsum dataset: https://huggingface.co/datasets/FiscalNote/billsum/viewer/default/train
# This can be an additional source for fine tuning because it includes text from bills. 

# This code creates a fine-tuning corpus based on the text files in a directory. These text files were prepared in the same manner as those of the WLHIC, but 
# they are not being used to search for witness testimony impact. Furthermore, we use the BillSum dataset to augment our fine tuning corpus with additional legislative text. 
def generate_finetuning_corpus(directory_path:str, output_path:str, corpus_size:int = 50000):
  """
  Generates a fine-tuning corpus for sentence transformers.

  Args:
      directory_path: Path to the directory containing text files.
      output_path: Path to save the generated dataset as a numpy array.
      corpus_size: Number of triplets (anchor, positive, negative) to generate.
  """

  # Initialize empty lists to store sentences and document information
  sentences = []
  documents = {}
  current_doc = []

  # Loop through all files in the directory
  for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
      # Open file, read content, and split into sentences
      with open(file_path, 'r') as f:
        for line in f:
          sentences.extend(line.strip().split('.'))
          current_doc.extend(line.strip().split('.'))
      documents[filename] = current_doc
      current_doc = []  # Reset document list for next file

  # Filter out sentences less than or equal to 5 words
  filtered_sentences = [s for s in sentences if len(s.split()) > 5]

  # Initialize empty numpy array for triplets
  triplets = np.empty((corpus_size, 3), dtype=object)

  # Loop until we have the desired number of triplets
  for i in range(corpus_size):
    # Choose a random anchor sentence
    anchor_idx = sample(range(len(filtered_sentences)), 1)[0]
    anchor_sentence = filtered_sentences[anchor_idx]
    anchor_doc = [k for k, v in documents.items() if anchor_sentence in v][0]

    # Choose a positive sentence (adjacent from same document)
    positive_options = [
        idx for idx, sentence in enumerate(filtered_sentences)
        if sentence != anchor_sentence and (idx == anchor_idx - 1 or idx == anchor_idx + 1)
        and documents[filename_from_idx(filtered_sentences, idx)] == anchor_doc
    ]
    if positive_options:
      positive_idx = sample(positive_options, 1)[0]
      positive_sentence = filtered_sentences[positive_idx]

    # Choose a negative sentence (random from different doc)
    negative_doc = choice(list(documents.keys()))
    while negative_doc == anchor_doc:
      negative_doc = choice(list(documents.keys()))
    negative_options = [
        sentence for sentence in filtered_sentences
        if documents[filename_from_idx(filtered_sentences, idx)] == negative_doc
    ]
    negative_idx = sample(range(len(negative_options)), 1)[0]
    negative_sentence = negative_options[negative_idx]

    # Add the triplet to the numpy array
    triplets[i] = (anchor_sentence, positive_sentence, negative_sentence)

  # Save the triplets as a numpy array
  np.save(output_path, triplets)


def filename_from_idx(sentences, idx):
  """
  Helper function to get the filename of a sentence based on its index.
  """
  for filename, doc in documents.items():
    if sentences[idx] in doc:
      return filename
  return None







generate_pretraining_corpus("Data/papers.jsonl")