'''
The goal is to generate two corpora and output csv files that will be used in model_training.py for extended pre-training and fine-tuning of the sentence transformers model. 
'''
import json
import numpy as np
import os
from random import choice
import re
from nltk.tokenize import sent_tokenize, word_tokenize

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



# Look into using billsum dataset: https://huggingface.co/datasets/FiscalNote/billsum/viewer/default/train
# This can be an additional source for fine tuning because it includes text from bills. 

# This code creates a fine-tuning corpus based on the text files in a directory. These text files were prepared in the same manner as those of the WLHIC, but 
# they are not being used to search for witness testimony impact. Furthermore, we use the BillSum dataset to augment our fine tuning corpus with additional legislative text. 
def generate_finetuning_corpus(directory_path:str, output_path:str, corpus_size:int = 500000):
  """
  Generates a fine-tuning corpus for sentence transformers.

  Args:
      directory_path: Path to the directory containing text files.
      output_path: Path to save the generated dataset as a numpy array.
      corpus_size: Number of triplets (anchor, positive, negative) to generate.
  """

  # Initialize an empty list to hold all sentences
  sentences = []

  # Walk through all files in the directory
  for root, dirs, files in os.walk(directory_path):
      for file in files:
          if file.endswith(".txt"):
              # Open the file and read its content
              with open(os.path.join(root, file), 'r') as f:
                  text = f.read()
                  # Tokenize the text into sentences
                  sents = sent_tokenize(text)
                  # Filter out sentences that are too short
                  sents = [sent for sent in sents if len(word_tokenize(sent)) > 5]
                  # Add the sentences to the list, keeping track of their origin
                  sentences.extend([(sent, file) for sent in sents])
  
  # Initialize the numpy array to hold the triplets
  triplets = np.empty((corpus_size, 3), dtype=object)
  
  # Generate the triplets
  for i in range(corpus_size):
      # Randomly select an anchor sentence
      anchor, anchor_file = choice(sentences)
      # Find the index of the anchor sentence in the list
      anchor_index = next((i for i, (sent, file) in enumerate(sentences) if sent == anchor and file == anchor_file), None)
      # Select a positive match
      pos_index = choice([anchor_index - 1, anchor_index + 1])
      while sentences[pos_index][1] != anchor_file:
          pos_index = choice([anchor_index - 1, anchor_index + 1])
      positive, _ = sentences[pos_index]
      # Select a negative match
      negative, _ = choice(sentences)
      while negative == anchor or negative == positive or _ == anchor_file:
          negative, _ = choice(sentences)
      # Add the triplet to the numpy array
      triplets[i] = (anchor, positive, negative)
    
  # Save the numpy array to a file
  np.save(output_path, triplets)






generate_pretraining_corpus("Data/papers.jsonl")