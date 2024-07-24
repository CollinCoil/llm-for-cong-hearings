"""
The goal is to generate two corpora and output csv files that will be used in pretrain_model.py for extended pre-training and finetune_model.py for fine-tuning 
of the sentence transformers model. 
"""
import json
import numpy as np
import os
from random import choice
import re
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer


def generate_pretraining_corpus(data_path:str):
  """
  This function takes a corpus of text and transforms it into a usable pretraining corpus. For the original project, the output of query_s2orc.py is the original pretraining corpus

  Args: 
    data_path: a string for the directory of the original corpus
  """
  abstracts = list()
  tokenizer = PunktSentenceTokenizer()  # Initialize PunktSentenceTokenizer

  # Opens the json lines file and extracts the abstract text 
  with open(data_path, "r", encoding='utf-8') as jsonfile:
    for line in jsonfile:
      data = json.loads(line)
      abstract = data.get("abstract")
        
      if abstract is not None:
        line_skip = "\n"
        abstract = re.sub(line_skip, " ", abstract)
        abstract = abstract.strip()

        # Split abstract into sentences using PunktSentenceTokenizer
        sentences = tokenizer.tokenize(abstract)

        # Append each sentence to the abstracts list
        abstracts.extend(sentences)

  abstract_array = np.array(abstracts)
  # Save to desired output file with UTF-8 encoding
  with open("Data/abstracts.txt", "w", encoding="utf-8") as f:
    np.savetxt(f, abstract_array, delimiter="\t", fmt="%s")  # Save as text file



def generate_finetuning_corpus(directory_path:str, output_path:str = "output.jsonl", corpus_size:int = 150000):
  """
  This code creates a fine-tuning corpus based on the text files in a directory. These text files were prepared in the same manner as those of the WLHIC, but 
  they are not being used to search for witness testimony impact.

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
              with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                  text = f.read()
                  # Tokenize the text into sentences
                  sents = sent_tokenize(text)
                  # Filter out sentences that are too short
                  sents = [sent for sent in sents if len(word_tokenize(sent)) > 8]
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
    
  # Save to desired output file with UTF-8 encoding
  with open("Data/finetuning_text.txt", "w", encoding="utf-8") as f:
    np.savetxt(f, triplets, delimiter="\t", fmt="%s")  # Save as text file





generate_pretraining_corpus("Data/papers.jsonl")
generate_finetuning_corpus(directory_path = '../../Fine Tuning Text', output_path = 'Data/finetuning_text.npy', corpus_size = 200000)