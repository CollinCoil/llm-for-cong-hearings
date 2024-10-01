"""
The goal is to generate a pre-training corporus and output a txt file that will be used in pretrain_model.py for extended pre-training
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

        # Only add sentences to the abstracts list if they have 3 or more words
        for sentence in sentences:
          if len(word_tokenize(sentence)) >= 3:
            abstracts.append(sentence)

  abstract_array = np.array(abstracts)
  # Save to desired output file with UTF-8 encoding
  with open("abstracts.txt", "w", encoding="utf-8") as f:
    np.savetxt(f, abstract_array, delimiter="\t", fmt="%s")  # Save as text file

