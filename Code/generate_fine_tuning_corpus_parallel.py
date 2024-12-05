"""
This file is an alternative to generate_training_corpora.py for creating the generate_training_corpora.py function generate_finetuning_corpus(). 
This is a parallelized implementation of the fine tuning corpus generation, allowing for much faster processing of the input txt files and triplet generation. 
The output txt file from this function will be used in finetune_model.py for fine-tuning of the sentence transformers model. 
"""

import os
import numpy as np
from random import choice
from nltk.tokenize import sent_tokenize, word_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def clean_sentence(sentence):
    """
    Clean a sentence by removing extraneous characters.
    """
    sentence = re.sub(r'[\t\n|]', ' ', sentence)  # Replace \t, \n, and | with a space
    sentence = re.sub(r'\s+', ' ', sentence)  # Replace multiple spaces with a single space
    return sentence.strip()  # Remove leading and trailing spaces


def process_file(file_path):
    """
    Process a single file: read the content, tokenize into sentences,
    filter out short sentences, and return a list of cleaned sentences with their origin.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        sents = sent_tokenize(text)
        sents = [clean_sentence(sent) for sent in sents if len(word_tokenize(sent)) > 8]

        # Filter out documents with fewer than 5 sentences
        if len(sents) < 5:
            return []  # Return an empty list if fewer than 5 sentences

        return [(sent, os.path.basename(file_path)) for sent in sents]


def generate_triplet(sentences, sentence_dict):
    """
    Generate a single triplet (anchor, positive, negative).
    """
    # Randomly select an anchor sentence
    anchor, anchor_file = choice(sentences)
    anchor_list = sentence_dict[anchor_file]
    anchor_index = anchor_list.index(anchor)
    
    # Select a positive match (adjacent sentence)
    pos_index = choice([index for index in [anchor_index - 1, anchor_index + 1] if 0 <= index < len(anchor_list)])
    positive = anchor_list[pos_index]

    # Select a negative match from two ranges
    negative_candidates = []
    
    # Negative range: 3 to 5 sentences before the anchor
    for offset in range(2, 6):
        neg_index = anchor_index - offset
        if 0 <= neg_index < len(anchor_list):
            neg_sentence = anchor_list[neg_index]
            if neg_sentence != anchor and neg_sentence != positive:
                negative_candidates.append(neg_sentence)

    # Negative range: 2 to 5 sentences after the anchor
    for offset in range(2, 6):
        neg_index = anchor_index + offset
        if 0 <= neg_index < len(anchor_list):
            neg_sentence = anchor_list[neg_index]
            if neg_sentence != anchor and neg_sentence != positive:
                negative_candidates.append(neg_sentence)

    # If there are valid negatives, choose one randomly
    negative = choice(negative_candidates) if negative_candidates else None
    
    return (anchor, positive, negative)


def generate_finetuning_corpus(directory_path: str, output_path: str = "output.jsonl", corpus_size: int = 150000):
    """
    This code creates a fine-tuning corpus based on the text files in a directory.

    Args:
        directory_path: Path to the directory containing text files.
        output_path: Path to save the generated dataset as a numpy array.
        corpus_size: Number of triplets (anchor, positive, negative) to generate.
    """

    sentences = []

    # Parallelize the file reading and sentence tokenization
    with ThreadPoolExecutor() as executor:
        futures = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".txt"):
                    futures.append(executor.submit(process_file, os.path.join(root, file)))

        for future in as_completed(futures):
            sentences.extend(future.result())

    # Create a dictionary to quickly access sentences by file
    sentence_dict = {}
    for sent, file in sentences:
        if file not in sentence_dict:
            sentence_dict[file] = []
        sentence_dict[file].append(sent)
    
    # Parallelize the triplet generation
    triplets = np.empty((corpus_size, 3), dtype=object)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_triplet, sentences, sentence_dict) for _ in range(corpus_size)]
        
        for i, future in enumerate(as_completed(futures)):
            triplets[i] = future.result()

    # Save to the desired output file with UTF-8 encoding
    with open(output_path, "w", encoding="utf-8") as f:
        np.savetxt(f, triplets, delimiter="|", fmt="%s", encoding="utf-8")


# Example Usage
# directory_path = r'path\to\finetuning\directory'
# output_path = r'output\file\name'
# generate_finetuning_corpus(directory_path, output_path, corpus_size = 500000)
