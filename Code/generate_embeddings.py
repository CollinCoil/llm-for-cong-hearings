"""
This function accepts a corpus of text and calculates a text embedding for each sentence. It accepts json lines files created by process_text.py and outputs embedding numpy arrays
for use in semantic_similarity.py. 
"""


import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def generate_embeddings(model:str, corpus_file:str, output_file:str):
    """
    This function uses a sentence transformed model that is fine tuned for semantic textual similarity to calculate a text embedding for each sentence. These
    embeddings capture semantic and syntactic content of the sentence. Similar sentences will have embeddings close to each other in embedding space. 

    Args: 
        model: a string of the directory where the model is stored
        corpus_file: a string of the json lines file location containing the text for embeddings
        output_file: a string of the file name and location that the output embeddings should be written
    """

    # Check if a GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the fine-tuned model
    model = SentenceTransformer('Models/finetuned_model', device=device)

    # Initialize an empty list to store embeddings
    embeddings = []

    # Open the corpus file
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Load the json data from the line
            data = json.loads(line)

            # Use the model to generate embeddings for the text
            embedding = model.encode(data['text'], device=device)

            # Add the embedding to the list
            embeddings.append(embedding)

    # Save the embeddings as a numpy array
    np.save(output_file, embeddings)


generate_embeddings(model = r'path/to/model', corpus_file = r'path/to/corpus', output_file = r'output/file/name')