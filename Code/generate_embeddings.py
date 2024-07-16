"""
This function accepts a corpus of text and calculates a text embedding for each sentence. 
"""


import json
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(model:str, corpus_file:str, output_file:str):
    # Load the fine-tuned model
    model = SentenceTransformer('Models/finetuned_model')

    # Initialize an empty list to store embeddings
    embeddings = []

    # Open the corpus file
    with open(corpus_file, 'r') as f:
        for line in f:
            # Load the json data from the line
            data = json.loads(line)

            # Use the model to generate embeddings for the text
            embedding = model.encode(data['Text'])

            # Add the embedding to the list
            embeddings.append(embedding)

    # Save the embeddings as a numpy array
    np.save(output_file, embeddings)


generate_embeddings(model = 'Models/finetuned_model', corpus_file = 'Data/lhic.jsonl', output_file = 'Data/lhic_embeddings.npy')