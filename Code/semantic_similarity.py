"""
This function calculates the cosine similarity between all embeddings in the witness corpus and the LHIC. It does this
by calculating the cosine similarity between each sentence. The output of this function is a JSON Lines file that 
contains the metadata from the Witness Corpus, metadata from the LHIC, and the sentence similarity score.  
This semantic similarity file can then be used to (1) manually review similar sentences for "hits" and 
(2) calculate metrics of witness impact. 
"""

import numpy as np
import json
import torch


def calculate_similarity(first_embeddings_path, first_corpus_jsonl,
                         second_embeddings_path, second_corpus_jsonl,
                         output_path, top_n=5):
    """
    For each sentence in the first corpus, find the top N most similar sentences
    between the sentence and all sentences in the second corpus using cosine similarity.

    Args:
        first_embeddings_path (str): Path to the .npy file containing sentence embeddings for the first corpus.
        first_corpus_jsonl (str): Path to the JSONL file with sentence metadata for the first corpus.
        second_embeddings_path (str): Path to the .npy file containing sentence embeddings for the second corpus.
        second_corpus_jsonl (str): Path to the JSONL file with sentence metadata for the second corpus.
        output_path (str): Path to save the output JSONL file.
        top_n (int): Number of top similar sentence pairs to extract for each sentence in the first corpus.
    """

    # Check if GPU is available and set device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the embeddings and transfer to the appropriate device
    first_embeddings = torch.tensor(np.load(first_embeddings_path)).to(device)
    second_embeddings = torch.tensor(np.load(second_embeddings_path)).to(device)

    # Load the metadata
    with open(first_corpus_jsonl, encoding='utf-8') as f:
        first_metadata = [json.loads(line) for line in f]

    with open(second_corpus_jsonl, encoding='utf-8') as f:
        second_metadata = [json.loads(line) for line in f]

    # Open the output file
    with open(output_path, 'w', encoding='utf-8') as f:

        # For each sentence in the first corpus
        for i, sentence in enumerate(first_metadata):
            # Get the embedding for the current sentence
            sentence_embedding = first_embeddings[i].unsqueeze(0)

            # Calculate similarity between the current sentence and all sentences in the second corpus
            similarities = torch.matmul(sentence_embedding, second_embeddings.T).squeeze(0)

            # Find the top-N similar sentences
            top_indices = torch.topk(similarities, top_n).indices

            # Write the top-N results to the output file
            for index in top_indices:
                f.write(json.dumps({
                    'first_metadata': sentence,
                    'second_metadata': second_metadata[index.item()],
                    'similarity': float(similarities[index].item())
                }, ensure_ascii=False) + '\n')

calculate_similarity(
    r"Data\Zenodo\wc_embeddings.npy",
    r"Data\Zenodo\witness_corpus.jsonl",
    r"Data\Zenodo\lhic_embeddings.npy",
    r"Data\Zenodo\legislative_history_impact_corpus.jsonl",
    r"Data\Zenodo\sentence_similarities.jsonl"
)
