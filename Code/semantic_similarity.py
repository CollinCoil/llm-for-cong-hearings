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
                          output_path, top_n=50):
    """
    For each document in the first corpus, find the top N most similar sentences
    between the document's sentences and all sentences in the second corpus using cosine similarity.

    Args:
        first_embeddings_path (str): Path to the .npy file containing sentence embeddings for the first corpus.
        first_corpus_jsonl (str): Path to the JSONL file with sentence metadata for the first corpus.
        second_embeddings_path (str): Path to the .npy file containing sentence embeddings for the second corpus.
        second_corpus_jsonl (str): Path to the JSONL file with sentence metadata for the second corpus.
        output_path (str): Path to save the output JSONL file.
        top_n (int): Number of top similar sentence pairs to extract for each document.
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
        # Group the sentences by document in the first corpus
        documents = {}
        for item in first_metadata:
            doc_id = item['document']
            if doc_id not in documents:
                documents[doc_id] = []
            documents[doc_id].append(item)

        # For each document in the first corpus
        for doc_id, sentences in documents.items():
            sentence_ids = [sentence['ID'] for sentence in sentences]

            # Slice the embeddings for this document
            doc_embeddings = first_embeddings[sentence_ids]

            # Calculate the similarity using PyTorch's matrix multiplication
            similarities = torch.matmul(doc_embeddings, second_embeddings.T)

            # Find the top-N similarities
            top_similarities = []
            for i, row in enumerate(similarities):
                top_indices = torch.topk(row, top_n).indices
                for index in top_indices:
                    top_similarities.append((i, index.item(), row[index].item()))

            # Sort and write the results
            top_similarities = sorted(top_similarities, key=lambda x: x[2], reverse=True)[:top_n]
            for first_idx, second_idx, sim_score in top_similarities:
                f.write(json.dumps({
                    'first_metadata': first_metadata[sentence_ids[first_idx]],
                    'second_metadata': second_metadata[second_idx],
                    'similarity': float(sim_score)
                }, ensure_ascii=False) + '\n')

calculate_similarity(
    r"Data\Zenodo\wc_embeddings.npy",
    r"Data\Zenodo\witness_corpus.jsonl",
    r"Data\Zenodo\lhic_embeddings.npy",
    r"Data\Zenodo\lhi_corpus.jsonl",
    r"Data\Zenodo\sentence_similarities.jsonl"
)
