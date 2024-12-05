"""
These functios calculate the cosine similarity between all embeddings in the witness corpus and the LHIC. It does this
by calculating the cosine similarity between each sentence. The output of this function is a JSON Lines file that 
contains the metadata from the Witness Corpus, metadata from the LHIC, and the sentence similarity score.  
This semantic similarity file can then be used to (1) manually review similar sentences for "hits" and 
(2) calculate metrics of witness impact. 
"""

import numpy as np
import json
import torch


def calculate_similarity_by_top_n(first_embeddings_path, first_corpus_jsonl,
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
                # Because my vectors are length 1, matmul is equivalent to cosine similarity. Must check if vectors are length 1 before proceeding

            # Find the top-N similar sentences
            top_indices = torch.topk(similarities, top_n).indices

            # Write the top-N results to the output file
            for index in top_indices:
                f.write(json.dumps({
                    'first_metadata': sentence,
                    'second_metadata': second_metadata[index.item()],
                    'similarity': float(similarities[index].item())
                }, ensure_ascii=False) + '\n')

# Example Usage
# calculate_similarity_by_top_n(
#     r"path\to\embeddings_1",
#     r"path\to\jsonl_1",
#     r"path\to\embeddings_2",
#     r"path\to\jsonl_2",
#     r"path\to\output"
# )


def calculate_similarity_by_threshold(first_embeddings_path, first_corpus_jsonl,
                         second_embeddings_path, second_corpus_jsonl,
                         output_path, similarity_threshold=0.9, batch_size=5000):
    """
    For each sentence in the first corpus, find all sentence pairs between the sentence and the second corpus
    where the cosine similarity exceeds a specified threshold.

    Args:
        first_embeddings_path (str): Path to the .npy file containing sentence embeddings for the first corpus.
        first_corpus_jsonl (str): Path to the JSONL file with sentence metadata for the first corpus.
        second_embeddings_path (str): Path to the .npy file containing sentence embeddings for the second corpus.
        second_corpus_jsonl (str): Path to the JSONL file with sentence metadata for the second corpus.
        output_path (str): Path to save the output JSONL file.
        similarity_threshold (float): Similarity threshold to consider when selecting sentence pairs.
        batch_size (int): Number of sentences to process in batches for faster similarity calculations.
    """

    # Check if GPU is available and set device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the embeddings and transfer to the appropriate device
    first_embeddings = torch.tensor(np.load(first_embeddings_path)).to(device)
    second_embeddings = torch.tensor(np.load(second_embeddings_path)).to(device)

    # Normalize embeddings for cosine similarity calculation
    first_embeddings = torch.nn.functional.normalize(first_embeddings, p=2, dim=1)
    second_embeddings = torch.nn.functional.normalize(second_embeddings, p=2, dim=1)

    # Load the metadata
    with open(first_corpus_jsonl, encoding='utf-8') as f:
        first_metadata = [json.loads(line) for line in f]

    with open(second_corpus_jsonl, encoding='utf-8') as f:
        second_metadata = [json.loads(line) for line in f]

    # Open the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        buffer = []  # Buffer to hold results before writing

        # Process in batches to improve performance
        num_sentences = len(first_embeddings)
        for start_idx in range(0, num_sentences, batch_size):
            end_idx = min(start_idx + batch_size, num_sentences)
            batch_embeddings = first_embeddings[start_idx:end_idx]  # Batch from first corpus

            # Calculate similarities for the entire batch at once
            similarities = torch.matmul(batch_embeddings, second_embeddings.T)
                # Because my vectors are length 1, matmul is equivalent to cosine similarity. Must check if vectors are length 1 before proceeding

            # Apply threshold and gather results
            for i in range(similarities.shape[0]):
                first_idx = start_idx + i
                sentence_similarities = similarities[i]

                # Find where similarity exceeds the threshold
                high_similarity_indices = (sentence_similarities >= similarity_threshold).nonzero(as_tuple=False).squeeze()

                # If high_similarity_indices is a 0-d tensor (a single value), wrap it in a list
                if high_similarity_indices.dim() == 0:
                    high_similarity_indices = [high_similarity_indices.item()]
                else:
                    high_similarity_indices = high_similarity_indices.tolist()

                # Store the results in the buffer
                for j in high_similarity_indices:
                    buffer.append({
                        'first_metadata': first_metadata[first_idx],
                        'second_metadata': second_metadata[j],
                        'similarity': float(sentence_similarities[j].item())
                    })

            # Write to file when buffer gets large
            if len(buffer) > 10000:
                for item in buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                buffer.clear()

        # Write any remaining buffered results to file
        for item in buffer:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# similarity_threshold = 0.75
# batch_size = 1000
# calculate_similarity_by_threshold(
#     r"path\to\embeddings_1",
#     r"path\to\jsonl_1",
#     r"path\to\embeddings_2",
#     r"path\to\jsonl_2",
#     r"path\to\output",
#     similarity_threshold,
#     batch_size
# )