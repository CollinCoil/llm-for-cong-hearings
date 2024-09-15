"""
This function calculates the cosine similarity between all embeddings in the witness corpus and the LHIC. It does this
by calculating the cosine similarity between each sentence. The output of this function is a JSON Lines file that 
contains the metadata from the Witness Corpus, metadata from the LHIC, and the sentence similarity score.  
This semantic similarity file can then be used to (1) manually review similar sentences for "hits" and 
(2) calculate metrics of witness impact. 
"""

import numpy as np
import json
from scipy.spatial.distance import cdist
import heapq

def get_top_similar_sentences(first_embeddings_path: str, first_corpus_jsonl: str, 
                              second_embeddings_path: str, second_corpus_jsonl: str, 
                              output_path: str, top_n: int = 30):
    """
    For each document in the first corpus, find the top N most similar sentences
    in the second corpus using cosine similarity.

    Args:
        first_embeddings_path (str): Path to the .npy file containing sentence embeddings for the first corpus.
        first_corpus_jsonl (str): Path to the JSONL file with sentence metadata for the first corpus.
        second_embeddings_path (str): Path to the .npy file containing sentence embeddings for the second corpus.
        second_corpus_jsonl (str): Path to the JSONL file with sentence metadata for the second corpus.
        output_path (str): Path to save the output JSONL file.
        top_n (int): Number of top similar sentences to extract for each document.
    """

    # Load embeddings
    first_embeddings = np.load(first_embeddings_path).astype(np.float32)
    second_embeddings = np.load(second_embeddings_path).astype(np.float32)

    # Load corpus metadata
    first_corpus = [json.loads(line) for line in open(first_corpus_jsonl, 'r', encoding='utf-8')]
    second_corpus = [json.loads(line) for line in open(second_corpus_jsonl, 'r', encoding='utf-8')]

    second_sentences = {entry["ID"]: entry for entry in second_corpus}

    # Prepare output file
    with open(output_path, 'w', encoding='utf-8') as outfile:

        # Process each document in the first corpus
        current_doc = None
        doc_sentences = []
        doc_embeddings = []

        for i, entry in enumerate(first_corpus):
            doc_id = entry['document']
            
            # Collect sentences for the current document
            if current_doc is None or doc_id == current_doc:
                doc_sentences.append(entry)
                doc_embeddings.append(first_embeddings[i])
                current_doc = doc_id
            else:
                # Process the previous document once it's finished collecting
                doc_embeddings = np.array(doc_embeddings)

                # Calculate similarity for each sentence in the document
                similarities = 1 - cdist(doc_embeddings, second_embeddings, metric='cosine')

                # Find top N similarities for each sentence in the document
                for idx, sim_vector in enumerate(similarities):
                    top_indices = heapq.nlargest(top_n, range(len(sim_vector)), key=sim_vector.__getitem__)

                    for j in top_indices:
                        top_entry = {
                            "witness_document": doc_sentences[idx]['document'],
                            "witness_ID": doc_sentences[idx]['ID'],
                            "witness_text": doc_sentences[idx]['text'],
                            "lhic_document": second_sentences[j]['document'],
                            "lhic_ID": second_sentences[j]['ID'],
                            "lhic_text": second_sentences[j]['text'],
                            "similarity_score": sim_vector[j]
                        }
                        outfile.write(json.dumps(top_entry) + '\n')

                # Reset for the next document
                current_doc = doc_id
                doc_sentences = [entry]
                doc_embeddings = [first_embeddings[i]]

        # Handle the last document
        if doc_sentences:
            doc_embeddings = np.array(doc_embeddings)
            similarities = 1 - cdist(doc_embeddings, second_embeddings, metric='cosine')

            for idx, sim_vector in enumerate(similarities):
                top_indices = heapq.nlargest(top_n, range(len(sim_vector)), key=sim_vector.__getitem__)

                for j in top_indices:
                    top_entry = {
                        "witness_document": doc_sentences[idx]['document'],
                        "witness_ID": doc_sentences[idx]['ID'],
                        "witness_text": doc_sentences[idx]['text'],
                        "lhic_document": second_sentences[j]['document'],
                        "lhic_ID": second_sentences[j]['ID'],
                        "lhic_text": second_sentences[j]['text'],
                        "similarity_score": sim_vector[j]
                    }
                    outfile.write(json.dumps(top_entry) + '\n')

    print(f"Top {top_n} similar sentences saved to {output_path}")

# Example usage
get_top_similar_sentences(
    "Data/wc_embeddings.npy", 
    "Data/Zenodo/witness_corpus.jsonl", 
    "Data/lhic_embeddings.npy", 
    "Data/Zenodo/lhic.jsonl", 
    "Data/top_similar_sentences.jsonl"
)