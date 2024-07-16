"""
This function calculates the cosine similarity between all embeddings in the witness corpus and the LHIC. It does this
by calculating the cosine similarity between each sentence. The output of this function is a similarity matrix where the 
rows correspond with a sentence from the witness corpus and the columns correspond with a sentence from the LHIC. 
This semantic similarity matrix can then be sorted by witness to determine which sentences they said are most similar to 
sentences in the LHIC. 
"""

import numpy as np
from scipy.spatial.distance import cdist


def semantic_similarity(witness_embeddings_path="Data/witness_embeddings.npy", lhic_embeddings_path="Data/lhic_embeddings.npy", output_path="Data/similarity_matrix.npy"):
  """
  Calculates cosine similarity between sentence embeddings in two NumPy files.

  Args:
    witness_embeddings_path: Path to the first NumPy file containing sentence embeddings (default: "witness_embeddings.npy").
    lhic_embeddings_path: Path to the second NumPy file containing sentence embeddings (default: "lhic_embeddings.npy").
    output_path: Path to save the cosine similarity matrix as a NumPy array (default: "similarity_matrix.npy").
  """

  # Load the sentence embeddings
  witness_embeddings = np.load(witness_embeddings_path)
  lhic_embeddings = np.load(lhic_embeddings_path)

  # Calculate cosine similarity matrix using efficient distance function
  similarity_matrix = 1 - cdist(witness_embeddings, lhic_embeddings, metric='cosine')

  # Save the similarity matrix
  np.save(output_path, similarity_matrix)

  print(f"Similarity matrix saved to: {output_path}")

# Example usage
semantic_similarity()