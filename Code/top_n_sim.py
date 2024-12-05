"""
This function processes a JSONL file containing sentence similarity data. It calculates the average similarity 
and count of sentence pairs for each document in the dataset. The results are written to a CSV file, which includes 
the document name, average similarity score, and count of sentence pairs. 

Optionally, the function can filter the output to include only the top N documents with the highest average similarity scores. 
This is useful for summarizing or analyzing documents with high similarity within a corpus.
"""

import json
import csv
from collections import defaultdict
from typing import Optional
import heapq


def process_similarity_data(input_file: str, output_file: str, top_n: Optional[int] = None) -> None:
    """
    Processes a JSONL file containing sentence similarity data, calculates average similarity 
    and count for each document, and writes the results to a CSV file.
    
    Parameters:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output CSV file.
        top_n (Optional[int]): If specified, only include the top N documents by average similarity.
    """
    # Initialize dictionary to store sums and counts
    document_data = defaultdict(lambda: {'similarity_sum': 0, 'count': 0})

    # Read JSONL file and aggregate similarity data
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            first_metadata = entry["first_metadata"]
            similarity = entry["similarity"]
            
            # Get the document name from first_metadata
            document = first_metadata["document"]
            
            # Update similarity sum and count for the document
            document_data[document]['similarity_sum'] += similarity
            document_data[document]['count'] += 1

    # If `top_n` is specified, find the top N documents by average similarity
    if top_n:
        document_data = heapq.nlargest(
            top_n,
            document_data.items(),
            key=lambda item: item[1]['similarity_sum'] / item[1]['count']
        )
    else:
        document_data = document_data.items()

    # Write the results to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['document', 'average_similarity', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for document, data in document_data:
            average_similarity = data['similarity_sum'] / data['count']  # Calculate average
            writer.writerow({
                'document': document,
                'average_similarity': average_similarity,
                'count': data['count']
            })

    print(f"Results saved to {output_file}")


# Example usage:
# process_similarity_data(
#     input_file='path/to/input',
#     output_file='path/to/output',
#     top_n=20
# )