"""
This function searches for the names of witnesses within a corpus of documents. It returns a json lines file 
with the witness name, the document, and the sentence. This provides a low-tech way to search for direct references
to witnesses in the corpus. 
"""

import json
import re
import multiprocessing as mp
import pandas as pd

def search_witness_in_line(witness_parts, line):
    """
    Searches for a witness in a single line of text.
    
    Args:
        witness_parts: A list of strings representing parts of the witness name.
        line: A JSON-formatted string from the sentences file.
    
    Returns:
        A dictionary with the witness, document, and text if a match is found, otherwise None.
    """
    data = json.loads(line)
    document = data['document']
    text = data['text']
    
    match = all(re.search(part, text, re.IGNORECASE) for part in witness_parts)
    
    if match:
        return {"document": document, "text": text}
    return None

def process_witness(witness, lines):
    """
    Processes a single witness by searching through all lines.
    
    Args:
        witness: A string representing the witness name.
        lines: A list of JSON-formatted strings from the sentences file.
    
    Returns:
        A list of dictionaries with matches for the witness.
    """
    witness_parts = witness.split()
    matches = []
    for line in lines:
        result = search_witness_in_line(witness_parts, line)
        if result:
            result['witness'] = witness
            matches.append(result)
    return matches

def witness_search_parallel(witness_list, sentences_file, output_file="Data/output.jsonl", num_workers=4):
    """
    Parallelized search for witness names in a JSON Lines file of sentences.
    
    Args:
        witness_list: A list containing strings of witness names.
        sentences_file: Path to a JSON Lines file containing documents and text.
        output_file: Path to the output JSON Lines file (default: output.jsonl).
        num_workers: Number of parallel processes to use (default: 4).
    """
    with open(sentences_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Create a pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Process each witness in parallel
        results = pool.starmap(process_witness, [(witness, lines) for witness in witness_list])
    
    # Flatten the list of results
    all_matches = [match for sublist in results for match in sublist]
    
    # Write the results to the output file
    with open(output_file, 'w', encoding='utf-8') as out:
        for match in all_matches:
            json.dump(match, out, ensure_ascii=False)
            out.write('\n')

# Example usage
witness_data = pd.read_csv("Data/witness_names.csv")
witness_list = witness_data["Witness"]
sentences_file = "Data/relevant_speeches_114_sentences.jsonl"  
output_file = "Data/Witness References/witness_direct_references_114_floor_speeches.jsonl"

if __name__ == '__main__':
    witness_search_parallel(witness_list, sentences_file, output_file)
