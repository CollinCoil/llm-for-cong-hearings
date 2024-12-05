"""
This function merges multiple JSONL files into a single JSONL file, re-indexing the IDs to ensure they are sequential 
across the merged dataset. It takes a list of input file paths and creates an output file with updated ID fields.

This is particularly useful for consolidating JSONL files with overlapping or non-sequential IDs into a single, 
clean dataset. Each record in the merged file will have a unique and sequential ID, starting from 0.
"""

import json

def merge_jsonl_files(file_list, output_file):
    """
    Merges a list of JSONL files into one JSONL file with re-indexed IDs.

    Parameters:
    - file_list: List of paths to the JSONL files to merge.
    - output_file: Path to the output JSONL file.
    """
    new_id = 0  # Initialize the new ID counter
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # Parse the JSON line
                    data = json.loads(line)
                    
                    # Update the ID to the new re-indexed value
                    data["ID"] = new_id
                    
                    # Write the updated line to the output file
                    outfile.write(json.dumps(data) + '\n')
                    
                    # Increment the new ID counter
                    new_id += 1

# Example usage:
# jsonl_files = [r"path\to\json_1", r"path\to\json_2",]
# output_file = r"path\to\output"
# merge_jsonl_files(jsonl_files, output_file)
