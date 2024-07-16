"""
This function searches for the names of witnesses within a corpus of documents. It returns a json lines file 
with the witness name, the document, and the sentence. This provides a low-tech way to search for direct references
to witnesses in the corpus. 
"""

import json
import re
import pandas as pd

def witness_search(witness_list:list, sentences_file:str, output_file:str="Data/output.jsonl"):
  """
  This function searches for witness names in a JSON Lines file of sentences.

  Args:
      witness_list: A list containing strings of witness names.
      sentences_file: Path to a JSON Lines file containing documents and text.
      output_file: Path to the output JSON Lines file (default: output.jsonl).
  """
  with open(sentences_file, 'r') as f, open(output_file, 'w') as out:
    for witness in witness_list:
      # Split witness name by spaces for individual word search
      witness_parts = witness.split()
      for line in f:
        data = json.loads(line)
        document = data['document']
        text = data['text']
        # Search for each part of the witness name using regular expression
        match = any(re.search(part, text, re.IGNORECASE) for part in witness_parts)
        if match:
          # If any part is found, write witness, document, and sentence to output
          output_line = {"witness": witness, "document": document, "text": text}
          json.dump(output_line, out, ensure_ascii=False)
          out.write('\n')

# Example usage
witness_data = pd.read_csv("Data/witness_data.csv")
witness_list = witness_data["Witness"]
sentences_file = "Data/sentences.jsonl"  # Replace with your actual file path
witness_search(witness_list, sentences_file)
