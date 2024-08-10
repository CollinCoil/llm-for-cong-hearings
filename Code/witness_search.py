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
  with open(sentences_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()  # Read all lines once
    
  with open(output_file, 'w', encoding='utf-8') as out:
      for witness in witness_list:
          witness_parts = witness.split()  # Split witness name by spaces for individual word search
          
          for line in lines:
              data = json.loads(line)
              document = data['document']
              text = data['text']
              
              # Search for each part of the witness name using regular expression
              match = all(re.search(part, text, re.IGNORECASE) for part in witness_parts)
              
              if match:
                  # If all parts are found, write witness, document, and sentence to output
                  output_line = {"witness": witness, "document": document, "text": text}
                  json.dump(output_line, out, ensure_ascii=False)
                  out.write('\n')

# Example usage
witness_data = pd.read_csv("Data/witness_names.csv")
witness_list = witness_data["Witness"]
sentences_file = "Data/relevant_speeches_110_sentences.jsonl"  # Replace with your actual file path
output_file = "Data/Witness References/witness_direct_references_110_floor_speeches_retry.jsonl"
witness_search(witness_list, sentences_file, output_file)
