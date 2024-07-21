"""
This function takes a directory of txt files and processes it for analysis. Specifically, it 
    (1) Reads all text files;
    (2) Split the text files by sentence;
    (3) Save the sentences in a pandas dataframe with information about which document the sentence came from;
    (4) Write the pandas dataframe as a json lines file.

This output is then used in generate_embeddings.py and witness_search.py. 
"""
import os
import pandas as pd
import json
from nltk.data import PunktSentenceTokenizer

def process_text(text_directory: str, output_filename: str = "Data/output.jsonl") -> None:
  """
  Processes a directory of text files for analysis using NLTK PunktSentenceTokenizer.

  Args:
    text_directory: The directory containing the text files.
    output_filename: The name of the output JSON lines file (default: "output.jsonl").
  """
  sentences = []
  tokenizer = PunktSentenceTokenizer()  # Initialize sentence tokenizer
  id = 0

  for filename in os.listdir(text_directory):
    if filename.endswith(".txt"):
      filepath = os.path.join(text_directory, filename)
      with open(filepath, "r") as file:
        text = file.read()
      document_sentences = tokenizer.tokenize(text)  # Tokenize text using NLTK
      for sentence in document_sentences:
        sentence = sentence.strip()  # Remove leading/trailing whitespace
        if sentence:  # Avoid empty sentences
          sentences.append({"ID": id, "document": filename, "text": sentence})
          id += 1

  # Create pandas dataframe
  df = pd.DataFrame(sentences)

  # Write dataframe as json lines
  with open(output_filename, "w") as outfile:
    for row in df.itertuples(index=False):
      json.dump(row, outfile)
      outfile.write("\n")

# Example usage
text_directory = "Data/text"
process_text(text_directory)



## TODO: Write code to process the speeches in 20 word chunks instead of by sentence. 

