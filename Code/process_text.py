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
from nltk.tokenize import PunktSentenceTokenizer
import json
import string

def process_text(text_directory: str, output_filename: str = "Data/output.jsonl") -> None:
  """
  Processes a directory of text files for analysis using NLTK PunktSentenceTokenizer.

  Args:
    text_directory: The directory containing the text files.
    output_filename: The name of the output JSON lines file (default: "output.jsonl").
  """
  sentences = []
  tokenizer = PunktSentenceTokenizer()  # Initialize sentence tokenizer


  for filename in os.listdir(text_directory):
    id = 0
    if filename.endswith(".txt"):
      filepath = os.path.join(text_directory, filename)
      with open(filepath, "r", encoding='utf-8') as file:
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
  with open(output_filename, "w", encoding='utf-8') as outfile:
    for row in df.itertuples(index=False):
      json.dump(row, outfile)
      outfile.write("\n")

# Example usage
text_directory = "Data/text"
# process_text(text_directory)



def process_floor_speeches(input_json: str, output_json: str) -> None:
    """
    Processes a JSON file containing speeches by stripping punctuation, splitting speeches into
    15-word sentences, and saving the processed data into a new JSON file.

    Args:
        input_json: Path to the input JSON file containing the speeches.
        output_json: Path to the output JSON file where processed sentences will be saved.
    """
    
    # Load the input JSON file
    sentences = []

    with open(input_json, 'r', encoding='utf-8') as infile:
        for line in infile:
            entry = json.loads(line.strip())  # Load each JSON object line by line
            
            speech_id = entry.get("speech_id")
            speech = entry.get("speech", "")
            
            # Remove punctuation
            speech = speech.translate(str.maketrans('', '', string.punctuation))
            
            # Split the speech into words
            words = speech.split()
            
            # Break the speech into 15-word sentences
            sentence_id = 0
            for i in range(0, len(words), 15):
                sentence_words = words[i:i + 15]
                sentence = ' '.join(sentence_words)
                sentences.append({
                    "sentence_id": sentence_id,  # Unique ID for each sentence
                    "speech_id": speech_id,
                    "sentence": sentence
                })
                sentence_id += 1

    # Save the processed sentences to the output JSON file
    with open(output_json, 'w', encoding='utf-8') as outfile:
        json.dump(sentences, outfile, indent=4)


process_floor_speeches(input_json = "Data/relevant_speeches_110.jsonl", output_json = "Data/relevant_speeches_110_sentences.jsonl")