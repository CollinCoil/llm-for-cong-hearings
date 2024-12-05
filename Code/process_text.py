"""
This function takes a directory of txt files and processes it for analysis. Specifically, it 
    (1) Reads all text files;
    (2) Split the text files by sentence;
    (3) Save the sentences as a json lines file.

This output is then used in generate_embeddings.py and witness_search.py. 
"""
import os
import pandas as pd
import json
from nltk.tokenize import PunktSentenceTokenizer
import json
import string

def process_text(text_directory: str, output_jsonl: str = "Data/output.jsonl") -> None:
    """
    Processes a directory of text files for analysis using NLTK PunktSentenceTokenizer.

    Args:
        text_directory: The directory containing the text files.
        output_filename: The name of the output JSON lines file (default: "output.jsonl").
    """
    sentences = []
    tokenizer = PunktSentenceTokenizer()  # Initialize sentence tokenizer
    id = 0  # Will help find each sentence in the semantic textual similarity matrix

    for filename in os.listdir(text_directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(text_directory, filename)
            with open(filepath, "r", encoding='utf-8') as file:
                text = file.read()

            document_sentences = tokenizer.tokenize(text)  # Tokenize text using NLTK

            # Initialize an empty list to store merged sentences
            merged_sentences = []
            buffer_sentence = ""

            for sentence in document_sentences:
                sentence = sentence.strip()  # Remove leading/trailing whitespace

                # Split sentence into words to count them
                words = sentence.split()

                if buffer_sentence:
                    # Merge the previous short sentence with the current one
                    sentence = buffer_sentence + " " + sentence
                    buffer_sentence = ""

                # Check if sentence has fewer than 5 words
                if len(words) < 5:
                    # Save it in the buffer to merge with the next sentence
                    buffer_sentence = sentence
                else:
                    merged_sentences.append(sentence)

            # Add any leftover buffer sentence if it exists
            if buffer_sentence:
                merged_sentences.append(buffer_sentence)

            # Add sentences to the output
            for sentence in merged_sentences:
                if sentence:  # Avoid empty sentences
                    sentences.append({"ID": id, "document": filename, "text": sentence})
                    id += 1

    # Write the processed sentences to JSON lines
    with open(output_jsonl, 'w', encoding='utf-8') as outfile:
        for sentence_entry in sentences:
            json.dump(sentence_entry, outfile, ensure_ascii=False)
            outfile.write('\n')

# Example Usage
# text_directory = r"path\to\directory"
# output_jsonl = r"path\to\output"
# process_text(text_directory, output_jsonl)



def process_floor_speeches(input_jsonl: str, output_jsonl: str) -> None:
    """
    Processes a JSONL file containing speeches by stripping punctuation, splitting speeches into
    15-word sentences, and saving the processed data into a new JSONL file.

    Args:
        input_jsonl: Path to the input JSON file containing the speeches.
        output_jsonl: Path to the output JSON file where processed sentences will be saved.
    """
    
    sentences = []

    with open(input_jsonl, 'r', encoding='utf-8') as infile:
        for line in infile:
            entry = json.loads(line.strip())  # Load each JSONL object line by line
            
            speech_id = entry.get("speech_id")
            speech = entry.get("speech", "")
            
            # Remove punctuation
            speech = speech.translate(str.maketrans('', '', string.punctuation))
            
            # Split the speech into words
            words = speech.split()
            
            # Break the speech into 15-word sentences
            sentence_id = 0
            for i in range(0, len(words), 20):
                sentence_words = words[i:i + 20]
                sentence = ' '.join(sentence_words)
                sentences.append({
                    "ID": sentence_id,  # Unique ID for each sentence
                    "document": speech_id,
                    "text": sentence
                })
                sentence_id += 1

    # Save the processed sentences to the output JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as outfile:
        for sentence_entry in sentences:
            json.dump(sentence_entry, outfile, ensure_ascii=False)
            outfile.write('\n')

# Example Usage
# process_floor_speeches(input_jsonl = "Data/relevant_speeches_114.jsonl", output_jsonl = "Data/relevant_speeches_114_sentences.jsonl")