"""
This program cleans some of the most common errors introduced in the text extraction process that are too tedious to manually clean. 
These errors include double spacing and inclusion of new line characters in the middle of sentences. This code performs cleaning
on directories of txt files, removing these errors. This function prepares text for process_text.py.
"""

import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List

def clean_text_file(filepath: str) -> None:
    """
    This function reads a text file and cleans many problematic elements in text. The cleaned content is saved back to the same file.

    Args:
        filepath: A string representing the path of the text file to be processed.
    """
    # Read the content of the file
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace new line characters and tabs with spaces.
    content = content.replace('\n', ' ').replace('\t', ' ')

    # Replace common UTF-8 characters with plain text equivalents. Also replace some common useless characters. 
    replacements = {
        'Mr.': 'Mr',
        'Mrs.': 'Mrs',
        'Ms.': 'Ms',
        'Dr.': 'Dr',
        '\u00a0': ' ',  # Non-breaking space
        '\u2019': "'",  # Right single quotation mark (apostrophe)
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2010': '-',  # Hyphen
        '\u2022': '', # Remove bullet point
        '\u00B6': '', # Remove paragraph symbol
        '\u2761': '', # Remove curved paragraph symbol
        '\u00A7': '', # Remove section symbol 
        '\u2026': '', # Remove ...
        '. . .': ' ', # Remove ...
        '...': ' ',
        '[': '',
        ']': '',
        '*': ''
    }

    # Apply the replacements
    for utf8_char, replacement in replacements.items():
        content = content.replace(utf8_char, replacement)
    

    # Use regular expression to replace multiple spaces with a single space
    content = re.sub(r' {2,}', ' ', content)

    # Write the cleaned content back to the file
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)

def clean_txt_files_in_parallel(directory: str) -> None:
    """
    This function loops through all text files in a given directory and processes them in parallel.
    Each text file is cleaned by replacing UTF-8 characters, new line characters with spaces, 
    reducing double spaces, and replacing non-breaking spaces with regular spaces.

    Args:
        directory: A string representing the path of the directory containing .txt files.
    """
    # List of all .txt files in the directory
    txt_files: List[str] = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.txt')]

    # Use ThreadPoolExecutor to process the files in parallel
    with ThreadPoolExecutor() as executor:
        # Map the clean_text_file function to the list of text files
        executor.map(clean_text_file, txt_files)

# Example usage
directory_path = r'/path/to/your/directory'
clean_txt_files_in_parallel(directory_path)



def clean_and_reindex_jsonl(input_file: str, output_file: str, filter_words: list, min_word_count: int = 0) -> None:
    """
    This function reads a JSON lines file and fixes many issues impacting the semantic similarity calculation. It eliminates small sentences from the dataset, 
    and it removes boilerplate sentences (e.g., "Thank you.") by eliminating sentences with containing common words in those sentences. It outputs a new, cleaned 
    and reindexed JSONL file. 

    Args:
        input_file: A string representing the path of the JSON lines file to be processed.
        output_file: A string representing the path of the JSON lines file to be output.
        filter_words: A list of strings containing the words appearing in boilerplate sentences.
        min_word_count: An integer specifying the minimum number of words that should appear in a sentence. 
    """

    cleaned_data = []
    
    # Convert filter words to lowercase for case-insensitive comparison
    filter_words = [word.lower() for word in filter_words]
    
    # Read the input JSON Lines file
    with open(input_file, 'r', encoding = "utf-8") as infile:
        for line in infile:
            entry = json.loads(line)
            text = entry['text'].lower()

            # Filter based on words and minimum word count
            if (not any(word in text for word in filter_words)) and (len(text.split()) >= min_word_count):
                cleaned_data.append(entry)
    
    # Reindex the IDs
    for idx, entry in enumerate(cleaned_data, start=1):
        entry['ID'] = idx

    # Write the cleaned data to a new JSON Lines file
    with open(output_file, 'w', encoding = "utf-8") as outfile:
        for entry in cleaned_data:
            outfile.write(json.dumps(entry) + '\n')

# Example usage
filter_words = ["thank", "time", "discussion", "opportunity"]
min_word_count = 8
clean_and_reindex_jsonl(r'Data\Zenodo\lhi_corpus.jsonl', r'Data\Zenodo\lhi_corpus_clean.jsonl', filter_words, min_word_count)


