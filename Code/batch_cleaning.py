"""
This program cleans some of the most common errors introduced in the text extraction process that are too tedious to manually clean. 
These errors include double spacing and inclusion of new line characters in the middle of sentences. This code performs cleaning
on directories of txt files, removing these errors. This function prepares text for process_text.py.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

def clean_text_file(filepath: str) -> None:
    """
    This function reads a text file, replaces UTF-8 encoded characters with plain text equivalents, 
    replaces new line characters with spaces, replaces non-breaking spaces (\u00a0) with regular spaces, 
    and removes double spaces using regular expressions. The cleaned content is saved back to the same file.

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


