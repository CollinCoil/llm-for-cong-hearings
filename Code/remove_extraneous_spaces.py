"""
This python code loops through text files in a directory and removes extraneous spaces that might have been added from the pdf to text file conversion. 
"""

import enchant
import os


d = enchant.Dict("en_US")

def remove_spaces(text: str):
    """
    Removes spaces within words while preserving spaces between words.

    Args: 
        text: the contents of a text file from which spaces are to be removed. 
    """

    fixed_text = []
    current_word = ""  # Store a word being constructed

    for word in text.split():
        combined_word = current_word + word

        if d.check(combined_word):
            # If combined word is valid, append it and start a new word
            current_word = combined_word
        else:
            # If combined word is not valid, append the previous word
            # and start a new word with the current one
            fixed_text.append(current_word)
            current_word = word

    # Append the last word if it exists
    if current_word:
        fixed_text.append(current_word)

    return " ".join(fixed_text)



# Def function to loop through files in directory
def clean_spaces_from_files_in_directory(directory: str):
    """
    A helper function to call remove_spaces on an entire directory of text files. 

    Args: 
        directory: a string for the directory containing the text files. 
    """
    for filename in os.listdir(directory):
        filename = directory + filename
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        cleaned_text = remove_spaces(text)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

clean_spaces_from_files_in_directory("Data/Witness Statement Txt/")

