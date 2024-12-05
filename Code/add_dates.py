"""
This function filters a JSONL file of sentence similarities based on document dates and ensures entries 
only remain if the first document's date is less than or equal to the second document's date. It adds the 
associated dates to the metadata for each document and writes the updated entries to a new JSONL file.

The function uses metadata from two separate CSV files to retrieve the dates for documents, making it 
suitable for filtering and annotating JSONL files with date constraints.
"""

import csv
import json
import pandas as pd

def filter_jsonl_by_date(jsonl_file, output_file, speech_metadata_file, document_metadata_file):
    """
    Filters a JSONL file based on document dates from two metadata CSV files.

    Parameters:
    - jsonl_file: Path to the input JSONL file.
    - output_file: Path to the output JSONL file.
    - speech_metadata_file: Path to the CSV file containing speech metadata (integer documents).
    - document_metadata_file: Path to the CSV file containing document metadata (string documents).
    """
    # Load CSV files into pandas DataFrames
    speech_metadata = pd.read_csv(speech_metadata_file)
    document_metadata = pd.read_csv(document_metadata_file)

    # Convert date columns in both CSVs to datetime format for easier comparison
    speech_metadata['date'] = pd.to_datetime(speech_metadata['date'], format='%Y%m%d')
    document_metadata['date'] = pd.to_datetime(document_metadata['date'], format='%Y%m%d')

    # Helper function to get the date for a given document
    def get_date_for_document(document):
        if isinstance(document, int):  # If document is an integer, lookup in speech_metadata
            result = speech_metadata[speech_metadata['document'] == document]
            if not result.empty:
                return result.iloc[0]['date']
        else:  # If document is text, lookup in document_metadata
            result = document_metadata[document_metadata['document'] == document]
            if not result.empty:
                return result.iloc[0]['date']
        return None  # Return None if the document is not found

    # Open and filter the JSONL file while adding dates to metadata
    with open(jsonl_file, 'r', encoding="utf-8") as infile, open(output_file, 'w', encoding="utf-8") as outfile:
        for line in infile:
            entry = json.loads(line)

            # Extract document values from metadata
            first_document = entry['first_metadata']['document']
            second_document = entry['second_metadata']['document']

            # Get the corresponding dates for both documents
            first_date = get_date_for_document(first_document)
            second_date = get_date_for_document(second_document)

            # If both dates are valid, compare them
            if first_date is not None and second_date is not None:
                if first_date <= second_date:  # Keep entry only if first date is less than or equal to second date
                    # Add dates to metadata
                    entry['first_metadata']['date'] = first_date.strftime('%Y%m%d')
                    entry['second_metadata']['date'] = second_date.strftime('%Y%m%d')

                    # Write the updated entry to the output file
                    outfile.write(json.dumps(entry) + '\n')
            else:
                # In case one or both dates are missing, you may decide to keep or skip the entry.
                # Here, we'll skip the entry if dates are missing
                if first_date is None:
                    print(f"{first_document} missing date")
                if second_date is None:
                    print(f"{second_document} missing date")

# Example Usage 
filter_jsonl_by_date(
    jsonl_file=r'path\to\jsonl',
    output_file=r'path\to\output',
    speech_metadata_file=r'path\to\speech\metadata',
    document_metadata_file=r'path\to\document\metadata'
)