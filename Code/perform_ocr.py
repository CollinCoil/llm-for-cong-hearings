"""
This program is designed to extract text from directories of PDFs, converting all pages to images, enhancing image contrast, 
and performing OCR on images. This is to be used for non-digital-native PDFs or PDFs that have high error when using other 
extraction tools (such as extract_text_from_corpus.py). This program loops over the pages of each file in the directory, 
converts each page to an image, and performs optical character recognition (OCR) to extract the text. 
"""

import os
import fitz
from PIL import Image, ImageEnhance
import pytesseract
import io
import concurrent.futures
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your path

def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy.

    Args:
        image: The input image.

    Returns:
        The preprocessed image.
    """
    # Enhance the contrast of the image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.25)

    return image

def postprocess_text(text):
    """
    Postprocess the text to correct errors and improve accuracy.

    Args:
        text: The input text.

    Returns:
        The postprocessed text.
    """
    # Remove leading apostrophes
    text = re.sub(r"^‘", "", text, flags=re.MULTILINE)  # These are commonly added by the program erroneously 

    return text

def extract_text_from_pdf(pdf_file: str, output_text_file: str):
    """
    This code extracts text from PDF documents that have not had OCR performed.

    Args:
        pdf_file: The path to the PDF file.
        output_text_file: The path to the output text file.
    """
    # Open the PDF file
    doc = fitz.open(pdf_file)

    # Create or open the output text file in write mode
    with open(output_text_file, 'w', encoding='utf-8') as txt_file:
        # Loop through each page in the PDF
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)

            # Render the page as a Pixmap object
            pix = page.get_pixmap(dpi=900)

            # Convert the Pixmap object to a PIL image
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Preprocess the image
            image = preprocess_image(image)

            # Perform OCR on the preprocessed image
            extracted_text = pytesseract.image_to_string(image)

            # Postprocess the text
            extracted_text = postprocess_text(extracted_text)

            # Write the extracted text to the output file
            txt_file.write(extracted_text + "\n")

def process_pdf_directory(pdf_directory: str):
    """
    Processes all PDF files in the specified directory to extract text using OCR.
    
    Args:
        pdf_directory: The path to the directory containing the PDF files.
    """
    # List to store the futures
    futures = []

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Loop through each file in the directory
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                pdf_file = os.path.join(pdf_directory, filename)
                output_text_file = os.path.splitext(pdf_file)[0] + ".txt"
                future = executor.submit(extract_text_from_pdf, pdf_file, output_text_file)
                futures.append(future)

    # Wait for all futures to complete
    concurrent.futures.wait(futures)
    print(f"Processed all PDFs in {pdf_directory}")

# Example usage:
# pdf_directory = r"path\to\directory"
# process_pdf_directory(pdf_directory)
