'''
This program is designed to extract text from a specified range of pages in a PDF file, enhancing image contrast and performing OCR on images.
This is to be used for non-digital-native PDFs, or documents that have been scanned into PDF format. These documents (or parts of documents)
can be identified text cannot be highlighted. It loops over the selected pages of an input file, extracts the images the text is stored in, 
and performs optical character recognition (OCR) to extract the text. 
'''

import fitz
from PIL import Image, ImageEnhance
import pytesseract
import io

def extract_text_from_pdf(pdf_file, lower_page, upper_page, output_text_file="extracted_text.txt"):
    """
    param pdf_file: The path to the PDF file.
    param lower_page: The starting page (1-based index).
    param upper_page: The ending page (1-based index, inclusive).
    param output_text_file: The path to the output text file.
    """
    # Open the PDF file
    doc = fitz.open(pdf_file)

    # Create or open the output text file in write mode
    with open(output_text_file, 'w') as txt_file:
        # Loop through each page in the specified range
        for page_number in range(lower_page-1, upper_page):
            page = doc.load_page(page_number)
            
            # Get images on the page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Open the image with PIL
                image = Image.open(io.BytesIO(image_bytes))
                
                # Enhance the contrast of the image
                enhancer = ImageEnhance.Contrast(image)
                enhanced_image = enhancer.enhance(3)
                
                # Perform OCR on the enhanced image
                extracted_text = pytesseract.image_to_string(enhanced_image)
                
                # Write the extracted text to the output file
                txt_file.write(extracted_text + "\n")


extract_text_from_pdf("trial_hearing.pdf", 73, 122)