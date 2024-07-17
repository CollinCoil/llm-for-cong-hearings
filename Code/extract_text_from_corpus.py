"""
This program allows users to extract text from pdf files. There are some issues - this extraction program cannot work with non-OCR documents. It 
also extracts unnecessary text such as footnotes, charts, and headers. Manual cleaning may be necessary. The 
extract_from_files_in_directory program takes the directory where the pdf files are stored and extracts text. Note: Only pdf files can be in that 
directory for the program to run without error. 
"""

from typing import Union
from os.path import exists
import requests
from PyPDF2 import PdfReader
import os
import re


def text_extraction(pdf_name: str, pages: Union[int, list] = None, 
                     file_base_name: str = "text"):
  """
  Extracts text from PDF files. This text must be in a digitally native PDF or PDF that has had OCR performed. 

  Args: 
    pdf_name: file name from which text should be extracted
    pages: page number or list of pages from which text should be removed. 
    file_base_name: name for the output txt file
  """
  
  # Checks if pdf file is in directory, downloads if not in directory
  reader = PdfReader(pdf_name)
 
  # User wants all images from all pages
  if pages == None:
    for i in range(len(reader.pages)):
      page = reader.pages[i]
      text = page.extract_text()
      with open(f'{file_base_name}.txt', 'a', encoding='utf-8') as f:
        f.write(text)

  # User wants images from one page:
  if type(pages) == int:
    for i in range(len(reader.pages)):
      if i+1 == pages:
        page = reader.pages[i]
        text = page.extract_text()
        with open(f'{file_base_name}.txt', 'a') as f:
          f.write(text)
    
    
  

  # User wants images from list of pages: 
  if type(pages) == list:
    for i in range(len(reader.pages)):
      if i+1 in pages:
        page = reader.pages[i]
        text = page.extract_text()
        with open(f'{file_base_name}.txt', 'a') as f:
          f.write(text)



# Def function to loop through files in directory
def extract_from_files_in_directory(directory: str):
    """
    A helper function to apply text_extraction to all text files in a directory. 

    Args: 
      directory: the path of the directory containing the PDF files. 
    """
    for filename in os.listdir(directory):
      filename = "Data/Testimonies/" + filename
      output_name = re.sub(".pdf", ".txt", filename)
      text_extraction(filename, file_base_name = output_name)


extract_from_files_in_directory("Data/Testimonies")
      