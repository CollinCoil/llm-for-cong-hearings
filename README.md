This repository contains a variety of programs used in the paper **Congressional Witnesses Matter: Proving Witness Testimony Impact Using Large Language Models**. These techniques are infrequently used in political science, so they are intended to be as user-friendly as possible, documenting inputs, outputs, and interconnections between the programs. Furthermore, we provide some clean text files to practice using these models. Code for pre-training, fine-tuning, and using sentence transformer models are provided in this repository. Additionally, code to curate datasets are provided, which includes functionality to extract text from a variety of documents and prepare the text for analysis. Other scripts are provided for a variety of tasks, including using regular expressions and keyword extraction. 

# Setup
### Step 1: Set Up a Conda Environment
It is recommended to use Python 3.12.1 for this project to ensure package compatability. Otherwise, additional effort will need to be done to resolve dependency issues. To set and activate up a conda environment, run the following commands:

```bash
conda create -n llm_for_congress python=3.12.1
conda activate llm_for_congress
```

### Step 2: Install Microsoft Visual C++ Redistributable
Some of the code may require the Microsoft Visual C++ Redistributable. You can download and install it [here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

### Step 3: Install Tesseract
You will need to install Tesseract before running the setup file. Tesseract is necessary to perform OCR with pytesseract. If you are not performing OCR, you can remove pytesseract from the `requirements.txt` file. Instructions for Tesseract installation can be found [here](https://tesseract-ocr.github.io/tessdoc/Installation.html).

### Step 4: Run the Setup Script
After setting up the conda environment and installing the necessary dependencies, navigate to the root directory of this repository and run the following command:

```bash
pip install -v -e .
```
This command will install all the required packages listed in the requirements.txt file.

# Usage
We recommend creating a python notebook and importing the necessary functions from across the repository. That will prevent the need to toggle in between individual files when executing code. 

# Data
The data for this project is accessible on Zenodo. The link will be published here once it is available. 

# Paper and Citation
If you use this code, please use the following citation: 

```
@article{Coil2024Congressional,
  title={Congressional Witnesses Matter: Proving Witness Testimony Impact Using Large Language Models},
  author={Coil, Collin and Brucker, Caroline and Chen, Nicholas and Keith, Elizabeth and O'Connor, Karen},
  journal={Unpublished manuscript},
  year={2024}
}
```
A link to the paper will be published here once it is available. 
