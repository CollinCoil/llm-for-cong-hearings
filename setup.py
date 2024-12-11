from setuptools import setup, find_packages

# Function to read requirements from the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name='llm-for-cong-hearings',
    version='1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    include_package_data=True,
    description='This repository contains code, pretrained models, and explanations for how to use large language models and a variety of other natural language processing techniques to analyze congressional hearings.',
    author='Collin Coil',
    author_email='collin.a.coil@gmail.com',
    url='https://github.com/CollinCoil/llm-for-cong-hearings',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='==3.12',  # While other versions of python may work, this was originally developed with 3.12.1, and all specified dependencies work with python==3.12.1
)
