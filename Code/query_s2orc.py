"""
The following code was adapted from https://github.com/allenai/s2-folks/blob/main/examples/python/search_bulk/get_dataset.py
The goal is to output a json file containing the titles and abstracts of papers related to federal health insurance. This code retrieves information on 96396 papers. 
The abstracts in this json file will be integral to extend the pretraining of the sentence transformer model. 
"""

import requests
import json


def query_s2orc():
    """
    This function calls the S2ORC API to request papers associated with a query. Articles containing any of the queries are returned. About 90,000 articles are returned using the below query. 
    The queried papers are then saved as a json lines file. 
    """
    # | means "or", + means "and", and () searches for that set of words
    query = 'Obamacare | (Affordable Care Act) | PPACA | Medicare | Medicaid | (CHIP + Obama) | (ACA + Obama) | (marketplace + health + insurance) | (Skinny Repeal) | (Trump + Health + Insurance) | (Obama + Health + Insurance)'
    fields = 'title,publicationTypes,publicationDate,abstract'
    
    # The year in the following url may need to be changed if articles from a different time are desired
    url = f"http://api.semanticscholar.org/graph/v1/paper/search/bulk?query={query}&fields={fields}&year=2007-"
    r = requests.get(url).json()
    
    print(f"Will retrieve an estimated {r['total']} documents")
    retrieved = 0
    
    # Save articles in a json lines file
    with open(f"Data/papers.jsonl", "a") as file:
        while True:
            if "data" in r:
                retrieved += len(r["data"])
                print(f"Retrieved {retrieved} papers...")
                for paper in r["data"]:
                    print(json.dumps(paper), file=file)
            if "token" not in r:
                break
            r = requests.get(f"{url}&token={r['token']}").json()
    
    print(f"Done! Retrieved {retrieved} papers total")

query_s2orc()