'''
This file contains code used for extracting relevant floor speeches from the legislative record. The raw data for these can be found at 
https://data.stanford.edu/congress_text which is a Stanford University project that processes text from the US Congressional Record. 
I used the daily edition of HeinOnline to get floor speeches from the 110th-114th Congresses. That data is not perfectly delimited by the
| symbol, so some work may need to be done to correct these errors. 

'''

import pandas as pd

def filter_speeches(Congress: str):
    speeches_path = "Data/speeches_%s.txt" % Congress
    description_path = "Data/descr_%s.txt" % Congress
    speeches = pd.read_csv(speeches_path, sep='|', header=0)
    descriptions = pd.read_csv(description_path, sep='|', header=0)

    # Filtering out irrelevant speeches based on description
    ## Removing speeches made by non-members
    member_descriptions = descriptions[descriptions['gender'] != ("Special")]

    ## Filtering short speeches (those less than one minute long, about 120 words)
    member_descriptions_long = member_descriptions[member_descriptions["word_count"] > 120]

    # Using those descriptions to filter the speech data frame
    mask = speeches['speech_id'].isin(member_descriptions_long['speech_id'])
    speeches = speeches[mask]

    # Filtering out speeches unreltaed to Affordable Care Act
    relevant_words = {"healthcare", "health insurance", "medicaid eligibility", "patient protection", "hospital", "obamacare",
                     "ppaca", "aca", "marketplace", "affordable care act", "individual mandate",
                     "health care", "employer mandate", "loss ratio", "risk corridor"}
    mask = speeches['speech'].apply(lambda x: any(word in x.lower().split() for word in relevant_words))
    speeches = speeches[mask]

    print("You have selected %d speeches from the %sth Congress." %(speeches.shape[0], Congress))
    return(speeches)



# TODO: Fill out this function to save the speeches in a json file with some metadata (date, speaker)
def save_speeches():
    pass

congresses = ["110", "111", "112", "113", "114"]

for term in congresses: 
    filter_speeches(term)