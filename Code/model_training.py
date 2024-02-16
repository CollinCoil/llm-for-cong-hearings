'''
This file contains code used for extended pre-training and fine tuning of a sentence transformers model. 
It needs a pretrained model (which can be found on https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models)
and the pretraining corpora generated from using the generate_pretrainig_corpora.py file. 

'''
# TODO: Add methods for model extended pretraining and fine tuning

from sentence_transformers import SentenceTransformer, util

def pretrain_model(model, data):
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    pass



def fine_tune_model(model, data):
    pass

