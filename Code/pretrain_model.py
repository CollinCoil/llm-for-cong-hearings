"""
This file contains code used for extended pre-training of a sentence transformers model. 
It needs a pretrained model (which can be found on https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models)
and the pretraining corporus generated from using the generate_training_corpora.py file. 
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers import datasets,  losses
from torch.utils.data import DataLoader
import numpy as np

# The following code continues pretraining of a pretrained model with TSDAE, 
# See https://www.sbert.net/examples/unsupervised_learning/TSDAE/README.html

def pretrain_model(model: str, data, output_path: str):
    """
    This takes a pretrained sentence transformer model and a pretraining dataset to extend pretraining of the model. This is done to further expose the model to 
    the unqiue vocabulary in the primary data set. In our project, we use abstracts on articles about Congress and the Affordable Care Act to increase the model's awareness
    of words associated with the Affordable Care act. The model is saved once the extended pretraining is complete. 

    Args: 
        model: a string of the directory where the model is stored
        data: the fine-tuning data set
    """
    model_name = "pretrained_roberta"

    # Create the special denoising dataset that adds noise on-the-fly
    train_dataset = datasets.DenoisingAutoEncoderDataset(data)

    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)

    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(
        model, decoder_name_or_path=model_name, tie_encoder_decoder=True
        )
    
    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        scheduler="warmuplinear",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
        output_path="pretrained_roberta"
    )

    model.save(output_path)




# Example Usage
# with open("Data/abstracts.txt", 'r', encoding='utf-8') as file:
#         data = file.readlines()
#         pretraining_data = [line.strip() for line in data]
# model = SentenceTransformer("all-distilroberta-v1")
# output_path = "Models/pretrained_roberta"
# pretrain_model(model, pretraining_data, output_path)