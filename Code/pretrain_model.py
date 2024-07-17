'''
This file contains code used for extended pre-training of a sentence transformers model. 
It needs a pretrained model (which can be found on https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models)
and the pretraining corporus generated from using the generate_trainig_corpora.py file. 

'''

from sentence_transformers import SentenceTransformer
from sentence_transformers import datasets,  losses
from torch.utils.data import DataLoader
import numpy as np
import nltk
nltk.download('punkt')

# The following code continues pretraining of a pretrained model with TSDAE, 
# See https://www.sbert.net/examples/unsupervised_learning/TSDAE/README.html

def pretrain_model(model, data):
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
        weight_decay=0,
        scheduler="warmuplinear",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
        save_best_model=True,
        output_path="pretrained_roberta"
    )

    model.save("Models/pretrained_roberta")







pretraining_data = np.load("Data/abstracts.npy")
model = SentenceTransformer("all-distilroberta-v1")
pretrain_model(model, pretraining_data)