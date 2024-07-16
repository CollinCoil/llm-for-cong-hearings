'''
This file contains code used for extended pre-training and fine tuning of a sentence transformers model. 
It needs a pretrained model (which can be found on https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models)
and the pretraining or fine tuning corpora generated from using the generate_trainig_corpora.py file. 

'''

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
import numpy as np
import torch
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




def fine_tune_model(model:str, data):
    # Load the pre-trained model
    model = SentenceTransformer(model)

    # Load the fine-tuning data
    data = np.load('Data/finetuning_data.npy')

    # Create a list of InputExample from the data
    examples = []
    for anchor, positive, negative in data:
        examples.append(InputExample(texts=[anchor, positive, negative], label=0.0))

    # Create a DataLoader for the examples
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)

    # Define the loss function
    loss_function = losses.TripletLoss(model=model)

    # Set up a warmup step for the optimizer
    warmup_steps = int(len(train_dataloader) * 0.1)  # Warmup for 10% of the training data

    # Configure the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-8)

    # Schedule the learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-5, total_steps=len(train_dataloader), epochs=5, steps_per_epoch=len(train_dataloader), pct_start=0.3
    )

    # Fine-tune the model
    model.fit(train_objectives=[(train_dataloader, loss_function)], epochs=5, warmup_steps=warmup_steps, optimizer=optimizer, scheduler=scheduler)

    # Save the fine-tuned model
    model.save('Models/finetuned_model')



pretraining_data = np.load("Data/abstracts.npy")
finetuning_data = np.load("Data/finetuning_data.npy")
model = SentenceTransformer("all-distilroberta-v1")
pretrain_model(model, pretraining_data)