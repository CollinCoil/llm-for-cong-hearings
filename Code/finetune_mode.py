'''
This file contains code used for fine tuning of a pre-trained sentence transformers model. 
It needs a pretrained model (which is the output of pretrain_model.py)
fine tuning corpora generated from using the generate_trainig_corpora.py file. 

'''

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses
from torch.utils.data import DataLoader
import numpy as np
import torch
import nltk
nltk.download('punkt')


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



finetuning_data = np.load("Data/finetuning_data.npy")
model = "Model/pretrained_roberta"
fine_tune_model(model, finetuning_data)