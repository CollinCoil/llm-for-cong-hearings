"""
This file contains code used for fine tuning of a pre-trained sentence transformers model. 
It needs a pretrained model (which is the output of pretrain_model.py)
fine tuning corpora generated from using the generate_trainig_corpora.py file. 
"""

import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import TripletLoss, TripletDistanceMetric
from torch.utils.data import DataLoader



def read_triplet_data(file_path):
    """
    Reads triplet data from a text file where each line is a separate triplet example.
    
    Args:
        file_path: The path to the text file containing the triplet data.
    
    Returns:
        A list of tuples, each containing three strings (anchor, positive, negative).
    """
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            anchor, positive, negative = line.strip().split('|')
            triplets.append((anchor, positive, negative))
    return triplets



def fine_tune_model(base_model_path: str, data_path: str, output_model_path: str):
    """
    This takes a pretrained sentence transformer model and a fine-tuning dataset to fine tune the model. It uses the triplet loss function, so training data must be set up as 
    (anchor sentence, similar sentence, dissimilar sentence). Once the model is fine tuned, it is saved. 

    Args: 
        model_path: a string of the directory where the model is stored
        data_path: the fine-tuning data set
    """
    # Check if a GPU is available and print it out
    if torch.cuda.is_available():
        device = 'cuda'
        print("GPU is available. Training on GPU.")
    else:
        device = 'cpu'
        print("GPU is not available. Training on CPU.")

    # Load the pre-trained model and move it to the GPU
    model = SentenceTransformer(base_model_path).to(device)

    # Load the fine-tuning data
    data = read_triplet_data(data_path)

    # Create a list of InputExample from the data
    examples = [InputExample(texts=[anchor, positive, negative]) for anchor, positive, negative in data]

    # Create a DataLoader for the examples
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=128, num_workers=16)

    # Define the loss function with cosine distance
    loss_function = TripletLoss(
        model=model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=0.3  # Use a smaller margin for cosine distance
    )

    # Fine-tune the model on the GPU
    model.fit(train_objectives=[(train_dataloader, loss_function)],
              epochs=3,
              scheduler="warmuplinear",
              optimizer_params={"lr": 5e-5},
              show_progress_bar=True,
              output_path=output_model_path)

    # Save the fine-tuned model
    model.save(output_model_path)


# Example Usage
if __name__ == '__main__':
    base_model_path = r"path\to\model"
    data_path = r"path\to\data"
    output_model_path = r"Models\finetuned_model"
    fine_tune_model(base_model_path, data_path, output_model_path)