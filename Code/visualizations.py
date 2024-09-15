"""
This code uses the semantic similarity score to make visualizations. It makes a heatmap and graph. 
"""

import networkx as nx
import pandas as pd
from pyvis.network import Network
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def create_similarity_graph(similarity_matrix, witness_sentences, document_sentences, threshold=0.5, show_labels=False):
    # Convert similarity_matrix to standard Python float types
    similarity_matrix = similarity_matrix.astype(float)

    # Create a NetworkX graph
    G = nx.Graph()

    # Add witness nodes
    for i, sentence in enumerate(witness_sentences):
        label = sentence if show_labels else None
        G.add_node(sentence, label=label, group='witness')

    # Add document nodes
    for j, sentence in enumerate(document_sentences):
        label = sentence if show_labels else None
        G.add_node(sentence, label=label, group='document')

    # Add weighted edges based on similarity score
    for i, witness_sentence in enumerate(witness_sentences):
        for j, document_sentence in enumerate(document_sentences):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                G.add_edge(witness_sentence, document_sentence, weight=similarity)

    # Set witness node size proportional to the number of connections (degree) above the threshold
    for node in G.nodes:
        if G.nodes[node]['group'] == 'witness':
            degree = G.degree(node)
            G.nodes[node]['size'] = 5 + degree
        else:
            G.nodes[node]['size'] = 10

    # Visualize the graph using PyVis
    net = Network(notebook=True, height='800px', width='100%', bgcolor='#222222', font_color='white')

    # Add nodes and edges from the NetworkX graph to PyVis
    net.from_nx(G)

    # Disable physics for a static graph (this should come after net initialization)
    net.toggle_physics(False)

    # Add a button to toggle node visibility
    net.show_buttons(filter_=['nodes'])

    # Color the nodes by group and apply size
    for node in net.nodes:
        if node['group'] == 'witness':
            node['color'] = '#FF5733'
        else:
            node['color'] = '#33FF57'
        
        node['size'] = G.nodes[node['id']]['size']

    # Show the graph in the browser or save to an HTML file
    net.show('Data/semantic_similarity_graph.html')








def create_similarity_heatmap(similarity_matrix, witness_sentences, document_sentences, transformation=None, clip_range=(0.9, 1.0)):
    # Apply transformation
    if transformation == 'log':
        transformed_matrix = log_transform(similarity_matrix)
    elif transformation == 'clip':
        transformed_matrix = clip_transform(similarity_matrix, clip_range[0], clip_range[1])
    else:
        transformed_matrix = similarity_matrix  # No transformation
    
    # Create a pandas DataFrame for easier plotting
    df = pd.DataFrame(transformed_matrix, index=witness_sentences, columns=document_sentences)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap='coolwarm', cbar_kws={'label': 'Transformed Semantic Similarity'}, linewidths=0.5)

    plt.title('Transformed Semantic Similarity Heatmap')
    plt.xlabel('Document Sentences')
    plt.ylabel('Witness Sentences')

    plt.tight_layout()
    plt.show()

def log_transform(matrix):
    # Apply log transformation to the matrix
    return np.log(matrix + 1e-9)

def clip_transform(matrix, low, high):
    # Clip values within the specified range
    return np.clip(matrix, low, high)


similarity_matrix = np.load("Data/similarity_matrix.npy")
similarity_matrix = similarity_matrix[:25, :25]

# Create sentence labels (replace with actual sentences)
witness_sentences = [f"witness_sentence_{i}" for i in range(similarity_matrix.shape[0])]
document_sentences = [f"document_sentence_{i}" for i in range(similarity_matrix.shape[1])]

# Call the function to create the graph
create_similarity_graph(similarity_matrix, witness_sentences, document_sentences, threshold=0.995, show_labels=False)
# create_similarity_heatmap(similarity_matrix, witness_sentences, document_sentences, transformation="clip", clip_range=(0.99, 1.0))
