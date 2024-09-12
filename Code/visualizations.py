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

# Example similarity matrix (replace with your data)
similarity_matrix = np.random.rand(10, 10)  # Replace with your similarity data

# Create sentence labels (replace with actual sentences)
witness_sentences = [f"witness_sentence_{i}" for i in similarity_matrix.shape[0]]
document_sentences = [f"document_sentence_{i}" for i in similarity_matrix.shape[1]]

# Create a NetworkX graph
G = nx.Graph()

# Add witness nodes
for i, sentence in enumerate(witness_sentences):
    G.add_node(sentence, label=sentence, group='witness')

# Add document nodes
for j, sentence in enumerate(document_sentences):
    G.add_node(sentence, label=sentence, group='document')

# Add weighted edges based on similarity score
threshold = 0.5  # Only show edges above a certain similarity
for i, witness_sentence in enumerate(witness_sentences):
    for j, document_sentence in enumerate(document_sentences):
        similarity = similarity_matrix[i, j]
        if similarity > threshold:
            G.add_edge(witness_sentence, document_sentence, weight=similarity)

# Set witness node size proportional to the number of connections (degree) above the threshold
for node in G.nodes:
    if G.nodes[node]['group'] == 'witness':
        # Degree counts the number of edges (connections) the node has
        degree = G.degree(node)
        G.nodes[node]['size'] = 5 + degree * 2  # Base size + degree-based scaling
    else:
        G.nodes[node]['size'] = 10  # Fixed size for document nodes

# Visualize the graph using PyVis
net = Network(notebook=True, height='800px', width='100%', bgcolor='#222222', font_color='white')

# Add nodes and edges from the NetworkX graph to PyVis
net.from_nx(G)

# Color the nodes by group and apply size
for node in net.nodes:
    if node['group'] == 'witness':
        node['color'] = '#FF5733'  # Example color for witness sentences
    else:
        node['color'] = '#33FF57'  # Example color for document sentences
    
    node['size'] = G.nodes[node['id']]['size']  # Set node size

# Show the graph in the notebook or save to an HTML file
net.show('semantic_similarity_graph.html')







# Example similarity matrix (replace with your data)
similarity_matrix = np.random.rand(10, 10)  # Replace with your similarity data

# Create sentence labels (replace with actual sentences)
witness_sentences = [f"witness_sentence_{i}" for i in range(10)]
document_sentences = [f"document_sentence_{i}" for i in range(10)]

# Create a pandas DataFrame for easier plotting
df = pd.DataFrame(similarity_matrix, index=witness_sentences, columns=document_sentences)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=False, cmap='YlGnBu', cbar_kws={'label': 'Semantic Similarity'}, linewidths=0.5)

plt.title('Semantic Similarity Heatmap')
plt.xlabel('Document Sentences')
plt.ylabel('Witness Sentences')

plt.tight_layout()
plt.show()
