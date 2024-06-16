# from pprint import pprint

import requests
import matplotlib.pyplot as plt
import networkx as nx

from lib.utils import load_json_file

# git source: https://github.com/machinereading/frameBERT
# NLTK download: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
# model download: https://drive.google.com/drive/folders/1bDUoIniUNUm2I0ztXWo6hitOpLmU9lv4
# FrameNet dataset: https://www.kaggle.com/datasets/nltkdata/framenet/data

dataset = load_json_file("shared_data/paper_c_3_11_sentences_with_frame_net.json")

datapoint = dataset[0]


# define IP address and port
# IP = "141.43.202.175"
IP = "localhost"
PORT = "5001"


def get_frames_graph(sent):
  # print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  # pprint(match_issues(sent))
  response = requests.get(f"http://{IP}:{PORT}/api/extract_frames_graph/" + sent)
  response = response.json()
  # results = response["frames"]["textae"]
  print(f"> Sentence: {sent}")
  return response


def draw_frame_graph(frame_data):
  g = nx.DiGraph()

  # Track the unique nodes
  unique_nodes = {}

  for frame, role, value in frame_data:
    frame_id = frame.split(':')[1]
    role_type = role.split(':')[1] if ':' in role else role

    frame_node_label = frame_id.replace('_', ' ').title()

    # Differentiate between entity and other role types
    if 'fe:' in role:
      entity_label = value  # Use entity name as label
    else:
      entity_label = f'{role_type} ({value})'

    # Add frame node
    if frame_node_label not in unique_nodes:
      g.add_node(frame_node_label, color='skyblue')
      unique_nodes[frame_node_label] = True

    # Add entity node
    if entity_label not in unique_nodes:
      g.add_node(entity_label, color='lightgrey' if 'fe:' in role else 'orange')
      unique_nodes[entity_label] = True

    # Add edge with role as an attribute
    g.add_edge(frame_node_label, entity_label, role=role_type)

  # Use a different layout if needed
  pos = nx.kamada_kawai_layout(g)
  pos = {node: (x * 250, y * 250) for node, (x, y) in pos.items()}  # Scale positions

  # Scale node size based on degree
  node_size = [g.degree(node) * 300 for node in g.nodes()]

  # Draw nodes and labels
  nx.draw_networkx_nodes(g, pos, node_size=node_size, node_color=[g.nodes[node]['color']
                                                                  for node in g.nodes()], alpha=0.7)
  nx.draw_networkx_labels(g, pos, font_size=8)

  # Draw edges
  nx.draw_networkx_edges(g, pos, alpha=0.5, edge_color='gray')

  # Draw edge labels
  edge_labels = {edge: g[edge[0]][edge[1]]['role'] for edge in g.edges()}
  nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color='red', font_size=6)

  plt.show()


# Get frames data in graph format
parsed = get_frames_graph(datapoint["text"])
# Plot the graph
draw_frame_graph(parsed["frames"])
