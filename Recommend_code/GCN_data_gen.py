from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import from_networkx
import torch
import pandas as pd

data = pd.read_csv("Dataset/merged_data_embeddings_notsyn.csv")

# PyTorch Geometric形式に変換
edge_index = torch.tensor([data['user_id'].tolist(), data['MAL_ID'].tolist()], dtype=torch.long)
# エッジ特徴量として評価値を追加
edge_attr = torch.tensor(data['rating'].tolist(), dtype=torch.float)

# ノード数の計算（ユーザー + アニメ）
num_users = data['user_id'].nunique()
num_animes = data['MAL_ID'].nunique()
num_nodes = num_users + num_animes

# PyTorch Geometricのデータオブジェクト
graph_data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

torch.save(graph_data, "Dataset/Graph_data/graph_data.pt")
