import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

anime_tv = pd.read_csv('Dataset/merged_data_embeddings.csv')
print(anime_tv.columns)

anime_tv_action = anime_tv[anime_tv['Genres'].str.contains('Action', na=False)]
#anime_tv_action.to_csv('anime_tv_action.csv', index=False)