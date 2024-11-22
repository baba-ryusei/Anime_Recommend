from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class AnimeDataset(Dataset):
    def __init__(self, data):
        # ユーザーの特徴量
        self.user_idx = data['user_id'].values
        
        # アニメの特徴量
        self.anime_idx = data['MAL_ID'].values
        self.item_embeddings = torch.tensor(np.stack(data['Genre_Embedding'].values), dtype=torch.float32)
        #self.item_features = data[['Genre_Embedding', 'Synopsis_Embedding']].values.astype(np.float32)
        
        # 評価ラベル
        self.ratings = data['rating'].values.astype(np.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        user_info = (self.user_idx[idx])
        #item_info = (self.anime_idx[idx], self.item_features[idx])
        item_info = (self.anime_idx[idx])
        label = self.ratings[idx]
        anime_emb = self.item_embeddings[idx]
        
        return user_info, item_info, label, anime_emb
    
if __name__ == '__main__':
    anime = pd.read_csv('Dataset/merged_tv_action_embeddings_gen.csv')

    # ユーザーIDとアニメIDを連続したインデックスに変換
    user_ids = anime['user_id'].unique()
    anime_ids = anime['MAL_ID'].unique()

    user_id_map = {id: idx for idx, id in enumerate(user_ids)}
    anime_id_map = {id: idx for idx, id in enumerate(anime_ids)}

    anime['user_id'] = anime['user_id'].map(user_id_map)
    anime['MAL_ID'] = anime['MAL_ID'].map(anime_id_map)
    
    print("Max user_idx:", anime['user_id'].max(), "Expected max:", len(user_id_map) - 1)
    print("Max anime_idx:", anime['MAL_ID'].max(), "Expected max:", len(anime_id_map) - 1)
    
    train, test = train_test_split(anime, test_size=0.2, random_state=42)
    train_dataset = AnimeDataset(train)
    test_dataset = AnimeDataset(test)
    
    #train.to_csv("Dataset/TwoTower_data/train_dataset.csv", index=False)
    #test.to_csv("Dataset/TwoTower_data/test_dataset.csv", index=False)
    torch.save(train_dataset, "Dataset/TwoTower_data/train_dataset_tv_action_gen.pt")
    torch.save(test_dataset, "Dataset/TwoTower_data/test_dataset_tv_action_gen.pt")
        
