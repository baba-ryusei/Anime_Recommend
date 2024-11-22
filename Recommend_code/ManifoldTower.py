import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from TwoTower_data_gen import AnimeDataset
from torch.utils.data import Dataset, DataLoader
import geoopt

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"

class TwoTowerNNModel(nn.Module):
    def __init__(self, num_users, num_animes, embedding_dim, hidden_dim=32):
        super(TwoTowerNNModel, self).__init__()
        
        self.manifold = geoopt.PoincareBall()
        
        # ユーザーとアイテムの埋め込み層
        self.user_embedding = geoopt.ManifoldParameter(self.manifold.random((num_users, embedding_dim)), manifold=self.manifold)
        self.anime_embedding = geoopt.ManifoldParameter(self.manifold.random((num_animes, embedding_dim)), manifold=self.manifold)
        
        # ユーザーとアイテムタワーに追加する全結合層（Dense層）
        self.user_dense = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.anime_dense = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, user_ids, anime_ids):
        # ユーザーとアイテムの埋め込みベクトル
        user_embeds = self.user_embedding[user_ids]
        anime_embeds = self.anime_embedding[anime_ids]
        
        # 各タワーの全結合層を通過
        user_output = self.user_dense(user_embeds)
        anime_output = self.anime_dense(anime_embeds)
        
        # ユーザーとアイテムベクトルの内積を計算（類似度スコア）
        similarity_score = -self.manifold.dist(user_output, anime_output)
        
        return similarity_score    
    
class HyperbolicDistanceLoss(nn.Module):
    def __init__(self, manifold):
        super(HyperbolicDistanceLoss, self).__init__()
        self.manifold = manifold

    def forward(self, user_output, anime_output, ratings):
        # ユーザとアイテムのベクトル間の双曲距離を計算
        distances = self.manifold.dist(user_output, anime_output)
        
        # レーティングを双曲空間にマッピング（Poincare Ballにレーティングを埋め込む）
        rating_embeddings = self.manifold.expmap0(ratings.unsqueeze(-1))
        
        # 双曲距離でレーティングとの距離を計算
        loss = self.manifold.dist(distances.unsqueeze(-1), rating_embeddings).mean()
        
        return loss
    
def train(device):
    #train = pd.read_csv("Recommend/anime/train_dataset/train_dataset.csv")
    #merge_id = pd.read_csv('Recommend/anime/pre_dataset/merge_id.csv')
    #train_dataset = AnimeDataset(train)
    train_dataset = torch.load('Recommend/anime/train_dataset/train_dataset.pt')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    
    #num_users = merge_id.shape[0]
    #num_animes = merge_id.shape[0]
    num_users = 69123
    num_animes = 2365
    embedding_dim = 50
    hidden_dim = 32 
    
    model = TwoTowerNNModel(num_users, num_animes, embedding_dim).to(device)

    #criterion = nn.MSELoss()  
    criterion = HyperbolicDistanceLoss(model.manifold)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for user_ids, anime_ids, ratings in train_loader:
            user_ids = user_ids.long().to(device)
            anime_ids = anime_ids.long().to(device)
            ratings = ratings.float().to(device)
            optimizer.zero_grad()
            #outputs = model(user_ids, anime_ids)
            user_output = model.user_dense(model.user_embedding[user_ids])
            anime_output = model.anime_dense(model.anime_embedding[anime_ids])
            loss = criterion(user_output, anime_output, ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}')
    torch.save(model.state_dict(), 'Recommend/anime/model/ManifoldTower_model.pt')    
        
def test(device):
    #test_dataset = AnimeDataset(test)
    #test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    s=1
    return s
            
def candidate(device, target_user_id):
    num_animes = 2365
    num_users = 69123
    target_user_id = target_user_id
    model = TwoTowerNNModel(num_users, num_animes, embedding_dim=50).to(device)
    model.load_state_dict(torch.load('Recommend/anime/model/ManifoldTower_model.pt'))
    
    all_anime_ids = torch.arange(num_animes)
    all_anime_embeddings = model.anime_embedding(all_anime_ids) 
    
    all_user_ids = torch.tensor([target_user_id]).to(device)
    all_user_embeddings = model.user_embedding(all_user_ids)
    
    similarity_score = (all_anime_embeddings @ all_user_embeddings.T).squeeze()
    
    top_k = 10
    top_k_indices = torch.topk(similarity_score, top_k).indices
    recommended_anime_ids = all_anime_ids[top_k_indices]
    
    return recommended_anime_ids            
        
if __name__ == '__main__':
    device = "cuda"
    select=["train","test","candidate"]
    choice=input(f"Choose a method to execute ({', '.join(select)}): ")
    
    """
    train_data = pd.read_csv("Recommend/anime/train_dataset/train_dataset.csv")
    user_ids = train_data['user_idx'].unique()
    anime_ids = train_data['anime_idx'].unique()
    """

    if choice == "train":
        print("-----------------Training-----------------------")
        train(device)
        
    elif choice == "test" :
        print("------------------Test---------------------")
        test(device)
        
    else:
        print("------------------candidate------------")
        recommended_anime_id = candidate(device, target_user_id=69000)                
        print(recommended_anime_id)