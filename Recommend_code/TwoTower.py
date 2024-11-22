import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from TwoTower_data_gen import AnimeDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"

class TwoTowerNNModel(nn.Module):
    def __init__(self, num_users, num_animes, embedding_dim, hidden_dim):
        super(TwoTowerNNModel, self).__init__()
        
        # ユーザーとアイテムの埋め込み層
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.anime_embedding = nn.Embedding(num_animes, embedding_dim)
        
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
        user_embeds = self.user_embedding(user_ids)
        anime_embeds = self.anime_embedding(anime_ids)
        
        # 各タワーの全結合層を通過
        user_output = self.user_dense(user_embeds)
        anime_output = self.anime_dense(anime_embeds)
        
        # ユーザーとアイテムベクトルの内積を計算（類似度スコア）
        similarity_score = (user_output * anime_output).sum(dim=1)
        
        return similarity_score    
    
def train(device,data):
    data = data
    train_dataset = torch.load('Dataset/TwoTower_data/train_dataset_tv_action.pt')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    
    num_users = data['user_id'].nunique()
    num_animes = data['MAL_ID'].nunique()
    embedding_dim = 50
    hidden_dim = 32 
    
    model = TwoTowerNNModel(num_users, num_animes, embedding_dim).to(device)

    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for user_ids, anime_ids, ratings in train_loader:
            user_ids = user_ids.long().to(device)
            anime_ids = anime_ids.long().to(device)
            ratings = ratings.float().to(device)
            optimizer.zero_grad()
            outputs = model(user_ids, anime_ids)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}')
    torch.save(model.state_dict(), 'Model/TwoTower_model_tv_action_genre.pt')    
        
def test(device,data):
    data = data
    test_dataset = AnimeDataset(test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    num_users = data['user_id'].nunique()
    num_animes = data['MAL_ID'].nunique()
    embedding_dim = 50
    hidden_dim = 32 
    
    model = TwoTowerNNModel(num_users, num_animes, embedding_dim).to(device)
    model.load_state_dict(torch.load("Model/TwoTower_model.pt"))

    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.eval()
        train_loss = 0
        for user_ids, anime_ids, ratings in test_loader:
            user_ids = user_ids.long().to(device)
            anime_ids = anime_ids.long().to(device)
            ratings = ratings.float().to(device)
            optimizer.zero_grad()
            outputs = model(user_ids, anime_ids)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(test_loader)}')
    torch.save(model.state_dict(), 'Model/TwoTower_model_tv_action_genre.pt')   

def precision_recall_at_k(device, data, target_user_id, k=10):
    num_users = data['user_id'].nunique()
    num_animes = data['MAL_ID'].nunique()
    
    model = TwoTowerNNModel(num_users, num_animes, embedding_dim=50).to(device)
    model.load_state_dict(torch.load('Model/TwoTower_model.pt'))
    model.eval()

    # データとモデルから候補のアニメを生成
    all_anime_ids = torch.arange(num_animes).to(device)
    user_embedding = model.user_embedding(torch.tensor([target_user_id]).to(device))
    all_anime_embeddings = model.anime_embedding(all_anime_ids)

    # 推定スコアを計算
    similarity_scores = (all_anime_embeddings @ user_embedding.T).squeeze()
    top_k_indices = torch.topk(similarity_scores, k).indices
    recommended_anime_ids = all_anime_ids[top_k_indices].cpu().numpy()

    # 実際に視聴済みのアニメを取得
    watched_anime_ids = data[data['user_id'] == target_user_id]['MAL_ID'].values

    # 推奨リスト内の一致するアニメの数
    hits = np.intersect1d(recommended_anime_ids, watched_anime_ids)

    # Precision@KとRecall@K
    precision = len(hits) / k
    recall = len(hits) / len(watched_anime_ids) if len(watched_anime_ids) > 0 else 0.0

    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}: {recall:.4f}")

    return precision, recall
 
    
# アニメの名前から類似したアニメを出力    
def recommend_by_anime_name(device, data, input_anime_name, top_k=10):
    anime_data = pd.read_csv('anime_syn.csv')
    
    # アニメ名からMAL_IDを取得
    if input_anime_name not in anime_data['Name'].values:
        raise ValueError(f"Anime '{input_anime_name}' not found in the dataset.")
    
    input_anime_id = anime_data.loc[anime_data['Name'] == input_anime_name, 'MAL_ID'].values[0]
    
    num_users = data['user_id'].nunique()
    num_animes = data['MAL_ID'].nunique()
    embedding_dim = 50
    model = TwoTowerNNModel(num_users, num_animes, embedding_dim).to(device)
    model.load_state_dict(torch.load('Model/TwoTower_model_tv_action.pt'))
    
    all_anime_ids = torch.arange(num_animes).to(device)
    all_anime_embeddings = model.anime_embedding(all_anime_ids)
    print("anime_emb_size=",all_anime_embeddings)
    
    input_anime_embedding = model.anime_embedding(torch.tensor([input_anime_id]).to(device))
    print("input_anime_emb_size=",input_anime_embedding)
    
    similarity_scores = (all_anime_embeddings @ input_anime_embedding.T).squeeze()
    
    top_k_indices = torch.topk(similarity_scores, top_k).indices
    recommended_anime_ids = all_anime_ids[top_k_indices]
    
    # IDからアニメ名を取得
    recommended_anime_names = anime_data[anime_data['MAL_ID'].isin(recommended_anime_ids.cpu().numpy())]['Name'].tolist()
    
    return recommended_anime_names


# ユーザーIDから候補となるアニメIDを取得
def candidate(device, data,target_user_id):
    num_users = data['user_id'].nunique()
    num_animes = data['MAL_ID'].nunique()
    target_user_id = target_user_id
    model = TwoTowerNNModel(num_users, num_animes, embedding_dim=50).to(device)
    model.load_state_dict(torch.load('Model/TwoTower_model.pt'))
    
    all_anime_ids = torch.arange(num_animes).to(device)
    all_anime_embeddings = model.anime_embedding(all_anime_ids) 
    
    all_user_ids = torch.tensor([target_user_id]).to(device)
    all_user_embeddings = model.user_embedding(all_user_ids)
    
    similarity_score = (all_anime_embeddings @ all_user_embeddings.T).squeeze()
    
    top_k = 10
    top_k_indices = torch.topk(similarity_score, top_k).indices
    recommended_anime_ids = all_anime_ids[top_k_indices]
    
    return recommended_anime_ids   

# アニメIDをアニメ名に変換
def id_convert_name(recommended_anime_ids):
    data = pd.read_csv('anime_syn.csv')
    recommended_anime_ids = recommended_anime_ids.cpu().numpy()
    recommended_anime_names = data[data['MAL_ID'].isin(recommended_anime_ids)]['Name'].tolist()
    
    return recommended_anime_names

def candidate_based_cluster(device, data, target_user_id, num_clusters):
    num_animes = data["MAL_ID"].nunique()
    num_users = data["user_id"].nunique()
    embedding_dim = 50
    
    model = TwoTowerNNModel(num_users, num_animes,embedding_dim).to(device)
    model.load_state_dict(torch.load('Model/TwoTower_model.pt'))
    
    all_anime_ids =torch.arange(num_animes).to(device)
    all_anime_embedding = model.anime_embedding(all_anime_ids).detach().cpu().numpy()
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    anime_clusters = kmeans.fit_predict(all_anime_embedding)
    
    target_user_embedding = model.user_embedding(torch.tensor([target_user_id],device=device))
    
    user_cluster = kmeans.predict(target_user_embedding.detach().cpu().numpy())[0]
    
    cluster_anime_indices = [i for i, cluster_id in enumerate(anime_clusters) if cluster_id == user_cluster]
    cluster_anime_ids = all_anime_ids[cluster_anime_indices]    
    
    cluster_anime_embedding = torch.tensor([all_anime_embedding[cluster_anime_indices]]).to(device)
    similarity_score = (cluster_anime_embedding @ target_user_embedding.T).squeeze()
    
    top_k = 10
    top_k_indices = torch.topk(similarity_score, top_k).indices
    recommended_anime_ids = cluster_anime_ids[top_k_indices]
    
    return recommended_anime_ids.cpu().numpy()

def recommend_by_anime_name(device, data, input_anime_name, num_clusters=10, top_k=10):
    anime_data = pd.read_csv('anime_syn.csv')  # 'anime_syn.csv'にアニメ名とMAL_IDが格納されていると仮定
    
    if input_anime_name not in anime_data['Name'].values:
        raise ValueError(f"Anime '{input_anime_name}' not found in the dataset.")
    
    input_anime_id = anime_data.loc[anime_data['Name'] == input_anime_name, 'MAL_ID'].values[0]
    
    num_animes = data["MAL_ID"].nunique()
    num_users = data["user_id"].nunique()
    embedding_dim = 50
    
    model = TwoTowerNNModel(num_users, num_animes, embedding_dim).to(device)
    model.load_state_dict(torch.load('Model/TwoTower_model_tv_action.pt'))
    
    all_anime_ids = torch.arange(num_animes).to(device)
    all_anime_embedding = model.anime_embedding(all_anime_ids).detach().cpu().numpy()
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    anime_clusters = kmeans.fit_predict(all_anime_embedding)

    input_anime_embedding = model.anime_embedding(torch.tensor([input_anime_id]).to(device))
    
    input_anime_cluster = kmeans.predict(input_anime_embedding.detach().cpu().numpy())[0]
    cluster_anime_indices = [i for i, cluster_id in enumerate(anime_clusters) if cluster_id == input_anime_cluster]
    cluster_anime_ids = all_anime_ids[cluster_anime_indices]
    
    cluster_anime_embedding = torch.tensor([all_anime_embedding[cluster_anime_indices]]).to(device)
    similarity_score = (cluster_anime_embedding @ input_anime_embedding.T).squeeze()
    
    top_k_indices = torch.topk(similarity_score, top_k).indices
    recommended_anime_ids = cluster_anime_ids[top_k_indices].cpu().numpy()
    
    recommended_anime_names = anime_data[anime_data['MAL_ID'].isin(recommended_anime_ids)]['Name'].tolist()
    
    return recommended_anime_names


# データセットのタイトルをキーワードで検索
def searchanime(keyword):
    data = pd.read_csv("anime_syn.csv")
    matching_titles = data[data['Name'].str.contains(keyword, case=False, na=False)]['Name'].tolist()
    print(matching_titles)
        
        
if __name__ == '__main__':
    device = "cuda"
    data = pd.read_csv('Dataset/merged_data_embeddings_notsyn.csv')
    target_user_id = 61829
    input_anime_name = "Initial D Fourth Stage"
    keyword = "Death"
    
    select=["train","test","precision","two_name", "two_id", "cluster_name", "cluster_id","search"]
    choice=input(f"Choose a method to execute ({', '.join(select)}): ")

    if choice == "train":
        print("-----------------Training-----------------------")
        train(device,data)
        
    elif choice == "test" :
        print("------------------Test---------------------")
        test(device,data)
        
    elif choice == "precision":
        print("-------------------precision-----------------")
        precision_recall_at_k(device, data, target_user_id, k=10)  
        
    elif choice == "two_name":
        print("--------------Recommendation by TwoTower---------------")
        print(recommend_by_anime_name(device, data, input_anime_name, top_k=10))
        
    elif choice =="two_id":
        print("--------------Recommendation by TwoTower---------------")
        recommended_anime_ids = candidate(device, data,target_user_id)   
        print(id_convert_name(recommended_anime_ids))    
        
    elif choice =="cluster_id":
        print("--------------Recommendation by TwoTower and Clustering---------------")
        recommended_anime_ids = candidate_based_cluster(device,data,target_user_id,10)
        print(id_convert_name(torch.tensor(recommended_anime_ids)))   
        
    elif choice =="cluster_name":
        print("--------------Recommendation by TwoTower and Clustering---------------")
        print(recommend_by_anime_name(device, data, input_anime_name, num_clusters=10, top_k=10))    
        
    elif choice =="search":
        print("-----------------------Search anime-----------------")
        searchanime(keyword)     
        