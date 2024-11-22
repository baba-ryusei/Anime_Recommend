import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from TwoTower import TwoTowerNNModel
import numpy as np

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

def load_and_prepare_data(data_path, anime_path):
    data = pd.read_csv(data_path)
    anime_data = pd.read_csv(anime_path)
    return data, anime_data

# TwoTowerモデルから特徴量を生成
def extract_features(data, model, device):
    num_users = data['user_id'].nunique()
    num_animes = data['MAL_ID'].nunique()
    embedding_dim = 50
    hidden_dim = 32
    
    model = TwoTowerNNModel(num_users, num_animes, embedding_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load('Model/TwoTower_model.pt'))
    model.eval()

    features, labels = [], []
    for _, row in data.iterrows():
        user_id = torch.tensor([row['user_id']], device=device)
        anime_id = torch.tensor([row['MAL_ID']], device=device)
        rating = row['rating']
        print(f"Max user_id in data: {data['user_id'].max()}")
        print(f"Number of users in embedding: {model.user_embedding.num_embeddings}")

            # 埋め込み層のサイズを取得
        num_users = model.user_embedding.num_embeddings
        num_animes = model.anime_embedding.num_embeddings

        assert 0 <= user_id < num_users, f"Invalid user_id: {user_id}"
        assert 0 <= anime_id < num_animes, f"Invalid anime_id: {anime_id}"

        user_embedding = model.user_embedding(user_id)
        anime_embedding = model.anime_embedding(anime_id)
        similarity_score = (user_embedding * anime_embedding).sum().item()

        # 特徴量: 類似度スコア + 埋め込みのベクトル要素
        feature = [
            similarity_score,
            *user_embedding.squeeze().detach().cpu().numpy(),
            *anime_embedding.squeeze().detach().cpu().numpy()
        ]

        features.append(feature)
        labels.append(rating)

    return pd.DataFrame(features), pd.Series(labels)

# XGBoost
def train_gradient_boosting(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6
    }
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def evaluate_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    rmse = calculate_rmse(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}")
    return y_pred

# 入力されたアニメからおすすめのアニメを推薦
def recommend_with_boosting(input_anime_name, data, anime_data, two_tower_model, gb_model, top_k=10):
    if input_anime_name not in anime_data['Name'].values:
        raise ValueError(f"Anime '{input_anime_name}' not found in the dataset.")

    input_anime_id = anime_data.loc[anime_data['Name'] == input_anime_name, 'MAL_ID'].values[0]
    input_anime_embedding = two_tower_model.anime_embedding(torch.tensor([input_anime_id], device=device))

    candidates = []
    for _, row in data.iterrows():
        user_id = torch.tensor([row['user_id']], device=device)
        user_embedding = two_tower_model.user_embedding(user_id)
        similarity_score = (user_embedding * input_anime_embedding).sum().item()

        feature = [
            similarity_score,
            *user_embedding.squeeze().cpu().numpy(),
            *input_anime_embedding.squeeze().cpu().numpy()
        ]
        candidates.append(feature)

    candidates_df = pd.DataFrame(candidates)
    dtest = xgb.DMatrix(candidates_df)
    scores = gb_model.predict(dtest)

    top_k_indices = scores.argsort()[-top_k:][::-1]
    recommended_anime_ids = data.iloc[top_k_indices]['MAL_ID'].values

    recommended_anime_names = anime_data[anime_data['MAL_ID'].isin(recommended_anime_ids)]['Name'].tolist()
    return recommended_anime_names

if __name__ == "__main__":
    data_path = 'Dataset/merged_data_embeddings_notsyn.csv'
    anime_path = 'anime_syn.csv'

    data, anime_data = load_and_prepare_data(data_path, anime_path)

    # 特徴量の抽出
    num_users = data['user_id'].nunique()
    num_animes = data['MAL_ID'].nunique()
    two_tower_model = TwoTowerNNModel(num_users, num_animes, embedding_dim=50, hidden_dim=32).to(device)
    features, labels = extract_features(data, two_tower_model, device)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    gb_model = train_gradient_boosting(X_train, y_train)

    evaluate_model(gb_model, X_test, y_test)

    # アニメ推薦
    input_anime_name = "Initial D Fourth Stage"
    recommendations = recommend_with_boosting(input_anime_name, data, anime_data, two_tower_model, gb_model, top_k=10)
    print("Recommended Anime:")
    for anime in recommendations:
        print(anime)
