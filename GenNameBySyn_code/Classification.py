import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import ast
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['Embedding'] = data['Embedding'].apply(
        lambda x: re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', x)
    ).apply(ast.literal_eval)
    embeddings = torch.tensor(data['Embedding'].tolist(), dtype=torch.float32)
    labels = torch.tensor(data['Cluster_Label'].values, dtype=torch.long)
    return data, embeddings, labels

class AnimeEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):
        return {'embeddings': self.embeddings[idx], 'labels': self.labels[idx]}

    def __len__(self):
        return len(self.labels)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerClassifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        x = self.transformer_encoder(x)
        return self.fc(x[:, 0, :])  # クラス分類用

def train_model(model, criterion, optimizer, train_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# アニメ推薦システム
def recommend_anime(synopsis, embedding_data, model, sentence_model, top_k=5):
    synopsis_embedding = sentence_model.encode([synopsis], convert_to_tensor=True)

    with torch.no_grad():
        cluster_probs = model(synopsis_embedding.to(device))
        predicted_cluster = torch.argmax(cluster_probs, dim=1).item()

    cluster_data = embedding_data[embedding_data['Cluster_Label'] == predicted_cluster]

    cluster_embeddings = np.stack(cluster_data['Embedding'].values)
    similarities = cosine_similarity([synopsis_embedding.cpu().numpy()], cluster_embeddings).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]
    recommendations = cluster_data.iloc[top_indices]

    return recommendations[['Name', 'Score', 'Favorites']]

def main():
    data, embeddings, labels = load_and_preprocess_data("Dataset/Clustered_data/clustered_synopsis_data.csv")
    train_emb, test_emb, train_lbl, test_lbl = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    train_dataset = AnimeEmbeddingDataset(train_emb, train_lbl)
    test_dataset = AnimeEmbeddingDataset(test_emb, test_lbl)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = train_emb.size(1)
    num_classes = len(torch.unique(labels))
    model = TransformerClassifier(input_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 20
    train_model(model, criterion, optimizer, train_loader, num_epochs)

    evaluate_model(model, test_loader)

    torch.save(model.state_dict(), "Model/transformer_classifier.pth")
    print("Model saved.")

    # 推論と推薦
    sentence_model = SentenceTransformer('all-MiniLM-L12-v2')
    #sentence_model = SentenceTransformer('bert-base-uncased')
    synopsis_input = "A young boy embarks on a journey to become the world's strongest warrior."
    recommendations = recommend_anime(synopsis_input, data, model, sentence_model)
    print(recommendations)

if __name__ == "__main__":
    main()
