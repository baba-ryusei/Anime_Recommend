from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


def load_and_filter_data(filepath, genre_filter="Action"):
    data = pd.read_csv(filepath)
    data = data.dropna()  # 欠損値を削除
    data = data[data['Genres'].str.contains(genre_filter, na=False)]  # 指定ジャンルでフィルタリング
    return data

# あらすじの埋め込みベクトル生成
def generate_embeddings(model, texts, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()
        batch_embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=True)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def save_embeddings(data, syn_embeddings, genre_embeddings, output_path):
    data['Genre_Embedding'] = list(genre_embeddings)
    data['Sypnopsis_Embedding'] = list(syn_embeddings)
    output_data = data[['MAL_ID', 'Name', 'Score', 'Completed', 'Favorites', 'Genre_Embedding', 'Sypnopsis_Embedding']]
    output_data.to_csv(output_path, index=False)

def main():
    model = SentenceTransformer('all-MiniLM-L12-v2')

    input_path = "tv_merged.csv"
    output_path = "anime_tv_action_embeddings.csv"
    data = load_and_filter_data(input_path)

    syn_embeddings = generate_embeddings(model, data['sypnopsis'])
    genre_embeddings = generate_embeddings(model, data['Genres'])

    save_embeddings(data, syn_embeddings, genre_embeddings, output_path)
    print("Embeddings successfully saved to:", output_path)


if __name__ == "__main__":
    main()
