import pandas as pd
anime_syn = pd.read_csv("anime_tv_action_embeddings.csv")
rating = pd.read_csv("Dataset/Pre_data/rating_complete.csv")

# MAL_ID と anime_id をキーとして結合
merged_data = anime_syn.merge(rating, left_on='MAL_ID', right_on='anime_id', how='inner')

# 必要に応じてカラムを整理
#merged_data = merged_data.rename(columns={'anime_id': 'ANIME_ID'})
merged_data = merged_data.drop(columns=['anime_id'])
merged_data = merged_data[["MAL_ID","user_id","Genre_Embedding","rating"]]
print(merged_data)
merged_data.to_csv('Dataset/merged_tv_action_embeddings_gen.csv', index=False)

