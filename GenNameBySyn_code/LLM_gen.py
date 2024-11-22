import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#model_name = "facebook/mbart-large-50-many-to-many-mmt"
#model_name = "Helsinki-NLP/opus-mt-ja-en"
model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# バックトランスレーション関数（日本語に変換後英語に変換する関数）
def back_translation(text, source_lang="en_XX", intermediate_lang="ja_XX", target_lang="en_XX"):
    try:
        # ステップ1: 英語 -> 日本語
        tokenizer.src_lang = source_lang
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(intermediate_lang)
        intermediate_output = model.generate(
            inputs.input_ids,
            max_length=512,
            num_beams=5,
            forced_bos_token_id=forced_bos_token_id,
            early_stopping=True,
        )
        intermediate_text = tokenizer.decode(intermediate_output[0], skip_special_tokens=True)

        # ステップ2: 日本語 -> 英語
        tokenizer.src_lang = intermediate_lang
        inputs = tokenizer(intermediate_text, return_tensors="pt", truncation=True)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        final_output = model.generate(
            inputs.input_ids,
            max_length=512,
            num_beams=5,
            forced_bos_token_id=forced_bos_token_id,
            early_stopping=True,
        )
        back_translated_text = tokenizer.decode(final_output[0], skip_special_tokens=True)

        return back_translated_text
    except Exception as e:
        print(f"Error in back-translation: {e}")
        return text  # エラー時は元のテキストを返す

# パラフレーズ生成関数
def generate_paraphrase(text, source_lang="en_XX", target_lang="en_XX", max_length=512, num_return_sequences=10):
    try:
        tokenizer.src_lang = source_lang
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=5,
            num_return_sequences=num_return_sequences,
            temperature=2.0, # 多様性の調整
            top_k=100, # 上位kの単語を選出
            top_p=0.80, # 累積確率がp以内の単語を選出
            forced_bos_token_id=forced_bos_token_id,
            early_stopping=True,
        )
        paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return paraphrases
    except Exception as e:
        print(f"Error in paraphrase generation: {e}")
        return [text]  # エラー時は元のテキストをリストとして返す

# フィルタリング関数（類似度を計算して重複排除）
def filter_similar_paraphrases(paraphrases, threshold=0.8):
    vectorizer = TfidfVectorizer().fit_transform(paraphrases)
    similarity_matrix = cosine_similarity(vectorizer)
    filtered = []
    for i, text in enumerate(paraphrases):
        if all(similarity_matrix[i, j] < threshold for j in range(len(filtered))):
            filtered.append(text)
    return filtered

# メイン処理
def process_anime_csv(input_file="Pre_data/new_anime_dataset_TV_Genre.csv", output_file="grow_LLM_data/anime_with_paraphrases.csv"):
    data = pd.read_csv(input_file)
    if 'sypnopsis' not in data.columns:
        print("Error: 'sypnopsis' column not found in the input CSV.")
        return

    all_paraphrases = []

    for i, row in data.iterrows():
        original_text = row['sypnopsis']
        print(f"Processing {i+1}/{len(data)}: {original_text[:50]}...")  

        back_translated = back_translation(original_text)

        paraphrases = generate_paraphrase(back_translated)
        paraphrases.append(back_translated) 

        unique_paraphrases = filter_similar_paraphrases([original_text] + paraphrases)

        all_paraphrases.append(", ".join(unique_paraphrases)) 
        time.sleep(1)  # モデル負荷を抑えるための遅延

    data['paraphrases'] = all_paraphrases

    data.to_csv(output_file, index=False)
    print(f"Processing completed. Saved to {output_file}")

if __name__ == "__main__":
    process_anime_csv()
