from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# モデルとトークナイザのロード
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 翻訳の設定（ソース言語とターゲット言語）
source_lang = "en_XX"  # 英語
target_lang = "ja_XX"  # 日本語

# トークナイザにターゲット言語を設定
tokenizer.src_lang = source_lang

# 翻訳元のテキスト
text = "In times of need, if you look in the right place, you just may see a strange telephone number scrawled in red. If you call this number, you will hear a young man introduce himself as the Yato God. Yato is a minor deity and a self-proclaimed Delivery God, who dreams of having millions of worshippers. Without a single shrine dedicated to his name, however, his goals are far from being realized. He spends his days doing odd jobs for five yen apiece, until his weapon partner becomes fed up with her useless master and deserts him. Just as things seem to be looking grim for the god, his fortune changes when a middle school girl, Hiyori Iki, supposedly saves Yato from a car accident, taking the hit for him. Remarkably, she survives, but the event has caused her soul to become loose and hence able to leave her body. Hiyori demands that Yato return her to normal, but upon learning that he needs a new partner to do so, reluctantly agrees to help him find one. And with Hiyori's help, Yato's luck may finally be turning around."
inputs = tokenizer(text, return_tensors="pt", truncation=True)

# デコーダー（出力側）のターゲット言語トークンを設定
forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

# 翻訳の実行
outputs = model.generate(
    inputs.input_ids, 
    max_length=1000, 
    num_beams=5, 
    forced_bos_token_id=forced_bos_token_id,  # ターゲット言語の指定
    early_stopping=True
)

# 結果をデコード
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)
