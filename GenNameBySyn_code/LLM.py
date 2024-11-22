import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig, AdamW, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import ast
import re

#anime_syn = pd.read_csv('Pre_data/new_anime_dataset_TV_Genre.csv')
anime_syn = pd.read_csv('grow_LLM_data/anime_with_paraphrases.csv')
syn_id = anime_syn[['MAL_ID', 'Name', 'sypnopsis','paraphrases']].dropna() 

unique_ids = syn_id['MAL_ID'].unique()
id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
syn_id['new_id'] = syn_id['MAL_ID'].map(id_map)
syn_id['combined_text'] = syn_id['sypnopsis'] + " " + syn_id['paraphrases']

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_texts, test_texts, train_labels, test_labels = train_test_split(
    syn_id['combined_text'].tolist(),
    syn_id['new_id'].values,
    test_size=0.2,
    random_state=42
)

# データのトークン化
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

class AnimeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AnimeDataset(train_encodings, train_labels)
test_dataset = AnimeDataset(test_encodings, test_labels)


# 通常のファインチューニング
def tuning():    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(id_map))

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_steps=500,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = test_dataset
    )
    
    trainer.train()
    model.save_pretrained("Model/BERT_model")
    tokenizer.save_pretrained("Model/BERT_tokenizer")
    
# LoRAモデルを用いたファインチューニング    
def tuning_lora():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(id_map))

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # シーケンス分類タスク
        r=8,                         # LoRAの低ランク次元
        lora_alpha=32,               # LoRAのスケール係数
        lora_dropout=0.1,            # ドロップアウトの割合
        target_modules=["attention.output.dense", "intermediate.dense"] # LoRAモデルの適用対象モジュール
    )
    
    # LoRAモデル
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        learning_rate=5e-6,
        lr_scheduler_type="linear",
        warmup_steps=500,
        weight_decay=0.01
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    
    num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()
    model.save_pretrained("Model/Para_LoRA_BERT_model")
    tokenizer.save_pretrained("Model/Para_LoRA_BERT_tokenizer")
    
# QLoRAを用いたチューニング    
def tuning_qlora():
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["attention.output.dense", "intermediate.dense"], 
    )

    # QLoRAモデル
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(id_map), load_in_4bit=True) 
    model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        fp16=False,  # 16ビット精度でのトレーニングを有効に
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_steps=500,
        weight_decay=0.01
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)

    num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=(optimizer, scheduler)
    )
    
    trainer.train()

    model.save_pretrained("Model/Cluster_QLoRA_BERT_model")
    tokenizer.save_pretrained("Model/Cluster_QLoRA_BERT_tokenizer")

    
# 蒸留損失（Teacherモデルの出力とStudentモデルの出力をソフトマックスで比較）    
def distillation_loss(student_logits, teacher_logits, true_labels, temperature=2.0, alpha=0.5): 
    soft_teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    distillation_loss = F.kl_div(soft_student_probs, soft_teacher_probs, reduction="batchmean") * (temperature ** 2)

    true_loss = F.cross_entropy(student_logits, true_labels)

    return alpha * distillation_loss + (1 - alpha) * true_loss       
    
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        student_outputs = model(**inputs)
        
        loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, labels)
        
        return (loss, student_outputs) if return_outputs else loss
     
def Distillation():
    teacher_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(id_map))
    student_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(id_map))

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs"
    )

    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    student_model.save_pretrained("Model/Cluster_Distilled_Student_Model")
    tokenizer.save_pretrained("Model/Cluster_Distilled_Student_Tokenizer")
    
    
def test():
    model = AutoModelForSequenceClassification.from_pretrained("Model/Cluster_LoRA_BERT_model", num_labels=len(id_map))
    tokenizer = AutoTokenizer.from_pretrained("Model/Cluster_LoRA_BERT_tokenizer")

    trainer = Trainer(
        model=model,
    )
    
    outputs = trainer.predict(test_dataset)
    predictions = torch.argmax(torch.tensor(outputs.predictions), dim=1)

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    test_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    torch.save(test_results, "LLM_result/evaluation_results.pt")
    

# あらすじを入力するとアニメの名前を返す関数
def recommend_anime(input_syn, model, tokenizer, id_to_name):
    inputs = tokenizer(input_syn, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    outputs = model(**inputs)
    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    
    # 予測IDに対応するアニメ名を取得
    anime_name = id_to_name.get(predicted_id, "Anime not found")
    print(anime_name)
    return anime_name

# あらすじを入力すると上位10件のアニメ名を返す関数
def recommend_anime_top10(input_syn, model, tokenizer, id_to_name):
    inputs = tokenizer(input_syn, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    outputs = model(**inputs)
    top_k = 10  # 上位10件
    top_ids = torch.topk(outputs.logits, top_k, dim=1).indices.squeeze().tolist()
    
    # 予測IDに対応するアニメ名を取得
    anime_names = [id_to_name.get(predicted_id, "Anime not found") for predicted_id in top_ids]
    
    # 上位10件のアニメ名を出力
    for i, anime_name in enumerate(anime_names, 1):
        print(f"Rank {i}: {anime_name}")
    
    return anime_names


# データセットのタイトルをキーワードで検索
def searchanime(syn_id, keyword):
    matching_titles = syn_id[syn_id['Name'].str.contains(keyword, case=False, na=False)]['Name'].tolist()
    print(matching_titles)


# アニメの名前を入力するとあらすじを返す関数
def get_synopsis_from_name(name_to_sypnopsis, anime_name):
    sypnopsis = name_to_sypnopsis.get(anime_name, "Synopsis not found")
    print(sypnopsis)
    return sypnopsis


if __name__ =='__main__':    
    
    model = AutoModelForSequenceClassification.from_pretrained("Model/BERT_model", num_labels=len(id_map))
    tokenizer = AutoTokenizer.from_pretrained("Model/BERT_tokenizer")
    
    id_to_name = dict(zip(syn_id['MAL_ID'], syn_id['Name']))
    name_to_sypnopsis = dict(zip(syn_id['Name'], syn_id['sypnopsis']))
    
    input_syn = "In times of need, if you look in the right place, you just may see a strange telephone number scrawled in red. If you call this number, you will hear a young man introduce himself as the Yato God. Yato is a minor deity and a self-proclaimed Delivery God, who dreams of having millions of worshippers. Without a single shrine dedicated to his name, however, his goals are far from being realized. He spends his days doing odd jobs for five yen apiece, until his weapon partner becomes fed up with her useless master and deserts him. Just as things seem to be looking grim for the god, his fortune changes when a middle school girl, Hiyori Iki, supposedly saves Yato from a car accident, taking the hit for him. Remarkably, she survives, but the event has caused her soul to become loose and hence able to leave her body. Hiyori demands that Yato return her to normal, but upon learning that he needs a new partner to do so, reluctantly agrees to help him find one. And with Hiyori's help, Yato's luck may finally be turning around."
    anime_name = "Jormungand"
    
    options = {
        "finetuning": tuning,   
        "tuning_lora": tuning_lora,
        "tuning_qlora": tuning_qlora,
        "Distillation": Distillation,
        "test": test,
        "get_name": lambda: recommend_anime_top10(input_syn, model, tokenizer, id_to_name),
        "get_syn": lambda: get_synopsis_from_name(name_to_sypnopsis, input("Enter the anime name: ")),
        "search": lambda: searchanime(syn_id, input("Enter the keyword: "))
    }
    
    print("Choose a method to execute:")
    for key in options.keys():
        print(f"- {key}")
    choice = input("Enter your choice: ")

    if choice in options:
        options[choice]()
    else:
        print("Invalid choice. Please select a valid option.")
            