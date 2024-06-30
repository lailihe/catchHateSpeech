import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
import joblib

# 데이터 파일이 저장된 경로
data_dir = 'C:/Selectstar_Tunip_HUMANE Lab_opendata'

# 모든 JSON 파일 로드
data = []
for file_name in os.listdir(data_dir):
    if file_name.endswith('.json'):
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            data.append(json.load(file))

# DataFrame으로 변환
df = pd.DataFrame(data)

# 필요한 필드 추출
df = df[['대상하이라이트', '혐오 클래스']]

# NaN 값 제거
df = df.dropna(subset=['혐오 클래스', '대상하이라이트'])

# 혐오 클래스 인코딩
df['혐오 클래스'] = df['혐오 클래스'].map({'N': 0, 'Y': 1})

# 데이터 분할 (8:1:1)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# BERT 토크나이저와 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# 데이터셋 변환 함수
def encode_data(df, tokenizer):
    return tokenizer(df['대상하이라이트'].tolist(), padding=True, truncation=True, return_tensors='pt')

# 데이터셋 인코딩
train_encodings = encode_data(train_df, tokenizer)
val_encodings = encode_data(val_df, tokenizer)
test_encodings = encode_data(test_df, tokenizer)

# PyTorch 데이터셋 클래스
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 데이터셋 생성
train_dataset = HateSpeechDataset(train_encodings, train_df['혐오 클래스'].tolist())
val_dataset = HateSpeechDataset(val_encodings, val_df['혐오 클래스'].tolist())
test_dataset = HateSpeechDataset(test_encodings, test_df['혐오 클래스'].tolist())

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 모델 훈련
trainer.train()

# 검증 데이터 평가
val_pred = trainer.predict(val_dataset)
print("Validation Set Performance:")
print(classification_report(val_df['혐오 클래스'], val_pred.predictions.argmax(axis=1)))

# 테스트 데이터 평가
test_pred = trainer.predict(test_dataset)
print("Test Set Performance:")
print(classification_report(test_df['혐오 클래스'], test_pred.predictions.argmax(axis=1)))
print(confusion_matrix(test_df['혐오 클래스'], test_pred.predictions.argmax(axis=1)))

# 모델 저장
model.save_pretrained('hate_speech_bert_model')
tokenizer.save_pretrained('hate_speech_bert_tokenizer')

print("모델 저장 완료!")
