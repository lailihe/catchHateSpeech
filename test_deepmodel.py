import torch
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
import re

# 저장된 모델과 토크나이저 로드
model = BertForSequenceClassification.from_pretrained('hate_speech_bert_model')
tokenizer = BertTokenizer.from_pretrained('hate_speech_bert_tokenizer')

# 텍스트 전처리 함수
def clean_text(text):
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z\s]', '', text)

# 혐오 표현 감지 함수
def detect_hate_speech(text, model, tokenizer, threshold=0.7):
    cleaned_text = clean_text(text)
    inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prob = predictions[0][1].item()
    return prob >= threshold, prob

# 샘플 문장 목록
sample_sentences = [
    "저 사람은 진짜 한녀 같아.",
    "짱개들은 정말 못생겼어.",
    "그녀는 완전 미개해.",
    "동남아 냄새나서 못 참겠어.",
    "바퀴벌레 같은 인간들.",
    "오늘 날씨가 정말 좋네요.",
    "저 국개들을 도저히 가만히 둘 수 없다.",
    "그는 매우 친절한 사람입니다.",
    "독도의 명물에 대해서 소개할게.",
    "감히 윤썩널 칭찬을 하다니."
]

# 샘플 문장 분석
for sentence in sample_sentences:
    is_hate_speech, probability = detect_hate_speech(sentence, model, tokenizer)
    if is_hate_speech:
        print(f"혐오 표현 감지됨: {sentence} | 혐오 정도: {probability:.2f}")
    else:
        print(f"혐오 표현 없음: {sentence}")

