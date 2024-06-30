import joblib
import re
from konlpy.tag import Okt
import os as sys

# 형태소 분석기 로드
okt = Okt()

def clean_text(text):
    # 텍스트 전처리: 공백을 표준화 - 여러개의 공백을 하나의 공백으로
    # 특수문자 제거
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z\s]', '', text)
    return text.strip()

# 형태소 분석기를 사용하여 텍스트 토큰화
def tokenize(text):
    return okt.morphs(text)

# 저장된 모델과 벡터라이저 로드
# 벡터라이저: TF-IDF 벡터라이저를 사용하여 텍스트 데이터를 벡터 형태로 변환 (텍스트 데이터를 숫자 벡터로 변환하여 머신러닝 모델에 입력)
model1 = joblib.load('hate_speech_model.pkl')
vectorizer1 = joblib.load('tfidf_vectorizer.pkl')

model2 = joblib.load('UPhate_speech_ensemble_model.pkl')
vectorizer2 = joblib.load('UPtfidf_vectorizer.pkl')

# 벡터라이저의 tokenizer 재설정
vectorizer2.tokenizer = tokenize

# 혐오 감지 함수
# 방법: 입력된 텍스트에서 토큰을 추출하고, 각 토큰이 혐오 표현인지 예측
# 예측 결과는 각 토큰과 그 확률로 나타남
def detect_hate_speech_model1(text, model, vectorizer, threshold=0.7):
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    hate_speech_tokens = []
    probabilities = []

    for token in tokens:
        token_str = ' '.join([token])
        vectorized_text = vectorizer.transform([token_str])
        prediction_proba = model.predict_proba(vectorized_text)[0][1]
        
        if prediction_proba >= threshold:
            hate_speech_tokens.append(token)
            probabilities.append(prediction_proba)
    
    return hate_speech_tokens, probabilities

def detect_hate_speech_model2(text, model, vectorizer, threshold=0.7):
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    hate_speech_tokens = []
    probabilities = []

    for token in tokens:
        token_str = ' '.join([token])
        vectorized_text = vectorizer.transform([token_str])
        prediction_proba = model.predict_proba(vectorized_text)[0][1]
        
        if prediction_proba >= threshold:
            hate_speech_tokens.append(token)
            probabilities.append(prediction_proba)
    
    return hate_speech_tokens, probabilities

print("혐오 표현 탐지 프로그램입니다. 문장을 입력하세요 (종료하려면 Ctrl+C):")

try:
    while True:
        user_input = input(">> ")
        
        hate_speech_tokens1, probabilities1 = detect_hate_speech_model1(user_input, model1, vectorizer1)
        hate_speech_tokens2, probabilities2 = detect_hate_speech_model2(user_input, model2, vectorizer2)
        
        print("\n모델 1 (Logistic Regression 기반):")
        if hate_speech_tokens1:
            print(f"혐오 표현이 감지되었습니다. 문장: {user_input}")
            for token, prob in zip(hate_speech_tokens1, probabilities1):
                print(f"단어: {token}, 혐오 정도: {prob:.2f}")
        else:
            print("혐오 표현이 감지되지 않았습니다.")
        
        print("\n모델 2 (앙상블 모델):")
        if hate_speech_tokens2:
            print(f"혐오 표현이 감지되었습니다. 문장: {user_input}")
            for token, prob in zip(hate_speech_tokens2, probabilities2):
                print(f"단어: {token}, 혐오 정도: {prob:.2f}")
        else:
            print("혐오 표현이 감지되지 않았습니다.")
        
except KeyboardInterrupt:
    print("\n프로그램을 종료합니다.")
    sys.exit()
