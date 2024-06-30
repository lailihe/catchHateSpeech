import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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

# 텍스트 벡터화
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['대상하이라이트'])
X_val = vectorizer.transform(val_df['대상하이라이트'])
X_test = vectorizer.transform(test_df['대상하이라이트'])

y_train = train_df['혐오 클래스']
y_val = val_df['혐오 클래스']
y_test = test_df['혐오 클래스']

# 로지스틱 회귀 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 검증 데이터 평가
y_val_pred = model.predict(X_val)
print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred))

# 테스트 데이터 평가
y_test_pred = model.predict(X_test)
print("Test Set Performance:")
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

# 모델 및 벡터라이저 저장
joblib.dump(model, 'hate_speech_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("모델 저장 완료!")
