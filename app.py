# =========================
# Fake News Detection Flask App (Fixed)
# =========================

import pandas as pd
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib
from flask import Flask, render_template, request

# -------------------------
# 1. Load dataset and balance (train once)
# -------------------------
df = pd.read_csv("news.csv")
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.strip() != '']
df['label'] = df['label'].str.lower().map({'fake': 0, 'real': 1})

df_fake = df[df['label'] == 0]
df_real = df[df['label'] == 1]

df_real_upsampled = resample(df_real,
                             replace=True,
                             n_samples=len(df_fake),
                             random_state=42)

df_balanced = pd.concat([df_fake, df_real_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)  # shuffle

# -------------------------
# 2. Text cleaning
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df_balanced['text'] = df_balanced['text'].apply(clean_text)

# -------------------------
# 3. Train TF-IDF and Random Forest (train once, save)
# -------------------------
X = df_balanced['text']
y = df_balanced['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2), max_features=8000)
X_train_tfidf = tfidf.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Save trained model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Model trained and saved successfully!")

# -------------------------
# 4. Flask App
# -------------------------
app = Flask(__name__)

# Load model and vectorizer (only once)
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    if news_text.strip() == "":
        return render_template('index.html', news_text=news_text, prediction="Please enter some text!")
    
    cleaned_text = clean_text(news_text)
    vect_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vect_text)[0]
    result = "REAL" if prediction == 1 else "FAKE"
    
    return render_template('index.html', news_text=news_text, prediction=result)

# -------------------------
# 5. Run Flask
# -------------------------
if __name__ == "__main__":
    print("Flask server starting... open http://127.0.0.1:5000/ in your browser")
    app.run(debug=True)
