# Sentiment_classifier.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("text_data.csv")

# Features and labels
X = df['text']
y = df['label']

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained — Accuracy on test set: {accuracy * 100:.2f}%")

# Interactive testing
print("\nWelcome to the Sentiment Classifier! (Type 'exit' to quit)\n")

positive_count = 0
negative_count = 0
neutral_count = 0

while True:
    user_input = input("Enter your sentence: ")
    if user_input.lower() == "exit":
        break
    vectorized_input = vectorizer.transform([user_input])
    prediction = model.predict(vectorized_input)[0]
    print(f"Sentiment: {prediction}\n")

    if prediction == "positive":
        positive_count += 1
    elif prediction == "negative":
        negative_count += 1
    else:
        neutral_count += 1

# Session Summary
print("\nSession Summary:")
print(f"Positive: {positive_count} sentence(s)")
print(f"Negative: {negative_count} sentence(s)")
print(f"Neutral : {neutral_count} sentence(s)")
