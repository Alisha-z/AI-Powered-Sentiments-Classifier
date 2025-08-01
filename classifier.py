import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("sentiment_dataset.csv")

# Split data
X = df['text']
y = df['label']

# Create a pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "sentiment_model.pkl")

# Track results
results = {"Positive": 0, "Negative": 0, "Neutral": 0}

# Start user input loop
print("Welcome to the Sentiment Classifier! (Type 'exit' to quit)\n")
while True:
    text = input("Enter your sentence: ")
    if text.lower() == 'exit':
        break

    prediction = model.predict([text])[0]
    print(f"Sentiment: {prediction}\n")

    # Save result
    results[prediction] += 1
    with open("classified_results.txt", "a") as f:
        f.write(f"{text} => {prediction}\n")

# Final summary
print("\nSession Summary:")
for label, count in results.items():
    print(f"{label}: {count} sentence(s)")
