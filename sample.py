import joblib
joblib.dump(pipeline, 'sentiment_model.pkl')

model = joblib.load("sentiment_model.pkl")
prediction = model.predict(["Your test sentence"])

model = joblib.load("sentiment_model.pkl")
prediction = model.predict(["Your test sentence"])


model = joblib.load("sentiment_model.pkl")
vectorizer = model.named_steps['tfidf']

# Save separately
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
