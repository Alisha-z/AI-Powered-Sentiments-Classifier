import tkinter as tk
from tkinter import messagebox
import joblib
import os

# Load trained model and vectorizer
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    messagebox.showerror("Error", "Trained model or vectorizer not found. Please run training first.")
    exit()

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Statistics counters
positive_count = 0
negative_count = 0
neutral_count = 0


def classify_sentiment():
    global positive_count, negative_count, neutral_count
    text = entry.get()

    if text.lower() == "exit":
        messagebox.showinfo("Session Summary",
                            f"Positive: {positive_count}\nNegative: {negative_count}\nNeutral: {neutral_count}")
        root.quit()
        return

    if text.strip() == "":
        messagebox.showwarning("Input Error", "Please enter a sentence.")
        return

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    if prediction == "positive":
        positive_count += 1
    elif prediction == "negative":
        negative_count += 1
    elif prediction == "neutral":
        neutral_count += 1

    result_label.config(text=f"Sentiment: {prediction.capitalize()}", fg="blue")
    entry.delete(0, tk.END)


# GUI Setup
root = tk.Tk()
root.title("Sentiment Classifier")
root.geometry("400x250")
root.config(bg="#f0f0f0")

tk.Label(root, text="Enter a sentence:", font=("Arial", 12), bg="#f0f0f0").pack(pady=10)
entry = tk.Entry(root, width=50, font=("Arial", 12))
entry.pack(pady=5)

tk.Button(root, text="Classify Sentiment", command=classify_sentiment, font=("Arial", 12), bg="lightgreen").pack(
    pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f0f0f0")
result_label.pack(pady=20)

tk.Button(root, text="Exit & Show Summary", command=lambda: classify_sentiment(), font=("Arial", 10),
          bg="lightcoral").pack(pady=10)

root.mainloop()
