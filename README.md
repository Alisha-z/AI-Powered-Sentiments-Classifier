# AI-Powered-Sentiments-Classifier

# 🤖 AI-Powered Sentiment Classifier

This project demonstrates different stages of building a **Sentiment Text Classifier** using Python and machine learning libraries. The system analyzes user input and classifies it as **Positive**, **Negative**, or **Neutral**.

---

## 🧠 Project Variants and Purpose of Each File

### 🔹 1. `classifier.py`
A basic rule-based sentiment classifier using manual matching and keyword logic.

- **Dataset used**: `dataset.csv`
- **No ML training required**
- **Suitable for: Beginners and demonstrations of logic-based AI**

---

### 🔹 2. `text_sentiment_classifier.py`
An improved version of the classifier using **scikit-learn**, **TfidfVectorizer**, and **LogisticRegression**.

- **Dataset used**: `text_data.csv`
- **Trains a model using Tfidf features**
- Saves trained model to:  
  - `sentiment_model.pkl`  
  - `vectorizer.pkl`
- Must be run once before using the main GUI-based file

---

### 🔹 3. `sentiment_classifier.py` (🌟 Main File)
The main sentiment classifier app with a user-friendly **GUI built using Tkinter**.

- Loads `sentiment_model.pkl` and `vectorizer.pkl`
- Takes user input through a GUI
- Shows predicted sentiment
- Tracks and displays the session summary
- **Dataset used during training**: `sentiments_dataset.csv`  
  (used internally in `text_sentiment_classifier.py`)

---

## 📊 Datasets

- `dataset.csv` → Used in `classifier.py`  
- `text_data.csv` → Used in `text_sentiment_classifier.py`  
- `sentiments_dataset.csv` → Final version for training the ML model (used by `sentiment_classifier.py`)

---

## 🧪 How to Use

### ⚙️ Step 1: Train the Model
Run the following to train and save the model:

```bash
python text_sentiment_classifier.py
