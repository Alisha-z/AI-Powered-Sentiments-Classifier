# text_sentiment_classifier.py

from textblob import TextBlob

# Counters to track sentiment counts
positive_count = 0
negative_count = 0
neutral_count = 0

print("💬 Welcome to the Sentiment Classifier!")
print("Type 'exit' to end the session.\n")

# File to save results
output_file = "sentiment_results.txt"

# Loop to take user input repeatedly
while True:
    user_input = input("Enter a sentence: ")

    if user_input.lower() == 'exit':
        break

    # Create TextBlob object
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity

    # Classify based on polarity score
    if polarity > 0:
        sentiment = "Positive"
        positive_count += 1
    elif polarity < 0:
        sentiment = "Negative"
        negative_count += 1
    else:
        sentiment = "Neutral"
        neutral_count += 1

    print(f"Sentiment: {sentiment}\n")

    # Save to file
    with open(output_file, "a", encoding="utf-8") as file:
        file.write(f"Input: {user_input} | Sentiment: {sentiment}\n")

# Summary at the end
print("📊 Session Summary:")
print(f"Positive: {positive_count}")
print(f"Negative: {negative_count}")
print(f"Neutral : {neutral_count}")
