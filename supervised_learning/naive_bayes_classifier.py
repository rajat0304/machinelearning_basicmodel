from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
reviews = [
    "This movie is great and fun",    # Positive
    "I love this awesome movie",      # Positive
    "This movie is bad and boring",   # Negative
    "I hate this terrible movie"      # Negative
]
labels = ["Positive", "Positive", "Negative", "Negative"]
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(reviews)
model = MultinomialNB()
model.fit(X_train, labels)
new_review = ["The movie was great and I love it"]
X_test = vectorizer.transform(new_review)
prediction = model.predict(X_test)

print(f"Review: '{new_review[0]}'")
print(f"Model ka Result: {prediction[0]}")
