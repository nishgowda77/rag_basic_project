from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read text data
with open("data.txt", "r") as file:
    data = file.readlines()

# Ask question
question = input("Ask a question: ")

# Create vectors
vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(data + [question])

# Similarity checking
similarity = cosine_similarity(vectors[-1], vectors[:-1])

# Convert similarity scores
scores = similarity.flatten()

# Sort answers
sorted_indices = scores.argsort()[::-1]

print("\nTOP MATCHES:\n")

# Print top 3 answers
for i in sorted_indices[:3]:
    print(f"{data[i].strip()} --> Score: {scores[i]:.2f}")