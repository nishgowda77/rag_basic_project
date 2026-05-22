import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read data
with open("data.txt", "r") as file:
    data = file.readlines()

# Title
st.title("Mini RAG Chatbot")

# User input
question = st.text_input("Ask a question")

if question:
    
    # Create vectorizer
    vectorizer = TfidfVectorizer()

    # Convert text to vectors
    vectors = vectorizer.fit_transform(data + [question])

    # Similarity check
    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    # Scores
    scores = similarity.flatten()

    # Best matches
    sorted_indices = scores.argsort()[::-1]

    st.subheader("Top Answers")

    # Show top 3 answers
    for i in sorted_indices[:3]:
        st.write(f"{data[i].strip()} --> Score: {scores[i]:.2f}")