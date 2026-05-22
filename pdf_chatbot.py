import streamlit as st
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    data = text.split(".")

    question = st.text_input("Ask a question from PDF")

    if question:

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(data + [question])

        similarity = cosine_similarity(vectors[-1], vectors[:-1])

        scores = similarity.flatten()

        sorted_indices = scores.argsort()[::-1]

        st.subheader("Answers")

        for i in sorted_indices[:3]:

            if scores[i] > 0:

                st.success(data[i])

                st.caption(f"Score: {scores[i]:.2f}")