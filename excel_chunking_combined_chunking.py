import pandas as pd
import streamlit as st
#from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#import chromadb
from groq import Groq

client = Groq(api_key="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")


def combine_rows_columns(df):
    """Combines rows and columns into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        chunk = " | ".join(f"{col}: {val}" for col, val in row.items())
        chunks.append(chunk)
    return chunks


def vectorizing(get_chunk):
    # Load a pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Convert the text chunks into vectors
    vectors = model.encode(get_chunk)
    return vectors

    
    # Initialize Chroma client and create a collection
    # client = chromadb.Client()
    # collection = client.create_collection("my_collection")

    # # Add the vectors to the Chroma collection
    # collection.add(
    #     documents=get_chunk,  # Text data
    #     embeddings=vectors  # Corresponding vectors
    # )

    #st.write("Vectors stored into db successfully")

# # Function to get relevant chunks using TF-IDF and cosine similarity
def get_relevant_chunks(query, chunks, top_n=100): 
    """
    Retrieve the most relevant chunks using SentenceTransformer and cosine similarity.
    
    Args:
        query (str): The search query.
        chunks (list of str): A list of text chunks.
        top_n (int): Number of top relevant chunks to return (default is 6).
    
    Returns:
        list of str: Top relevant chunks sorted by relevance.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks + [query])
    
    
    # # Load a pre-trained SentenceTransformer model
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # # Convert the text chunks into vectors
    # vectors = model.encode(chunks)
    
    cosine_sim = cosine_similarity(vectors[-1:], vectors[:-1])
    #st.write(cosine_sim)
    relevant_indices = cosine_sim[0].argsort()[-top_n:][::-1]
    return [chunks[i] for i in relevant_indices]

# def get_relevant_chunks(query, chunks, threshold=0.3):
#     """
#     Retrieve all chunks with similarity above a threshold using TF-IDF vectorization.
    
#     Args:
#         query (str): The search query.
#         chunks (list of str): A list of text chunks.
#         threshold (float): Minimum similarity score to consider a chunk relevant.
    
#     Returns:
#         list of str: Relevant chunks sorted by similarity.
#     """
#     # Create a TfidfVectorizer
#     vectorizer = TfidfVectorizer()
    
#     # Fit and transform the chunks and the query
#     vectors = vectorizer.fit_transform(chunks + [query])
    
#     # Separate the vectors for the chunks and the query
#     chunk_vectors = vectors[:-1]  # All except the last (which is the query)
#     query_vector = vectors[-1]  # The last one is the query
    
#     # Compute cosine similarity
#     cosine_sim = cosine_similarity(query_vector, chunk_vectors).flatten()
#     st.write(cosine_sim)
    
#     # Filter chunks by similarity threshold
#     relevant_indices = [i for i, sim in enumerate(cosine_sim) if sim >= threshold]
    
#     # Sort by similarity in descending order
#     relevant_indices = sorted(relevant_indices, key=lambda i: cosine_sim[i], reverse=True)
    
#     return [chunks[i] for i in relevant_indices]





# def get_relevant_chunks(query, chunks, top_n=5, similarity_weight=1.0, ranking_weight=0.5):
#     """
#     Retrieve relevant chunks based on a weighted combination of similarity score and rank.
    
#     Args:
#         query (str): The search query.
#         chunks (list of str): A list of text chunks.
#         top_n (int): The number of top relevant chunks to return.
#         similarity_weight (float): Weight for the cosine similarity score.
#         ranking_weight (float): Weight for ranking based on chunk relevance.
    
#     Returns:
#         list of str: Top relevant chunks based on weighted scores.
#     """
#     # Vectorize the query and the chunks using TF-IDF
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform(chunks + [query])

#     # Compute cosine similarity
#     cosine_sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

#     # Combine similarity scores with their respective ranks (higher rank = more relevance)
#     weighted_scores = [
#         (i, cosine_sim[i] * similarity_weight + (len(chunks) - i) * ranking_weight)
#         for i in range(len(chunks))
#     ]
    
#     # Sort based on the weighted score
#     weighted_scores.sort(key=lambda x: x[1], reverse=True)

#     # Return top_n chunks based on the weighted score
#     top_chunks = [chunks[i] for i, _ in weighted_scores[:top_n]]

#     return top_chunks





st.title("Excel Analyser")
uploaded_file=st.file_uploader("Upload an file",type=["csv"])
user_query=st.text_input("Enter the user query")
if st.button("submit"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        chunks = combine_rows_columns(df)
        relevant_chunks=get_relevant_chunks(user_query,chunks)
        #relevant_chunks = get_relevant_chunks(user_query, chunks, top_n=3, similarity_weight=1.0, ranking_weight=0.5)


        #st.write(relevant_chunks)
        if user_query:
            context = "\n\n".join(relevant_chunks)
        #st.write(context)
        # Initialize completion to avoid NameError
        completion = None
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"use the data {context} only and frame the answer for this question {user_query} in formal english"}],
                #messages=[{"role":"user","content":context}],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )
            response_text = ""
            for chunk in completion:
                response_text += chunk.choices[0].delta.content or ""
            st.write(response_text)  # Update the output dynamically    
        except Exception as e:
            st.write(e)
        # answer = genai.gemini.model.generate_content(
        #             f"use the data {context} and frame the answer for this question {question} use this template  in formal english"
        #         )
        # result_text = answer.candidates[0].content.parts[0].text
        #vectorizing(chunks)
        #st.write(chunks)
        
       

    else:
        st.warning("Please upload a file")
