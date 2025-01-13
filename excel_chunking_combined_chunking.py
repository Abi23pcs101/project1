import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

#client = Groq(api_key="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")   #llama 3.3

client=Groq(api_key="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")   #llama 3.2
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
    return vector

    #st.write("Vectors stored into db successfully")
def process_chunks_in_batches(query, chunks, batch_size=10):
    """Processes chunks in batches to handle large datasets."""
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_relevance = get_relevant_chunks(query, batch, top_n=batch_size)
        results.extend(batch_relevance)
    return results

# # Function to get relevant chunks using TF-IDF and cosine similarity
def get_relevant_chunks(query, chunks, top_n=30): 
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
def semantic_chunking(df, key_column, max_chunk_size=500):
    """
    Creates semantic chunks by grouping rows based on a key column and ensuring chunks fit within a size limit.

    Args:
        df (pd.DataFrame): The input DataFrame.
        key_column (str): The column to group rows by for semantic meaning.
        max_chunk_size (int): The maximum size of a chunk in characters or tokens.

    Returns:
        list: A list of semantic text chunks.
    """
    chunks = []
    current_chunk = ""
    
    for _, row in df.iterrows():
        # Create a text representation of the row
        row_text = " | ".join(f"{col}: {val}" for col, val in row.items())
        
        # Check if adding this row exceeds the max chunk size
        if len(current_chunk) + len(row_text) + 3 > max_chunk_size:  # +3 for separators
            chunks.append(current_chunk.strip())  # Save the current chunk
            current_chunk = ""  # Start a new chunk
        
        # Append the current row to the chunk
        current_chunk += row_text + " || "
    
    # Add the final chunk if not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


st.title("Honeypot Data Analyser")
uploaded_file=st.file_uploader("Upload an file",type=["csv"])
user_query=st.text_input("Enter the user query")
if st.button("submit"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        #chunks = semantic_chunking(df, key_column="date", max_chunk_size=500)
        #relevant_chunks = process_chunks_in_batches(user_query, chunks, batch_size=10)
        
        chunks = combine_rows_columns(df)
        relevant_chunks=get_relevant_chunks(user_query,chunks)
    
        #st.write(relevant_chunks)
        if user_query:
            context = "\n\n".join(relevant_chunks)
        #st.write(context)
        # Initialize completion to avoid NameError
        completion = None
        try:
            completion = client.chat.completions.create(
                #model="llama-3.3-70b-versatile",
                model="llama-3.2-1b-preview",
                messages=[{"role": "user", "content": f"use the data {context} only and frame the answer for this question {user_query} in formal english and give the answer as crisp and short conveying the complete meaning"}],
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
     
       

    else:
        st.warning("Please upload a file")
