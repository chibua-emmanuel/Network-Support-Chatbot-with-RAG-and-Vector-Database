




import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import weaviate
import json
import requests
import openai
from tqdm import tqdm, trange

# Initialize OpenAI
openai.api_key = 'Openai key'  # Replace with your actual OpenAI API key

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Extracting text from PDF documents
def preprocess_data(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(file_path)
            data.append(text)
    return data

# Initialize the Weaviate client
def initialize_weaviate():
    client = weaviate.Client(
        url="URL",  # Weaviate Cloud URL
        auth_client_secret=weaviate.AuthApiKey(api_key="#####")  # API key
    )
    return client

# Initialize local embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

# Preprocess data and index it in Weaviate
def index_data(client, data):
    for text in tqdm(data, desc="Indexing data"):
        embedding = generate_embedding(text)
        client.data_object.create(data_object={"text": text}, vector=embedding, class_name="NetworkSupport")

def generate_embedding(text):
    return embedding_model.encode([text])[0]

# Retrieve top K similar documents based on user query
def retrieve_similar_documents(query, client, top_k=3):
    result = client.query.get(
        className="NetworkSupport",
        properties=["text"],
        filters={
            "operator": "Similarity",
            "value": generate_embedding(query),
            "type": "vector"
        },
        limit=top_k
    )
    return result

# Generate response using OpenAI's GPT-4o model
def generate_response(context, query):
    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=f"{context}\n{query}",
        max_tokens=100,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

# Initialize Weaviate and preprocess data
cisco_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Cisco")
mikrotik_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Mikrotik")
all_data = cisco_data + mikrotik_data

client = initialize_weaviate()
index_data(client, all_data)

# Streamlit user interface
st.title("Network Support Chatbot")
st.write("Ask your questions about Cisco and Mikrotik devices.")

user_query = st.text_input("Enter your question:")
if user_query:
    results = retrieve_similar_documents(user_query, client)
    if results:
        context = "\n\n".join([result["text"] for result in results])
        response = generate_response(context, user_query)
        st.write("Response from GPT-4o:")
        st.write(response)
    else:
        st.write("No relevant records found.")
