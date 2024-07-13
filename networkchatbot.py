import os
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import weaviate
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.vectorstores import Weaviate as LangChainWeaviate
from langchain.embeddings import OpenAIEmbeddings
from tqdm import tqdm
import logging

# Set OpenAI API key as environment variable
os.environ["OPENAI_API_KEY"] = "..."

# Initialize OpenAI with LangChain
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model_name='gpt-4')

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(file_path):
    pdf_document = fitz.open(file_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Extracting text from PDF documents
def preprocess_data(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            try:
                text = extract_text_from_pdf(file_path)
                data.append(text)
            except Exception as e:
                logging.error(f"Error extracting text from {file_path}: {e}")
    return data

# Initialize the Weaviate client
def initialize_weaviate():
    client = weaviate.Client(
        url="...",  # Weaviate Cloud URL
        auth_client_secret=weaviate.AuthApiKey(api_key="...")  # API key
    )
    return client

# Initialize local embedding model with SentenceTransformer
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

# Preprocess data and index it in Weaviate using LangChain
def index_data(client, data):
    for text in tqdm(data, desc="Indexing data"):
        embedding = embedding_model.encode([text])[0]
        client.data_object.create(data_object={"text": text}, vector=embedding, class_name="NetworkSupport")

# Retrieve top K similar documents based on user query using LangChain
def retrieve_similar_documents(query, client, top_k=3):
    embedding = embedding_model.encode([query])[0]
    near_vector = {"vector": embedding, "certainty": 0.7}  # Adjust certainty as needed
    response = client.query.get("NetworkSupport", ["text"]).with_near_vector(near_vector).with_limit(top_k).do()
    return response.get("data", {}).get("Get", {}).get("NetworkSupport", [])

# Generate response using LangChain LLMChain with enhanced prompt template
def generate_response(context, query):
    prompt_template = (
        "You are a network support expert specializing in Cisco and Mikrotik devices. "
        "Based on the following context, provide a detailed and technical answer to the question. "
        "Use technical terms where appropriate and ensure the explanation is clear and thorough.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{query}\n\n"
        "Detailed Response:"
    )
    prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=context, query=query)
    return response

# Preprocess data and initialize Weaviate
def main():
    # Preprocess data
    cisco_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Cisco")
    mikrotik_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Mikrotik")
    all_data = cisco_data + mikrotik_data

    # Initialize Weaviate client
    client = initialize_weaviate()

    # Index data in Weaviate
    try:
        index_data(client, all_data)
        st.success("Data indexing completed successfully.")
    except Exception as e:
        logging.error(f"Error indexing data: {e}")
        st.error("Error indexing data. Please check logs for details.")

    # Streamlit user interface
    st.title("Network Support Chatbot")
    st.write("Ask your questions about Cisco and Mikrotik devices.")

    user_query = st.text_input("Enter your question:")
    if user_query:
        try:
            results = retrieve_similar_documents(user_query, client)
            if results:
                context = "\n\n".join([result['text'] for result in results])
                response = generate_response(context, user_query)
                st.write("Response from GPT-4:")
                st.write(response)
            else:
                st.write("No relevant records found.")
        except Exception as e:
            logging.error(f"Error processing user query: {e}")
            st.error("Error processing user query. Please check logs for details.")

if __name__ == "__main__":
    main()
