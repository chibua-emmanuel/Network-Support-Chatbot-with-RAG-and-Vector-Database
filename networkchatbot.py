import streamlit as st
import os
import glob
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings

class RAGSystem:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.vector_store = None
        self.chat_history = [{'role': 'system', 'content': "Assistant is a large language model trained by OpenAI."}]

    def pdf_to_langchain_document_parser(self, path: str):
        """Parse and split PDF documents into pages."""
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            st.success(f"Loaded {len(pages)} pages from the PDF document.")
            return pages[:10]  # Limiting to the first 10 pages for now
        except FileNotFoundError:
            st.error("File not found. Please check the path and try again.")
            return []
        except Exception as e:
            st.error(f"Error parsing PDF: {e}")
            return []

    def get_text_chunks(self, text: str):
        """Split text into manageable chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def get_vector_store(self, text_chunks: list):
        """Create a vector store using OpenAI embeddings."""
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
            st.success("Vector store created successfully!")
            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None

    def get_conversational_chain(self):
        """Set up the question-answering conversational chain."""
        prompt_template = """
        You are a knowledgeable subject matter expert in networks and telecommunication specializing in Cisco and Mikrotik devices. Provide helpful and relevant answers based on the given context.
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatOpenAI(model="gpt-4o", openai_api_key=self.openai_api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    def user_input(self, user_question: str):
        """Process user input and return a response."""
        if self.vector_store is None:
            st.error("RAG system has not been initialized. Please run initialize_rag first.")
            return "Initialization required."

        self.chat_history.append({'role': 'user', 'content': user_question})
        
        try:
            docs = self.vector_store.similarity_search(user_question)
            chain = self.get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            self.chat_history.append(response['output_text'])
            return response['output_text']
        except Exception as e:
            st.error(f"Error during question processing: {e}")
            return "An error occurred while processing your request."

    def initialize_rag(self, directory: str):
        """Initialize the RAG system with PDF documents."""
        pdf_docs = self.pdf_to_langchain_document_parser(directory)
        if pdf_docs:
            self.vector_store = self.get_vector_store(pdf_docs)
            return self.vector_store is not None  # Return True if initialization is successful
        else:
            st.error("Failed to initialize RAG system due to PDF loading issues.")
            return False

# Streamlit interface
def main():
    st.title("Network Support RAG System - Cisco & Mikrotik")
    st.write("Developed by Emmanuel Chibua, NU ID: 002799484")

    openai_api_key = "..."
    directory = "/Users/doc.pdf"

    # Initialize RAG System with session state
    if "rag_system" not in st.session_state:
        st.session_state["rag_system"] = RAGSystem(openai_api_key=openai_api_key)
        st.session_state["initialized"] = False

    # Initialize RAG System button
    if st.button("Initialize RAG System"):
        try:
            st.session_state["initialized"] = st.session_state["rag_system"].initialize_rag(directory)
            if st.session_state["initialized"]:
                st.success("RAG system initialized successfully!")
            else:
                st.error("RAG system initialization failed.")
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")

    user_question = st.text_input("Enter your question")

    # Ask Question button
    if st.button("Ask Question"):
        if not st.session_state["initialized"]:
            st.warning("Please initialize the RAG system first.")
        elif not user_question:
            st.warning("Please enter a question.")
        else:
            response = st.session_state["rag_system"].user_input(user_question)
            if response != "Initialization required.":
                st.write("**Answer:**")
                st.write(response)
            else:
                st.warning("Please ensure the RAG system is initialized before asking questions.")

if __name__ == "__main__":
    main()
