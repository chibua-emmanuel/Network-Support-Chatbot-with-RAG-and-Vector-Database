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
    def __init__(self,  openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.vector_store = None
        self.chat_history = [{'role': 'system', 'content': "Assistant is a large language model trained by OpenAI."}]

    def pdf_to_langchain_document_parser(self, path: str):
        all_pages = []
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        return pages[:10]

    def get_text_chunks(self, text: str):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def get_vector_store(self, text_chunks: list):
        embeddings = OpenAIEmbeddings(openai_api_key = self.openai_api_key)
        return FAISS.from_documents(text_chunks, embedding=embeddings)

    def get_conversational_chain(self):
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
        if self.vector_store is None:
            raise Exception("RAG system has not been initialized. Please run initialize_rag first.")
        self.chat_history.append({'role': 'user', 'content': user_question})
        docs = self.vector_store.similarity_search(user_question)
        chain = self.get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        self.chat_history.append(response['output_text'])
        return response

    def initialize_rag(self, directory: str):
        pdf_text = ""
        pdf_docs = self.pdf_to_langchain_document_parser(directory)
        self.vector_store = self.get_vector_store(pdf_docs)

# Streamlit interface
def main():
    st.title("Network Support RAG System- Cisco & Mikrotik")
    st.write("Developed by Emmanuel Chibua, NU ID: ...)

    openai_api_key = "..."
    directory = "/Users.....pdf.pdf"

    rag_system = RAGSystem(openai_api_key = openai_api_key)
    rag_system.initialize_rag(directory)
    st.success("RAG system initialized successfully!")

    user_question = st.text_input("Enter your question")

    if st.button("Ask Question"):
        response = rag_system.user_input(user_question)
        st.write(response)

if __name__ == "__main__":
    main()
