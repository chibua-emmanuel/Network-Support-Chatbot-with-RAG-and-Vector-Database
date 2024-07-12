# # # # # # import os
# # # # # # import pypdf
# # # # # # from sentence_transformers import SentenceTransformer
# # # # # # import chromadb
# # # # # # from chromadb.config import Settings

# # # # # # # Function to extract text from PDF
# # # # # # def extract_text_from_pdf(file_path):
# # # # # #     text = ""
# # # # # #     with open(file_path, 'rb') as file:
# # # # # #         reader = pypdf.PdfReader(file)
# # # # # #         for page in reader.pages:
# # # # # #             text += page.extract_text()
# # # # # #     return text

# # # # # # # Extracting text from PDF documents
# # # # # # def preprocess_data(directory):
# # # # # #     data = []
# # # # # #     for root, _, files in os.walk(directory):
# # # # # #         for file in files:
# # # # # #             if file.endswith('.pdf'):
# # # # # #                 file_path = os.path.join(root, file)
# # # # # #                 text = extract_text_from_pdf(file_path)
# # # # # #                 data.append({"text": text, "file_name": file})
# # # # # #     return data

# # # # # # cisco_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Cisco")
# # # # # # mikrotik_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Mikrotik")
# # # # # # all_data = cisco_data + mikrotik_data

# # # # # # # Initialize ChromaDB client and create a collection
# # # # # # try:
# # # # # #     client = chromadb.Client(Settings(persist_directory="chroma_db"))
# # # # # #     collection = client.create_collection("network_support")
# # # # # #     print("Successfully initialized ChromaDB.")
# # # # # # except Exception as e:
# # # # # #     print(f"Error initializing ChromaDB: {e}")
# # # # # #     raise

# # # # # # # Initialize local embedding model
# # # # # # embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')  # Ensure 'device' is specified as 'cpu' if not using GPU

# # # # # # # Preprocess data and index it in ChromaDB
# # # # # # def index_data(data):
# # # # # #     documents = []
# # # # # #     embeddings = []
# # # # # #     metadatas = []
    
# # # # # #     for i, record in enumerate(data):
# # # # # #         embedding = generate_embedding(record["text"])
# # # # # #         documents.append(str(i))
# # # # # #         embeddings.append(embedding)
# # # # # #         metadatas.append(record)
    
# # # # # #     collection.add(documents, embeddings, metadatas)

# # # # # # def generate_embedding(text):
# # # # # #     return embedding_model.encode(text).tolist()

# # # # # # index_data(all_data)


# # # # # # import streamlit as st
# # # # # # import requests
# # # # # # import json

# # # # # # st.title("Network Support Chatbot for Cisco and Mikrotik Devices")

# # # # # # user_query = st.text_input("Enter your question or issue:")

# # # # # # if user_query:
# # # # # #     context = retrieve_relevant_documents(user_query)
# # # # # #     response = generate_response(context, user_query)
# # # # # #     st.write("Response from Llama:")
# # # # # #     st.write(response)

# # # # # #     if st.button("Show Additional Analysis"):
# # # # # #         show_additional_analysis()

# # # # # # def retrieve_relevant_documents(query):
# # # # # #     embedding = generate_embedding(query)
# # # # # #     results = collection.query(query_embeddings=[embedding], n_results=3)
# # # # # #     return "\n".join([result["metadata"]["text"] for result in results["documents"]])

# # # # # # def generate_embedding(text):
# # # # # #     return embedding_model.encode(text).tolist()

# # # # # # def generate_response(context, query):
# # # # # #     url = "http://127.0.0.1:11434/api/v1/generate"
# # # # # #     headers = {"Content-Type": "application/json"}
# # # # # #     data = {
# # # # # #         "context": context,
# # # # # #         "query": query,
# # # # # #         "max_length": 512
# # # # # #     }
# # # # # #     response = requests.post(url, headers=headers, data=json.dumps(data))
# # # # # #     if response.status_code == 200:
# # # # # #         return response.json()["response"]
# # # # # #     else:
# # # # # #         return f"Error: {response.status_code}, {response.text}"

# # # # # # def show_additional_analysis():
# # # # # #     # Example: Analysis based on the dataset
# # # # # #     # This can be expanded as needed
# # # # # #     st.write("Additional analysis can be shown here.")




# # # # # import os
# # # # # import streamlit as st
# # # # # import pypdf
# # # # # from sentence_transformers import SentenceTransformer
# # # # # import weaviate
# # # # # import json
# # # # # import requests
# # # # # from tqdm import tqdm, trange

# # # # # # Function to extract text from PDF
# # # # # def extract_text_from_pdf(file_path):
# # # # #     text = ""
# # # # #     with open(file_path, 'rb') as file:
# # # # #         reader = pypdf.PdfReader(file)
# # # # #         for page in reader.pages:
# # # # #             text += page.extract_text()
# # # # #     return text

# # # # # # Extracting text from PDF documents
# # # # # def preprocess_data(directory):
# # # # #     data = []
# # # # #     for root, _, files in os.walk(directory):
# # # # #         for file in tqdm(files):
# # # # #             if file.endswith('.pdf'):
# # # # #                 file_path = os.path.join(root, file)
# # # # #                 text = extract_text_from_pdf(file_path)
# # # # #                 data.append({"text": text, "file_name": file})
# # # # #     return data

# # # # # # Initialize the Weaviate client and schema
# # # # # def initialize_weaviate():
# # # # #     client = weaviate.Client(
# # # # #         url="https://5xhpqy3xtnlrvxzhvg5cg.c0.europe-west3.gcp.weaviate.cloud",  # Weaviate Cloud URL
# # # # #         auth_client_secret=weaviate.AuthApiKey(api_key="d34PaHllTFk1fPYBlbdbpGuqdNbFWO4PuP88")  # API key
# # # # #     )

# # # # #     # Define the schema
# # # # #     schema = {
# # # # #         "classes": [
# # # # #             {
# # # # #                 "class": "NetworkSupport",
# # # # #                 "properties": [
# # # # #                     {
# # # # #                         "name": "text",
# # # # #                         "dataType": ["text"]
# # # # #                     },
# # # # #                     {
# # # # #                         "name": "file_name",
# # # # #                         "dataType": ["text"]
# # # # #                     }
# # # # #                 ]
# # # # #             }
# # # # #         ]
# # # # #     }

# # # # #     # Create the schema
# # # # #     client.schema.delete_all()
# # # # #     client.schema.create(schema)
# # # # #     return client

# # # # # # Initialize local embedding model
# # # # # embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

# # # # # # Preprocess data and index it in Weaviate
# # # # # def index_data(client, data):
# # # # #     for record in tqdm(data):
# # # # #         embedding = generate_embedding(record["text"])
# # # # #         properties = {
# # # # #             "text": record["text"],
# # # # #             "file_name": record["file_name"]
# # # # #         }
# # # # #         client.data_object.create(
# # # # #             data_object=properties,
# # # # #             class_name="NetworkSupport",
# # # # #             vector=embedding
# # # # #         )

# # # # # def generate_embedding(text):
# # # # #     return embedding_model.encode(text).tolist()

# # # # # # Retrieve top K similar documents based on user query
# # # # # def retrieve_similar_documents(query, client, top_k=3):
# # # # #     embedding = generate_embedding(query)
# # # # #     near_vector = {"vector": embedding, "certainty": 0.7}  # Adjust certainty as needed
# # # # #     results = client.query.get(
# # # # #         class_name="NetworkSupport",
# # # # #         properties=["text", "file_name"]
# # # # #     ).with_near_vector(near_vector).with_limit(top_k).do()
# # # # #     return results["data"]["Get"]["NetworkSupport"]

# # # # # # Generate response using locally running Llama instance
# # # # # def generate_response(context, query):
# # # # #     url = "http://127.0.0.1:11434/api/v1/generate"
# # # # #     headers = {"Content-Type": "application/json"}
# # # # #     data = {
# # # # #         "context": context,
# # # # #         "query": query,
# # # # #         "max_length": 512
# # # # #     }
# # # # #     response = requests.post(url, headers=headers, data=json.dumps(data))
# # # # #     if response.status_code == 200:
# # # # #         return response.json()["response"]
# # # # #     else:
# # # # #         return f"Error: {response.status_code}, {response.text}"

# # # # # # Initialize Weaviate and preprocess data
# # # # # cisco_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Cisco")
# # # # # mikrotik_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Mikrotik")
# # # # # all_data = cisco_data + mikrotik_data

# # # # # client = initialize_weaviate()
# # # # # index_data(client, all_data)

# # # # # # Streamlit user interface
# # # # # st.title("Network Support Chatbot")
# # # # # st.write("Ask your questions about Cisco and Mikrotik devices.")

# # # # # user_query = st.text_input("Enter your question:")
# # # # # if user_query:
# # # # #     results = retrieve_similar_documents(user_query, client)
# # # # #     if results:
# # # # #         context = "\n\n".join([result["text"] for result in results])
# # # # #         response = generate_response(context, user_query)
# # # # #         st.write("Response from Llama:")
# # # # #         st.write(response)
# # # # #     else:
# # # # #         st.write("No relevant records found.")

# # # # # # Option to view embeddings
# # # # # if st.checkbox("View embeddings and metadata"):
# # # # #     if st.button("Retrieve and display embeddings"):
# # # # #         try:
# # # # #             results = client.query.get(
# # # # #                 class_name="NetworkSupport",
# # # # #                 properties=["text", "file_name"]
# # # # #             ).do()
# # # # #             for result in results["data"]["Get"]["NetworkSupport"]:
# # # # #                 st.write(f"Text: {result['text']}")
# # # # #                 st.write(f"File Name: {result['file_name']}")
# # # # #                 st.write("="*80)
# # # # #         except Exception as e:
# # # # #             st.write(f"Error retrieving embeddings: {e}")






# # # # import os
# # # # import streamlit as st
# # # # import pypdf
# # # # from sentence_transformers import SentenceTransformer
# # # # import weaviate
# # # # import json
# # # # import requests
# # # # from tqdm import tqdm, trange

# # # # # Function to extract text from PDF
# # # # def extract_text_from_pdf(file_path):
# # # #     text = ""
# # # #     with open(file_path, 'rb') as file:
# # # #         reader = pypdf.PdfReader(file)
# # # #         for page in reader.pages:
# # # #             text += page.extract_text()
# # # #     return text

# # # # # Extracting text from PDF documents
# # # # def preprocess_data(directory):
# # # #     data = []
# # # #     for root, _, files in os.walk(directory):
# # # #         for file in tqdm(files):
# # # #             if file.endswith('.pdf'):
# # # #                 file_path = os.path.join(root, file)
# # # #                 text = extract_text_from_pdf(file_path)
# # # #                 data.append({"text": text, "file_name": file})
# # # #     return data

# # # # # Initialize the Weaviate client and schema
# # # # def initialize_weaviate():
# # # #     client = weaviate.Client(
# # # #         url="https://5xhpqy3xtnlrvxzhvg5cg.c0.europe-west3.gcp.weaviate.cloud",  # Weaviate Cloud URL
# # # #         auth_client_secret=weaviate.AuthApiKey(api_key="d34PaHllTFk1fPYBlbdbpGuqdNbFWO4PuP88")  # API key
# # # #     )

# # # #     # Define the schema
# # # #     schema = {
# # # #         "classes": [
# # # #             {
# # # #                 "class": "NetworkSupport",
# # # #                 "properties": [
# # # #                     {
# # # #                         "name": "text",
# # # #                         "dataType": ["text"]
# # # #                     },
# # # #                     {
# # # #                         "name": "file_name",
# # # #                         "dataType": ["text"]
# # # #                     }
# # # #                 ]
# # # #             }
# # # #         ]
# # # #     }

# # # #     # Create the schema if it doesn't exist
# # # #     if "NetworkSupport" not in client.schema.get()["classes"]:
# # # #         client.schema.create(schema)
# # # #     return client

# # # # # Initialize local embedding model
# # # # embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

# # # # # Preprocess data and index it in Weaviate
# # # # def index_data(client, data):
# # # #     for record in tqdm(data):
# # # #         embedding = generate_embedding(record["text"])
# # # #         properties = {
# # # #             "text": record["text"],
# # # #             "file_name": record["file_name"]
# # # #         }
# # # #         client.data_object.create(
# # # #             data_object=properties,
# # # #             class_name="NetworkSupport",
# # # #             vector=embedding
# # # #         )

# # # # def generate_embedding(text):
# # # #     return embedding_model.encode(text).tolist()

# # # # # Retrieve top K similar documents based on user query
# # # # def retrieve_similar_documents(query, client, top_k=3):
# # # #     embedding = generate_embedding(query)
# # # #     near_vector = {"vector": embedding, "certainty": 0.7}  # Adjust certainty as needed
# # # #     results = client.query.get(
# # # #         class_name="NetworkSupport",
# # # #         properties=["text", "file_name"]
# # # #     ).with_near_vector(near_vector).with_limit(top_k).do()
# # # #     return results["data"]["Get"]["NetworkSupport"]

# # # # # Generate response using locally running Llama instance
# # # # def generate_response(context, query):
# # # #     url = "http://127.0.0.1:11434/api/v1/generate"
# # # #     headers = {"Content-Type": "application/json"}
# # # #     data = {
# # # #         "context": context,
# # # #         "query": query,
# # # #         "max_length": 512
# # # #     }
# # # #     response = requests.post(url, headers=headers, data=json.dumps(data))
# # # #     if response.status_code == 200:
# # # #         return response.json()["response"]
# # # #     else:
# # # #         return f"Error: {response.status_code}, {response.text}"

# # # # # Initialize Weaviate and preprocess data
# # # # cisco_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Cisco")
# # # # mikrotik_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Mikrotik")
# # # # all_data = cisco_data + mikrotik_data

# # # # client = initialize_weaviate()
# # # # index_data(client, all_data)

# # # # # Streamlit user interface
# # # # st.title("Network Support Chatbot")
# # # # st.write("Ask your questions about Cisco and Mikrotik devices.")

# # # # user_query = st.text_input("Enter your question:")
# # # # if user_query:
# # # #     results = retrieve_similar_documents(user_query, client)
# # # #     if results:
# # # #         context = "\n\n".join([result["text"] for result in results])
# # # #         response = generate_response(context, user_query)
# # # #         st.write("Response from Llama:")
# # # #         st.write(response)
# # # #     else:
# # # #         st.write("No relevant records found.")



# # # import os
# # # import streamlit as st
# # # import pypdf
# # # from sentence_transformers import SentenceTransformer
# # # import weaviate
# # # import json
# # # import requests
# # # from tqdm import tqdm, trange

# # # # Function to extract text from PDF
# # # def extract_text_from_pdf(file_path):
# # #     text = ""
# # #     with open(file_path, 'rb') as file:
# # #         reader = pypdf.PdfReader(file)
# # #         for page in reader.pages:
# # #             text += page.extract_text()
# # #     return text

# # # # Extracting text from PDF documents
# # # def preprocess_data(directory):
# # #     data = []
# # #     for root, _, files in os.walk(directory):
# # #         for file in tqdm(files):
# # #             if file.endswith('.pdf'):
# # #                 file_path = os.path.join(root, file)
# # #                 text = extract_text_from_pdf(file_path)
# # #                 data.append({"text": text, "file_name": file})
# # #     return data

# # # # Initialize the Weaviate client
# # # def initialize_weaviate():
# # #     client = weaviate.Client(
# # #         url="https://5xhpqy3xtnlrvxzhvg5cg.c0.europe-west3.gcp.weaviate.cloud",  # Weaviate Cloud URL
# # #         auth_client_secret=weaviate.AuthApiKey(api_key="d34PaHllTFk1fPYBlbdbpGuqdNbFWO4PuP88")  # API key
# # #     )
# # #     return client

# # # # Initialize local embedding model
# # # embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

# # # # Preprocess data and index it in Weaviate
# # # def index_data(client, data):
# # #     for record in tqdm(data):
# # #         embedding = generate_embedding(record["text"])
# # #         properties = {
# # #             "text": record["text"],
# # #             "file_name": record["file_name"]
# # #         }
# # #         client.data_object.create(
# # #             data_object=properties,
# # #             class_name="NetworkSupport",
# # #             vector=embedding
# # #         )

# # # def generate_embedding(text):
# # #     return embedding_model.encode(text).tolist()

# # # # Retrieve top K similar documents based on user query
# # # def retrieve_similar_documents(query, client, top_k=3):
# # #     embedding = generate_embedding(query)
# # #     near_vector = {"vector": embedding, "certainty": 0.7}  # Adjust certainty as needed
# # #     results = client.query.get(
# # #         class_name="NetworkSupport",
# # #         properties=["text", "file_name"]
# # #     ).with_near_vector(near_vector).with_limit(top_k).do()
# # #     return results["data"]["Get"]["NetworkSupport"]

# # # # Generate response using locally running Llama instance
# # # def generate_response(context, query):
# # #     url = "http://127.0.0.1:11434/api/v1/generate"
# # #     headers = {"Content-Type": "application/json"}
# # #     data = {
# # #         "context": context,
# # #         "query": query,
# # #         "max_length": 512
# # #     }
# # #     response = requests.post(url, headers=headers, data=json.dumps(data))
# # #     if response.status_code == 200:
# # #         return response.json()["response"]
# # #     else:
# # #         return f"Error: {response.status_code}, {response.text}"

# # # # Initialize Weaviate and preprocess data
# # # cisco_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Cisco")
# # # mikrotik_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Mikrotik")
# # # all_data = cisco_data + mikrotik_data

# # # client = initialize_weaviate()
# # # index_data(client, all_data)

# # # # Streamlit user interface
# # # st.title("Network Support Chatbot")
# # # st.write("Ask your questions about Cisco and Mikrotik devices.")

# # # user_query = st.text_input("Enter your question:")
# # # if user_query:
# # #     results = retrieve_similar_documents(user_query, client)
# # #     if results:
# # #         context = "\n\n".join([result["text"] for result in results])
# # #         response = generate_response(context, user_query)
# # #         st.write("Response from Llama:")
# # #         st.write(response)
# # #     else:
# # #         st.write("No relevant records found.")





# # import os
# # import streamlit as st
# # from PyPDF2 import PdfReader
# # from sentence_transformers import SentenceTransformer
# # import weaviate
# # import json
# # import requests
# # import openai
# # from tqdm import tqdm, trange

# # # Initialize OpenAI
# # openai.api_key = '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'  # Replace with your actual OpenAI API key

# # # Function to extract text from PDF
# # def extract_text_from_pdf(file_path):
# #     pdf = PdfReader(file_path)
# #     text = ''
# #     for page in pdf.pages:
# #         text += page.extract_text()
# #     return text

# # # Extracting text from PDF documents
# # def preprocess_data(directory):
# #     data = []
# #     for filename in os.listdir(directory):
# #         if filename.endswith('.pdf'):
# #             file_path = os.path.join(directory, filename)
# #             text = extract_text_from_pdf(file_path)
# #             data.append(text)
# #     return data

# # # Initialize the Weaviate client
# # def initialize_weaviate():
# #     client = weaviate.Client(
# #         url="https://5xhpqy3xtnlrvxzhvg5cg.c0.europe-west3.gcp.weaviate.cloud",  # Weaviate Cloud URL
# #         auth_client_secret=weaviate.AuthApiKey(api_key="d34PaHllTFk1fPYBlbdbpGuqdNbFWO4PuP88")  # API key
# #     )
# #     return client

# # # Initialize local embedding model
# # embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

# # # Preprocess data and index it in Weaviate
# # def index_data(client, data):
# #     for text in tqdm(data, desc="Indexing data"):
# #         embedding = generate_embedding(text)
# #         client.data_object.create(data_object={"text": text}, vector=embedding, class_name="NetworkSupport")

# # def generate_embedding(text):
# #     return embedding_model.encode([text])[0]

# # # Retrieve top K similar documents based on user query
# # def retrieve_similar_documents(query, client, top_k=3):
# #     result = client.query.query.get(
# #         {
# #             "className": "NetworkSupport",
# #             "properties": ["text"],
# #             "filters": {
# #                 "operator": "Similarity",
# #                 "value": generate_embedding(query),
# #                 "type": "vector"
# #             },
# #             "limit": top_k
# #         }
# #     )
# #     return result

# # # Generate response using OpenAI's GPT-4o model
# # def generate_response(context, query):
# #     response = openai.Completion.create(
# #         engine="gpt-4o",
# #         prompt=f"{context}\n{query}",
# #         max_tokens=100,
# #         temperature=0.5,
# #     )
# #     return response.choices[0].text.strip()

# # # Initialize Weaviate and preprocess data
# # cisco_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Cisco")
# # mikrotik_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Mikrotik")
# # all_data = cisco_data + mikrotik_data

# # client = initialize_weaviate()
# # index_data(client, all_data)

# # # Streamlit user interface
# # st.title("Network Support Chatbot")
# # st.write("Ask your questions about Cisco and Mikrotik devices.")

# # user_query = st.text_input("Enter your question:")
# # if user_query:
# #     results = retrieve_similar_documents(user_query, client)
# #     if results:
# #         context = "\n\n".join([result["text"] for result in results])
# #         response = generate_response(context, user_query)
# #         st.write("Response from GPT-4o:")
# #         st.write(response)
# #     else:
# #         st.write("No relevant records found.")



# import os
# import streamlit as st
# from PyPDF2 import PdfReader
# from sentence_transformers import SentenceTransformer
# import weaviate
# import json
# import requests
# import openai
# from tqdm import tqdm, trange

# # Initialize OpenAI
# openai.api_key = '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'  # Replace with your actual OpenAI API key

# # Function to extract text from PDF
# def extract_text_from_pdf(file_path):
#     pdf = PdfReader(file_path)
#     text = ''
#     for page in pdf.pages:
#         text += page.extract_text()
#     return text

# # Extracting text from PDF documents
# def preprocess_data(directory):
#     data = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.pdf'):
#             file_path = os.path.join(directory, filename)
#             text = extract_text_from_pdf(file_path)
#             data.append(text)
#     return data

# # Initialize the Weaviate client
# def initialize_weaviate():
#     client = weaviate.Client(
#         url="https://5xhpqy3xtnlrvxzhvg5cg.c0.europe-west3.gcp.weaviate.cloud",  # Weaviate Cloud URL
#         auth_client_secret=weaviate.AuthApiKey(api_key="d34PaHllTFk1fPYBlbdbpGuqdNbFWO4PuP88")  # API key
#     )
#     return client

# # Initialize local embedding model
# embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

# # Preprocess data and index it in Weaviate
# def index_data(client, data):
#     for text in tqdm(data, desc="Indexing data"):
#         embedding = generate_embedding(text)
#         client.data_object.create(data_object={"text": text}, vector=embedding, class_name="NetworkSupport")

# def generate_embedding(text):
#     return embedding_model.encode([text])[0]

# # Retrieve top K similar documents based on user query
# def retrieve_similar_documents(query, client, top_k=3):
#     result = client.query.get(
#         {
#             "className": "NetworkSupport",
#             "properties": ["text"],
#             "filters": {
#                 "operator": "Similarity",
#                 "value": generate_embedding(query),
#                 "type": "vector"
#             },
#             "limit": top_k
#         }
#     )
#     return result

# # Generate response using OpenAI's GPT-4o model
# def generate_response(context, query):
#     response = openai.Completion.create(
#         engine="gpt-4o",
#         prompt=f"{context}\n{query}",
#         max_tokens=100,
#         temperature=0.5,
#     )
#     return response.choices[0].text.strip()

# # Initialize Weaviate and preprocess data
# cisco_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Cisco")
# mikrotik_data = preprocess_data("/Users/emmanuelchibua/Downloads/Network RAG pdf/Mikrotik")
# all_data = cisco_data + mikrotik_data

# client = initialize_weaviate()
# index_data(client, all_data)

# # Streamlit user interface
# st.title("Network Support Chatbot")
# st.write("Ask your questions about Cisco and Mikrotik devices.")

# user_query = st.text_input("Enter your question:")
# if user_query:
#     results = retrieve_similar_documents(user_query, client)
#     if results:
#         context = "\n\n".join([result["text"] for result in results])
#         response = generate_response(context, user_query)
#         st.write("Response from GPT-4o:")
#         st.write(response)
#     else:
#         st.write("No relevant records found.")




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
openai.api_key = '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'  # Replace with your actual OpenAI API key

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
        url="https://5xhpqy3xtnlrvxzhvg5cg.c0.europe-west3.gcp.weaviate.cloud",  # Weaviate Cloud URL
        auth_client_secret=weaviate.AuthApiKey(api_key="d34PaHllTFk1fPYBlbdbpGuqdNbFWO4PuP88")  # API key
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
