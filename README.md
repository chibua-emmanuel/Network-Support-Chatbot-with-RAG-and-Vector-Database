# Network Support Chatbot with RAG and Vector Database

# Network Support RAG System - Cisco & Mikrotik

Developed by Emmanuel Chibua,

## Overview

The Network Support RAG (Retrieval-Augmented Generation) System is designed to assist network engineers and technicians with queries related to Cisco and Mikrotik devices. The system leverages LangChain and OpenAI's GPT-4 to provide accurate and relevant responses based on the context provided in the input documents.

## Features

- **PDF Document Parsing**: Parses and splits PDF documents into manageable chunks.
- **Vector Store Creation**: Converts text chunks into embeddings and stores them in a vector store.
- **Conversational Chain**: Uses a predefined prompt template to generate responses based on user queries.
- **Streamlit Interface**: A user-friendly web interface to interact with the RAG system.

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or higher
- Pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/chibua-emmanuel/Network-Support-Chatbot-with-RAG-and-Vector-Database.git
cd network-support-rag
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

### Required Packages

Here are the main Python packages used in this project:

- `streamlit`
- `PyPDF2`
- `langchain`
- `langchain_openai`

## Usage

### Setting Up OpenAI API Key

Ensure you have your OpenAI API key. You can obtain one by signing up on the [OpenAI website](https://www.openai.com/).

Replace the placeholder API key in the script with your actual OpenAI API key:

```python
openai_api_key = "your-openai-api-key"
```

### Initialize RAG System

Before running the Streamlit app, ensure you have a directory containing the PDF documents to be used by the RAG system. Update the `directory` variable with the path to your PDF documents:

```python
directory = "/path/to/your/pdf/documents"
```

### Running the Streamlit App

To start the Streamlit app, run the following command in your terminal:

```bash
streamlit run networkbot.py
```

This will launch the web interface where you can interact with the RAG system.

## Code Structure

### RAGSystem Class

The `RAGSystem` class handles the core functionality of the RAG system:

- **Initialization**: Sets up the OpenAI API key and initializes the chat history.
- **PDF Parsing**: Converts PDF documents into text chunks.
- **Vector Store**: Creates a vector store from the text chunks using embeddings.
- **Conversational Chain**: Generates responses to user queries using a predefined prompt template.

### Main Function

The `main` function sets up the Streamlit interface and handles user interactions:

- **Title and Developer Information**: Displays the title and developer information.
- **RAG System Initialization**: Initializes the RAG system with the provided API key and PDF directory.
- **User Input**: Accepts user queries and displays the generated responses.

## Example Usage

Here is an example of how to use the RAG system:

1. Ensure the PDF documents are in the specified directory.
2. Run the Streamlit app using the command provided above.
3. Enter your question in the input field and click "Ask Question".
4. The system will process the query and display the response.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your branch.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

