from typing import List
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

def process_documents(file_paths: List[str]) -> List[str]:
    """
    Load and process documents from a list of file paths.

    This function takes a list of file paths to PDF documents, loads them using the PyMuPDFLoader,
    and then splits the text into chunks using the CharacterTextSplitter.

    Parameters:
    - file_paths (List[str]): A list of strings representing the file paths to the PDF documents.

    Returns:
    - List[str]: A list of strings where each string is a chunk of text from the processed documents.
    """
    loaders = [PyMuPDFLoader(path) for path in file_paths]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def generate_response(text, chain):
    """
    Generate a response using a given text and a chain.

    This function takes an input text and passes it to a chain method to invoke a response.
    It then extracts and displays the answer and relevant bike information from the response.

    Parameters:
    - text (str): The input text for generating the response.
    - chain: The chain or model used for generating the response.

    The function does not return a value but instead uses streamlit's info method to display the results.

    """
    # Pass the text as a value to the "input" key in the dictionary
    response = chain.invoke({"input": text})
    st.info(f'The answer is: {response["answer"].answer}')
    st.info(f'The relevant bike is: {response["answer"].bike}')
