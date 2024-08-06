# /**********************************************************************************************************
# Import modules
# /**********************************************************************************************************
# Import general modules in 
import json
import os
import streamlit as st
from dotenv import load_dotenv

# Import modules required for API call
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from functions.helper import generate_response

# Load modules required for RAG system
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load modules required for custom output parsing
from classes.custom_class import AnswerParser
from functions.helper import process_documents
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import PydanticOutputParser

# Load in yaml file for environment level variables
load_dotenv('env.yaml')

# For some reason we need to define these outside of the env file
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ###

# Read the file paths of the documents in
with open("config.json", 'r') as file:
    # Load the contents of the file into a dictionary
    file_paths = json.load(file)

# /**********************************************************************************************************
# Initialise models
# /**********************************************************************************************************
llm = ChatOpenAI(temperature=0.0, model_name=os.getenv("MODEL_NAME"))
embeddings = OpenAIEmbeddings()

# /**********************************************************************************************************
# Process documents and set up database
# /**********************************************************************************************************
# Process documents to use later in OpenAI model
documents = process_documents(file_paths['documents'])
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create vector store
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# /**********************************************************************************************************
# Set prompt template and defined chain
# /**********************************************************************************************************
prompt = ChatPromptTemplate(
    [
        ("system", "You are an AI developed to answer specific questions from customers about bike models"),
        ("user", "Instructions are to answer the question from the customer and then identify which bike is relevant to the answer. The response should be in json format"),
        ("assistant", "Understood, are there any constraints"),
        ("user", "Yes. Please use the specific information found here as a guide: {context}. If you do not know the answer, just say so. If the question is not about a bike, return 'Not a bike' in the bike field"),
        ("user", "Question: {input}")
    ]
)

# Set up a parser and chain
parser = PydanticOutputParser(pydantic_object=AnswerParser)
chain = prompt | llm | parser
chain = create_retrieval_chain(retriever, chain)

# # /**********************************************************************************************************
# Boot up local application
# /**********************************************************************************************************
st.title("🔗 Snyk RAG application 🙌")

with st.form("my_form"):
    
    # Template for the user to input their query
    text = st.text_area("Enter customer question:")
    
    # Create event triggered response
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        generate_response(text, chain)
   
