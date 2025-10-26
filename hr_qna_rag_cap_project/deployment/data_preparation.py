
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from datasets import load_dataset

#Libraries for processing dataframes,text
import json
import tiktoken

# Embedding model and Pdfreader libs
import langchain_community
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

#Chroma DB import
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import pdfplumber

class RAGDataPreparation:

    DATASET_PATH = "hf://datasets/rishabhsinghjk/HR-QnA-rag-dataspace/Flykite_Airlines_HRP.pdf"
    dataset_space = "rishabhsinghjk/HR-QnA-rag-dataspace"

    def __init__(self):
        pass

    def getPDFFileLoader(self):
        dataset = load_dataset(RAGDataPreparation.dataset_space)
        pdf_data = dataset["train"][0]["pdf"]
        # Create a temporary file path
        temp_pdf_path = "Flykite_Airlines_HRP.pdf"

        # If pdf_data is binary content
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_data)
        
        pdf_loader = PyMuPDFLoader(temp_pdf_path)
        return pdf_loader

    def getPDFFileContent(self):
        dataset = load_dataset(RAGDataPreparation.dataset_space)
        pdf_data = dataset["train"][0]["pdf"]
        # Create a temporary file path
        full_content = []
        for page_num, page in enumerate(pdf_data.pages):
            # Extract text
            text = page.extract_text()
            if text:
                full_content.append(text)

            content = "\n\n".join(full_content)
        return content

    def gethrPdfDocLoader(self):        
        # Define constants for the dataset and output paths
        api = HfApi(token=os.getenv("HF_TOKEN"))
        pdf_loader = self.getPDFFileLoader()
        hr_doc = pdf_loader.load()
        print("Dataset loaded successfully.")
        return pdf_loader

    def docChunks(self,pdf_content):
        #Defining the chunk size for creation of embedings.
        chunk_size=256

        #Defining the overlap size (ideally between 15-30)
        chunk_overlap=16

        # Initializing Text splitter class object for creating chunks from documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='cl100k_base',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        #Spliting and creating chunks list
        chunks = text_splitter.split_text(pdf_content)
        
        document_chunks = [
            Document(
                page_content=chunk,
                metadata={"page_number": ("page"+ str(i+1)), "source": "Flykite_Airlines_HRP.pdf"}
            )
            for i, chunk in enumerate(chunks)
        ]

        return document_chunks

    def createVectorDB(self,document_chunks):

        #Now we will use these chunks to create vector embedings and save them in a vector database.

        #Creating Embeding model for initiliazing vector DB
        embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')
        embedding_1 = embedding_model.embed_query(document_chunks[0].page_content)
        print("Dimension of the embedding vector ",len(embedding_1))

        # Create Persistent Data directory
        out_dir = 'airlines_cromadb'

        if not os.path.exists(out_dir):
          os.makedirs(out_dir)

        # Use Chroma Vector DB to store document chunks
        vectorstore_croma = Chroma.from_documents(
            document_chunks,
            embedding_model,
            persist_directory=out_dir
        )

        vectorstore_croma = Chroma(persist_directory=out_dir,embedding_function=embedding_model)
        return vectorstore_croma

    def createVectorDBRetriever(self,vectorstore_croma):
        #Defining retriever object for Chroma DB with similarity search
        retriever_cromadb = vectorstore_croma.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 2}
        )
        return retriever_cromadb    

    def getVectorDBRetriever(self):
        pdf_content = self.getPDFFileContent()
        document_chunks = self.docChunks(pdf_content)
        vectorstore_croma = self.createVectorDB(document_chunks)
        return self.createVectorDBRetriever(vectorstore_croma)
