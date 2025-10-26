
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

class RAGDataPreparation:

    DATASET_PATH = "hf://datasets/rishabhsinghjk/HR-QnA-rag-dataspace/Flykite_Airlines_HRP.pdf"

    def __init__(self):
        pass

    def gethrPdfDocLoader(self):
        global DATASET_PATH
        # Define constants for the dataset and output paths
        api = HfApi(token=os.getenv("HF_TOKEN"))
        pdf_loader = PyMuPDFLoader(DATASET_PATH)
        hr_doc = pdf_loader.load()
        print("Dataset loaded successfully.")
        return pdf_loader

    def docChunks(self,pdf_loader):
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
        document_chunks = pdf_loader.load_and_split(text_splitter)
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
        pdf_loader = self.gethrPdfDocLoader()
        document_chunks = self.docChunks(pdf_loader)
        vectorstore_croma = self.createVectorDB(document_chunks)
        return self.createVectorDBRetriever(vectorstore_croma)
