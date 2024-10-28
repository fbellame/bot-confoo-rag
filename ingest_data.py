from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

def embed_doc(directory_path):
    if len(os.listdir(directory_path)) > 0:
        # CHARGER LE FICHIER TEXTE DE DONNEES
        loader = DirectoryLoader(directory_path, glob="**/*.*")
        raw_documents = loader.load()
        
        # SPLITTER EN CHUNK DE 400 TOKENS AVEC OVERLAP de 100 TOKENS
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100, length_function=len)
        documents = text_splitter.split_documents(raw_documents)

        # EMBEDDED LES DOCUMENTS CHUNKS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Return both vectorstore and extracted references
        return vectorstore
