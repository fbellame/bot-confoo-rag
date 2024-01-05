from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os

def embed_doc(directory_path):
    if len(os.listdir(directory_path)) > 0:
        #
        # CHARGER LE FICHIER TEXTE DE DONNEES
        loader = DirectoryLoader(directory_path, glob="**/*.*")

        raw_documents = loader.load()
        
        # SPLITTER EN CHUNK DE 200 TOKENS
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap=0, length_function= len)
        documents = text_splitter.split_documents(raw_documents)

        # EMBEDDED LES DOCUMENTS CHUNKS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        return vectorstore
