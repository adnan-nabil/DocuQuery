from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import Distance, VectorParams

embeddings = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")
collection_name = "RAG-Project-Ollama"

class document_inject:

    def create(self, pdf_path):
        pdf_path = "/home/user/Documents/VSCode/CSN-RAG/data-manual/operational_manual_XYZ_company.pdf"
        loader = PyPDFLoader(pdf_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(data)

        client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        vector_store = Qdrant(
        client=client, 
        collection_name=collection_name, 
        embeddings=embeddings   
        )

        vector_store.add_documents(chunks)


        print(f"Success! {len(chunks)} chunks stored in Qdrant.")



if __name__ == "__main__":
    ingestor = document_inject()
    path = "/home/user/Documents/VSCode/CSN-RAG/data-manual/operational_manual_XYZ_company.pdf"
    ingestor.create(path)
        