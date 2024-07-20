import os

from dotenv import find_dotenv
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv(find_dotenv())


class DataLoader:
    """
    DataLoader class is responsible for loading documents from different sources:
    JSON files for listing and conversation data, and a PDF file for the welcome packet.
    It standardizes the metadata for each document loaded.
    """

    def __init__(self):
        base_path = os.path.join("resources", "sources")
        self.listing_file_path = os.path.join(base_path, "listing.json")
        self.conversation_file_path = os.path.join(base_path, "conversation.json")
        self.welcome_packet_file_path = os.path.join(base_path, "welcome_packet.pdf")

    def load(self):
        """
        Loads documents from various sources and standardizes their metadata.

        Returns:
            documents (list): A list of Document objects with standardized metadata.
        """
        documents = []
        conversation_documents = self._load_conversation()

        for doc in conversation_documents:
            metadata = doc.metadata
            metadata["source_name"] = "conversation"
            documents.append(Document(page_content=doc.page_content, metadata=metadata))

        listing_documents = self._load_listing()
        for doc in listing_documents:
            metadata = doc.metadata
            metadata["source_name"] = "listing"
            documents.append(Document(page_content=doc.page_content, metadata=metadata))

        welcome_packet_documents = self._load_welcome_packet()
        for doc in welcome_packet_documents:
            metadata = doc.metadata
            metadata["source_name"] = "welcome_packet"
            documents.append(Document(page_content=doc.page_content, metadata=metadata))

        return documents

    def _load_conversation(self):
        """
        Loads conversation data from a JSON file.

        Returns:
            list: A list of Document objects loaded from the conversation JSON file.
        """
        return JSONLoader(
            file_path=self.conversation_file_path,
            jq_schema=".result[]",
            text_content=False,
        ).load()

    def _load_listing(self):
        """
        Loads listing data from a JSON file.

        Returns:
            list: A list of Document objects loaded from the listing JSON file.
        """
        return JSONLoader(
            file_path=self.listing_file_path,
            jq_schema=".",
            text_content=False,
        ).load()

    def _load_welcome_packet(self):
        """
        Loads and splits the welcome packet PDF into smaller documents.

        Returns:
            list: A list of Document objects split from the welcome packet PDF.
        """
        welcome_packet_documents = PyPDFLoader(
            file_path=self.welcome_packet_file_path
        ).load()

        chunk_overlap = 20
        chunk_size = 400
        separators = ["\n\n", "\n", ". ", " ", ""]
        character_splitter = RecursiveCharacterTextSplitter(
            separators=separators, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return character_splitter.split_documents(welcome_packet_documents)


class VectorStore:
    """
    VectorStore class manages the storage and retrieval of document embeddings.
    It uses Chroma for vector storage and OpenAI embeddings for embedding generation.
    """

    def __init__(
        self,
        collection_name: str = "lead_ai_ml",
        embeddings: Embeddings = OpenAIEmbeddings(model="text-embedding-3-large"),
    ):
        self.embeddings = embeddings
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        self.retriever = self.vectorstore.as_retriever()

    def add_documents(self, documents: list[Document]):
        """
        Adds documents to the vector store.

        Args:
            documents (list): A list of Document objects to be added to the vector store.
        """
        self.vectorstore.add_documents(documents)

    def get_context_documents(self, query: str) -> list[Document]:
        """
        Retrieves documents similar to the query from the vector store.

        Args:
            query (str): The query string to search for similar documents.

        Returns:
            list: A list of Document objects that are similar to the query.
        """
        return self.vectorstore.similarity_search(query, k=10)

    def get_context_texts(self, query: str) -> list[str]:
        """
        Retrieves the text content of documents similar to the query.

        Args:
            query (str): The query string to search for similar documents.

        Returns:
            list: A list of strings containing the text content of similar documents.
        """
        return [doc.page_content for doc in self.get_context_documents(query)]

    def get_context_string(self, query: str) -> str:
        """
        Retrieves a concatenated string of the text content and source metadata of documents similar to the query.

        Args:
            query (str): The query string to search for similar documents.

        Returns:
            str: A concatenated string of the text content and source metadata of similar documents.
        """
        return "\n\n".join(
            [
                f"text: {doc.page_content}, source: {doc.metadata['source_name']}"
                for doc in self.get_context_documents(query)
            ]
        )
