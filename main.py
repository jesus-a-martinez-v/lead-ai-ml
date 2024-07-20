import gradio as gr
from dotenv import find_dotenv, load_dotenv

from data import DataLoader, VectorStore
from model import QAModel

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Initialize DataLoader and load documents
data_loader = DataLoader()
documents = data_loader.load()

# Initialize VectorStore and add loaded documents
vector_store = VectorStore()
vector_store.add_documents(documents)

# Initialize QAModel with the vector store
qa = QAModel(vector_store=vector_store)

# Define the Gradio interface for question answering
interface = gr.Interface(
    fn=lambda question: qa.answer(question),
    inputs="text",
    outputs="text",
    title="Listing Q&A",
    description="Enter a question.",
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch()
