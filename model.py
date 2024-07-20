import os.path

from dotenv import find_dotenv
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from data import VectorStore

# Load environment variables from a .env file
load_dotenv(find_dotenv())


class QAModel:
    """Question Answering model."""

    def __init__(
        self,
        vector_store: VectorStore,
        prompt_path=os.path.join("resources", "prompts", "system.txt"),
    ):
        """
        Initialize the QAModel with a vector store and a prompt template.

        Args:
            vector_store (VectorStore): The vector store to retrieve context documents.
            prompt_path (str): Path to the system prompt file.
        """
        self.vector_store = vector_store

        # Load the system prompt from a file
        with open(prompt_path, "r") as f:
            system_prompt = f.read()

        # Set up the chat prompt template with system and human messages
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                ("human", "Context: ###\n{context}###\n\nQuestion: ###\n{question}###"),
            ]
        )

        # Set up the language model (LLM) with specified parameters
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def answer(self, query: str) -> str:
        """
        Answer a question by retrieving relevant context and generating a response.

        Args:
            query (str): The question to be answered.

        Returns:
            str: The generated answer to the question.
        """
        # Retrieve the context string relevant to the query from the vector store
        context = self.vector_store.get_context_string(query)

        # Format the messages for the chat template with the question and context
        messages = self.chat_template.format_messages(question=query, context=context)

        # Invoke the language model with the formatted messages to generate a response
        response = self.llm.invoke(messages)

        # Return the content of the generated response
        return response.content
