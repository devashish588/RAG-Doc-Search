import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.security import validate_output


load_dotenv()


def build_llm(model: str = "llama-3.1-8b-instant"):
    """Create the Groq chat model used for answer generation."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(model_name=model, groq_api_key=api_key, temperature=0.1)


def build_prompt() -> ChatPromptTemplate:
    """Create a simple RAG prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use only the provided context to answer the user's question. "
                "If the answer is not in the context, say that you do not know.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )


def generate_answer(question: str, docs: List[Document]) -> str:
    """Generate a grounded answer from retrieved context."""
    llm = build_llm()
    prompt = build_prompt()
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return validate_output(answer)
