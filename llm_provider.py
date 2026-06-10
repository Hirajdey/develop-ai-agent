import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        return ChatOpenAI(
            model="gpt-5.5",
            temperature=0,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

    if provider == "ollama":
        return ChatOllama(
            model="llama3.2",
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
