import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


def get_groq_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.environ["GROQ_API_KEY"],
    )


def get_secondary_groq_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.environ["GROQ_API_KEY"],
    )


def get_gemini_llm(model: str = "gemini-2.5-flash", temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
